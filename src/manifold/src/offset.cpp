// Copyright 2023 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <random>

#include "csg_tree.h"
#include "glm/gtx/norm.hpp"
#include "impl.h"

namespace {
using namespace manifold;
using namespace glm;

// basic idea of the algorithm:
// We check which faces can intersect after offsetting.
// Say face f1 intersects with a set of faces F1,
// we find the list of faces F1' consisting of faces that are not connected to
// f1 in F1.
// We apply cuts that separate f1 from all faces in F1'.
// And then we do offsetting for each component, and union them together.
// This should be much more efficient than doing convex decomposition.

// this approximates the sweeped volume when a face is offsetted
struct FaceOffsetMortonBox {
  VecView<const Halfedge> halfedge;
  VecView<const vec3> vertPos;
  VecView<vec3> vertPosProjected;
  VecView<const vec3> triNormal;
  VecView<const vec3> vertNormal;
  const Box bBox;
  // positive offset only
  float offset;

  void operator()(thrust::tuple<uint32_t&, Box&, int> inout) {
    uint32_t& mortonCode = thrust::get<0>(inout);
    Box& faceBox = thrust::get<1>(inout);
    int face = thrust::get<2>(inout);
    for (const int i : {0, 1, 2}) {
      const Halfedge& edge = halfedge[3 * face + i];
      vec3 pos = vertPos[edge.startVert];
      faceBox.Union(pos);

      vec3 projected = vertNormal[edge.startVert];
      // overapproximate vertex offset by forming a cone that enclose the sphere
      // with r = offset
      projected = normalize(projected);
      projected /= dot(triNormal[face], projected);
      projected *= offset;
      vertPosProjected[edge.startVert] = pos + projected;
      faceBox.Union(vertPosProjected[edge.startVert]);
    }
    mortonCode = MortonCode(faceBox.Center(), bBox);
  }
};

std::array<vec3, 5> triPrismNormals(const std::array<vec3, 6>& pos) {
  // note: pos[i] is contained in face i,
  // and normal is pointing outward
  return {
      normalize(cross(pos[2] - pos[0], pos[1] - pos[0])),
      normalize(cross(pos[4] - pos[1], pos[0] - pos[1])),
      normalize(cross(pos[5] - pos[2], pos[1] - pos[2])),
      normalize(cross(pos[3] - pos[0], pos[2] - pos[0])),
      normalize(cross(pos[4] - pos[3], pos[5] - pos[3])),
  };
}

// separating axis test
// note that we consider touching as overlapping
bool triPrismOverlap(const std::array<vec3, 6>& a,
                     const std::array<vec3, 5>& aNorms,
                     const std::array<vec3, 6>& b,
                     const std::array<vec3, 5>& bNorms, const float precision) {
  // test A faces
  for (int i : {0, 1, 2, 3, 4}) {
    if (std::all_of(countAt(0), countAt(6), [&](int j) {
          return dot(b[j] - a[i], aNorms[i]) > precision;
        }))
      return false;
  }

  // test B faces
  for (int i : {0, 1, 2, 3, 4}) {
    if (std::all_of(countAt(0), countAt(6), [&](int j) {
          return dot(a[j] - b[i], bNorms[i]) > precision;
        }))
      return false;
  }

  // 0-2: edges in the triangle pos[0:3]
  // 6-8: edges in the triangle pos[3:6]
  // 3-5: edges connecting the two triangles
  auto getEdge = [](const std::array<vec3, 6>& verts, int index) {
    if (index < 3 || index > 5) {
      int offset = index > 2 ? 3 : 0;
      return verts[(index + 1) % 3 + offset] - verts[index % 3 + offset];
    }
    index -= 3;
    return verts[index + 3] - verts[index];
  };

  // edges test, there are 9*9 combinations...
  // see if we can disable some
  std::array<bool, 5> disableA = {false};
  std::array<bool, 5> disableB = {false};
  for (int i : {0, 1, 2}) {
    // in case the two triangles are parallel
    if (distance2(a[(i + 1) % 3] - a[i], a[(i + 1) % 3 + 3] - a[i + 3]) <
        kTolerance)
      disableA[i] = true;
    if (distance2(b[(i + 1) % 3] - b[i], b[(i + 1) % 3 + 3] - b[i + 3]) <
        kTolerance)
      disableB[i] = true;
  }
  // in case it is really a prism
  if (distance2(a[3] - a[0], a[4] - a[1]) < kTolerance) disableA[3] = true;
  if (distance2(a[4] - a[1], a[5] - a[2]) < kTolerance) disableA[4] = true;
  if (distance2(b[3] - b[0], b[4] - b[1]) < kTolerance) disableB[3] = true;
  if (distance2(b[4] - b[1], b[5] - b[2]) < kTolerance) disableB[4] = true;

  for (int i = 0; i < 9; ++i) {
    if (i < 5 && disableA[i]) continue;
    vec3 edgeA = normalize(getEdge(a, i));
    for (int j = 0; j < 9; ++j) {
      if (j < 5 && disableB[j]) continue;
      vec3 edgeB = normalize(getEdge(b, j));
      // they are basically parallel
      if (abs(dot(edgeA, edgeB)) >= 1 - kTolerance) continue;
      vec3 axis = cross(edgeA, edgeB);
      float aMin = std::numeric_limits<float>::max();
      float aMax = std::numeric_limits<float>::min();
      float bMin = std::numeric_limits<float>::max();
      float bMax = std::numeric_limits<float>::min();

      for (int k = 0; k < 6; ++k) {
        float aProj = dot(a[k], axis);
        float bProj = dot(b[k], axis);
        aMin = std::min(aMin, aProj);
        aMax = std::max(aMax, aProj);
        bMin = std::min(bMin, bProj);
        bMax = std::max(bMax, bProj);
      }
      if (aMax + precision <= bMin || aMin - precision >= bMax) return false;
    }
  }
  return true;
}

struct Halfspace {
  template <const int N>
  int test(const std::array<vec3, N>& pts) {
    int prevResult = 0;
    for (int i = 0; i < N; ++i) {
      float d = dot(pts[i], normal);
      int result;
      if (d >= signedDistance)
        result = 1;
      else if (d <= signedDistance)
        result = -1;
      else
        return 0;
      if (i > 0 && result != prevResult) return 0;
      prevResult = result;
    }
    return prevResult;
  }

  template <const int N>
  bool coplanar(const std::array<vec3, N>& pts) {
    for (int i = 0; i < N; ++i) {
      float d = dot(pts[i], normal);
      if (abs(signedDistance - d) > kTolerance) return false;
    }
    return true;
  }

  vec3 normal;
  float signedDistance;
};

// using separating axis theorem to find a cut
Halfspace findSeparatingPlane(const std::array<vec3, 3>& a,
                              const std::array<vec3, 3>& b) {
  // see if the face itself can act as a separating plane
  vec3 aNorm = normalize(cross(a[2] - a[0], a[1] - a[0]));
  float da = dot(a[0], aNorm);
  int r = (Halfspace{aNorm, da}).test<3>(b);
  if (r != 0) return {aNorm, da};

  vec3 bNorm = normalize(cross(b[2] - b[0], b[1] - b[0]));
  float db = dot(b[0], bNorm);
  r = (Halfspace{bNorm, db}).test<3>(a);
  if (r != 0) return {bNorm, db};

  for (int i : {0, 1, 2}) {
    for (int j : {0, 1, 2}) {
      vec axis = normalize(cross(b[(j + 1) % 3] - b[j], a[(i + 1) % 3] - a[i]));
      float aMin = std::numeric_limits<float>::max();
      float aMax = std::numeric_limits<float>::min();
      float bMin = std::numeric_limits<float>::max();
      float bMax = std::numeric_limits<float>::min();

      for (int k : {0, 1, 2}) {
        float aProj = dot(a[k], axis);
        float bProj = dot(b[k], axis);
        aMin = std::min(aMin, aProj);
        aMax = std::max(aMax, aProj);
        bMin = std::min(bMin, bProj);
        bMax = std::max(bMax, bProj);
      }
      if (aMax <= bMin) return {axis, aMax};
      if (aMin >= bMax) return {axis, aMin};
    }
  }

  for (vec3 p : a) printf("[%.3f, %.3f, %.3f], ", p.x, p.y, p.z);
  printf("\n");
  for (vec3 p : b) printf("[%.3f, %.3f, %.3f], ", p.x, p.y, p.z);
  printf("\n");
  throw std::runtime_error(
      "no separating plane found, is there a self intersection?");
}

void recursiveCut(Manifold part, const VecView<const Halfedge> halfedges,
                  const VecView<const vec3> vertPos, SparseIndices& collision,
                  std::vector<Manifold>& results) {
  if (part.IsEmpty()) return;
  if (collision.size() == 0) {
    results.push_back(part);
    return;
  }

  if (collision.size() > 128) {
    std::mt19937 rand;
    std::uniform_int_distribution<int> dist(0, collision.size() - 1);
    std::array<int, 16> idx;
    std::array<int, 16> counts;
    for (int i = 0; i < 16; i++) {
      idx[i] = dist(rand);
      std::array<vec3, 3> tri1;
      std::array<vec3, 3> tri2;
      for (int j : {0, 1, 2}) {
        tri1[j] =
            vertPos[halfedges[3 * collision.Get(idx[i], false) + j].startVert];
        tri2[j] =
            vertPos[halfedges[3 * collision.Get(idx[i], true) + j].startVert];
      }
      Halfspace h = findSeparatingPlane(tri1, tri2);
      counts[i] = count_if(
          autoPolicy(collision.size()), countAt(0), countAt(collision.size()),
          [&](int j) {
            std::array<vec3, 3> tri1;
            std::array<vec3, 3> tri2;
            for (int k : {0, 1, 2}) {
              tri1[k] =
                  vertPos[halfedges[3 * collision.Get(j, false) + k].startVert];
              tri2[k] =
                  vertPos[halfedges[3 * collision.Get(j, true) + k].startVert];
            }
            if (h.coplanar<3>(tri1)) return false;
            if (h.coplanar<3>(tri2)) return false;
            int t1 = h.test<3>(tri1);
            int t2 = h.test<3>(tri2);
            return t1 == 0 || t2 == 0 || t1 == t2;
          });
    }
    int id = idx[std::distance(counts.begin(),
                               std::min_element(counts.begin(), counts.end()))];
    VecView<int64_t> c = collision.AsVec64();
    std::swap(c[0], c[id]);
  }

  std::array<vec3, 3> tri1;
  std::array<vec3, 3> tri2;
  for (int i : {0, 1, 2}) {
    tri1[i] = vertPos[halfedges[3 * collision.Get(0, false) + i].startVert];
    tri2[i] = vertPos[halfedges[3 * collision.Get(0, true) + i].startVert];
  }

  Halfspace h = findSeparatingPlane(tri1, tri2);
  SparseIndices a, b;
  for (int i = 1; i < collision.size(); ++i) {
    for (int j : {0, 1, 2}) {
      tri1[j] = vertPos[halfedges[3 * collision.Get(i, false) + j].startVert];
      tri2[j] = vertPos[halfedges[3 * collision.Get(i, true) + j].startVert];
    }

    // in case our cut is coplanar to the triangle, we consider the triangle is
    // being eliminated
    if (h.coplanar<3>(tri1)) continue;
    if (h.coplanar<3>(tri2)) continue;

    int t1 = h.test<3>(tri1);
    int t2 = h.test<3>(tri2);
    // check if the triangles are put into two halves
    if (t1 == t2) {
      if (t1 != -1) a.Add(collision.Get(i, false), collision.Get(i, true));
      if (t1 != 1) b.Add(collision.Get(i, false), collision.Get(i, true));
    } else if (t1 == 0) {
      if (t2 == 1)
        a.Add(collision.Get(i, false), collision.Get(i, true));
      else
        b.Add(collision.Get(i, false), collision.Get(i, true));
    } else if (t2 == 0) {
      if (t1 == 1)
        a.Add(collision.Get(i, false), collision.Get(i, true));
      else
        b.Add(collision.Get(i, false), collision.Get(i, true));
    }
  }

  collision.Resize(0);
  std::pair<Manifold, Manifold> pair =
      part.SplitByPlane(h.normal, h.signedDistance);
  // TODO: veryfy this part
  recursiveCut(pair.first, halfedges, vertPos, a, results);
  recursiveCut(pair.second, halfedges, vertPos, b, results);
}

};  // namespace

namespace manifold {
// note that the morton code here are in general different from that of faces
std::vector<Manifold> Manifold::OffsetDecomposition(float offset) const {
  auto pImpl_ = GetCsgLeafNode().GetImpl();
  Vec<int> new2old(NumTri());
  Vec<glm::vec3> vertPosProjected(NumVert());
  SparseIndices collisionIndices;
  float precision = pImpl_->precision_;
  if (precision <= 0) precision = kTolerance;
  // broad phase
  {
    Vec<Box> boxes(NumTri(), {});
    Vec<uint32_t> mortonCodes(NumTri());
    for_each_n(autoPolicy(NumTri()),
               zip(mortonCodes.begin(), boxes.begin(), countAt(0)), NumTri(),
               FaceOffsetMortonBox(
                   {pImpl_->halfedge_.cview(), pImpl_->vertPos_.cview(),
                    vertPosProjected, pImpl_->faceNormal_.cview(),
                    pImpl_->vertNormal_.cview(), pImpl_->bBox_, offset}));

    auto policy = autoPolicy(new2old.size());
    sequence(policy, new2old.begin(), new2old.end());
    sort_by_key(policy, mortonCodes.begin(), mortonCodes.end(),
                new2old.begin());
    // permute boxes
    Vec<Box> tmp(std::move(boxes));
    boxes.resize(new2old.size());
    gather(autoPolicy(new2old.size()), new2old.begin(), new2old.end(),
           tmp.begin(), boxes.begin());

    Collider collider(boxes, mortonCodes);
    collisionIndices = collider.Collisions<true, false, Box>(boxes);
  }

  {
    auto getPoints = [&](int face) {
      std::array<vec3, 6> output;
      for (int i : {0, 1, 2}) {
        int vert = pImpl_->halfedge_[3 * face + i].startVert;
        output[i] = pImpl_->vertPos_[vert];
        output[i + 3] = vertPosProjected[vert];
      }
      return output;
    };

    auto collisions = collisionIndices.AsVec64();
    auto iter = remove_if<decltype(collisions.begin())>(
        autoPolicy(collisionIndices.size() * 20), collisions.begin(),
        collisions.end(), [&](int64_t pair) {
          int face1 = new2old[(pair >> 32) & ((1ul << 32) - 1)];
          int face2 = new2old[pair & ((1ul << 32) - 1)];
          if (face1 > face2) return true;

          std::array<vec3, 6> face1Pts = getPoints(face1);
          std::array<vec3, 6> face2Pts = getPoints(face2);
          bool shareVertex = false;
          for (int j : {0, 1, 2})
            for (int k : {0, 1, 2})
              if (distance2(face1Pts[j], face2Pts[k]) <= precision) return true;
          std::array<vec3, 5> face1Norms = triPrismNormals(face1Pts);
          std::array<vec3, 5> face2Norms = triPrismNormals(face2Pts);
          return !triPrismOverlap(face1Pts, face1Norms, face2Pts, face2Norms,
                                  precision);
        });
    collisionIndices.Resize(std::distance(collisions.begin(), iter));
    // remap indices
    for_each(autoPolicy(collisionIndices.size()), countAt(0),
             countAt(collisionIndices.size()), [&](int i) {
               collisionIndices.Set(i, new2old[collisionIndices.Get(i, false)],
                                    new2old[collisionIndices.Get(i, true)]);
             });
  }

  // compute decomposition
  {
    std::vector<Manifold> results;
    recursiveCut(*this, pImpl_->halfedge_, pImpl_->vertPos_, collisionIndices,
                 results);
    return results;
  }
}
}  // namespace manifold
