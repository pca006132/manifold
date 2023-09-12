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
#include <set>

#include "QuickHull.hpp"
#include "csg_tree.h"
#include "glm/gtx/norm.hpp"
#include "impl.h"

namespace {
constexpr float kCollinear = 0.00001;

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

// Reference: A 3D surface offset method for STL-format models
vec3 averageNormal(const vec3& a, const vec3& b) {
  float ab = dot(a, b);
  if (ab >= 1 - kCollinear) return a;

  mat2 m = {1, ab,  //
            ab, 1};
  mat2 invM = inverse(m);
  vec2 weights = invM * vec2(1, 1);
  return normalize(a * weights.x + b * weights.y);
}

vec3 averageNormal(const vec3& a, const vec3& b, const vec3& c) {
  float ab = dot(a, b);
  float ac = dot(a, c);
  float bc = dot(b, c);

  if (ab >= 1 - kCollinear) return averageNormal(b, c);
  if (ac >= 1 - kCollinear) return averageNormal(a, b);
  if (bc >= 1 - kCollinear) return averageNormal(a, c);

  mat3 m = {1,  ab, ac,  //
            ab, 1,  bc,  //
            ac, bc, 1};
  mat3 invM = inverse(m);
  vec3 weights = invM * vec3(1, 1, 1);
  return normalize(a * weights.x + b * weights.y + c * weights.z);
}

// reference: Offset triangular mesh using multiple normal vectors of a vertex
void MultiNormalOffset(const Manifold::Impl& impl, float offset) {
  Vec<bool> processed(impl.NumVert(), false);
  int numVert = impl.NumVert();
  int numHalfedge = impl.halfedge_.size();
  for (int i = 0; i < numHalfedge; ++i) {
    // skip newly added vertices
    if (impl.halfedge_[i].startVert >= numVert) continue;
    // skip processed vertices
    if (processed[impl.halfedge_[i].startVert]) continue;
    processed[impl.halfedge_[i].startVert] = true;

    std::vector<int> edges;
    std::vector<vec3> normals;
    std::vector<int> normalEdge;
    auto addNormal = [&](const vec3 normal) {
      for (int j = 0; j < normals.size(); ++j)
        if (dot(normal, normals[j]) >= 1 - kCollinear) return j;
      normals.push_back(normal);
      return static_cast<int>(normals.size() - 1);
    };
    std::vector<std::pair<int, int>> edgeNormalMap;
    // orbit startVert
    int current = i;
    do {
      edges.push_back(current);
      int a = addNormal(impl.faceNormal_[current / 3]);
      int b = addNormal(
          impl.faceNormal_[impl.halfedge_[current].pairedHalfedge / 3]);
      edgeNormalMap.push_back({a, b});
      if (a != b) normalEdge.push_back(current);
      current = NextHalfedge(impl.halfedge_[current].pairedHalfedge);
    } while (current != i);

    std::vector<int> normalMap(normals.size());
    // identity mapping by default
    sequence(ExecutionPolicy::Seq, normalMap.begin(), normalMap.end());
    if (normals.size() == 1) {
      // just a plane
      continue;
    }

    // handle concave normals
    int originalNormalSize = normals.size();
    std::vector<int> toRemove(originalNormalSize);
    sequence(ExecutionPolicy::Seq, toRemove.begin(), toRemove.end());
    for (int j = 0; j < originalNormalSize; ++j) {
      vec3 a = normals[j];
      vec3 b = normals[(j + 1) % originalNormalSize];
      vec3 out = impl.vertPos_[impl.halfedge_[normalEdge[j]].endVert] -
                 impl.vertPos_[impl.halfedge_[normalEdge[j]].startVert];
      if (dot(cross(a, b), out) <= 0) {
        // concave
        vec3 newNormal = averageNormal(a, b);
        int additional = -1;
        if (originalNormalSize >= 3) {
          // try after
          vec3 c = normals[(j + 2) % originalNormalSize];
          vec3 c1 = normals[(j + originalNormalSize - 1) % originalNormalSize];
          Halfedge e = impl.halfedge_[normalEdge[(j + 2) % originalNormalSize]];
          Halfedge e1 = impl.halfedge_[normalEdge[(j + originalNormalSize - 1) %
                                                  originalNormalSize]];
          out = impl.vertPos_[e.endVert] - impl.vertPos_[e.startVert];
          vec3 out1 = impl.vertPos_[e.endVert] - impl.vertPos_[e.startVert];
          if (dot(cross(b, c), out) <= 0) {
            newNormal = averageNormal(a, b, c);
            additional = (j + 2) % originalNormalSize;
          } else if (dot(cross(c1, a), out1) <= 0) {
            newNormal = averageNormal(c1, a, b);
            additional = (j + originalNormalSize - 1) % originalNormalSize;
          }
        }
        normalMap[j] = addNormal(newNormal);
        normalMap[(j + 1) % originalNormalSize] = normalMap[j];
        if (additional != -1) {
          normalMap[additional] = normalMap[j];
        }
      }
    }
    for (int j = 0; j < originalNormalSize; ++j)
      if (normalMap[j] < originalNormalSize) toRemove[normalMap[j]] = -1;

    std::sort(toRemove.begin(), toRemove.end());
    toRemove.erase(toRemove.begin(),
                   std::upper_bound(toRemove.begin(), toRemove.end(), -1));
    for (auto iter = toRemove.rbegin(); iter != toRemove.rend(); ++iter) {
      normals.erase(normals.begin() + *iter);
    }
    for (int j = 0; j < originalNormalSize; ++j) {
      int oldIndex = normalMap[j];
      int count = std::count_if(toRemove.begin(), toRemove.end(),
                                [oldIndex](int x) { return x < oldIndex; });
      normalMap[j] -= count;
    }
    for (std::pair<int, int>& edge : edgeNormalMap) {
      edge.first = normalMap[edge.first];
      edge.second = normalMap[edge.second];
    }

    if (normals.size() >= 3) {
      // project them onto a plane with normal = vertex normal,
      // and x axis defined by edge normal furthest from vertex normal
      // and order them in CW direction.
      vec3 vertNormal = impl.vertNormal_[impl.halfedge_[i].startVert];

      float minAngle = 1.0;
      int minIndex = -1;
      for (int j = 0; j < normals.size(); ++j) {
        float angle = abs(dot(normals[j], vertNormal));
        if (angle < minAngle) minAngle = angle;
        minIndex = j;
      }
      vec3 xaxis = normalize(normals[minIndex] -
                             vertNormal * dot(normals[minIndex], vertNormal));
      vec3 yaxis = cross(vertNormal, xaxis);
      Vec<float> angles;
      Vec<int> normalMap(normals.size());
      sequence(ExecutionPolicy::Seq, normalMap.begin(), normalMap.end());
      for (const vec3& normal : normals) {
        angles.push_back(atan2(dot(normal, yaxis), dot(normal, xaxis)));
      }
      std::stable_sort(zip(angles.begin(), normals.begin(), normalMap.begin()),
                       zip(angles.end(), normals.end(), normalMap.end()),
                       [](const thrust::tuple<float, vec3, int>& a,
                          const thrust::tuple<float, vec3, int>& b) {
                         return thrust::get<0>(a) > thrust::get<0>(b);
                       });
      int lastSecond = -1;
      int s = normalMap.size();

      for (std::pair<int, int>& pair : edgeNormalMap) {
        pair.first = normalMap[pair.first];
        pair.second = normalMap[pair.second];
        if (pair.second == lastSecond) std::swap(pair.first, pair.second);
#if MANIFOLD_DEBUG
        ASSERT(lastSecond == -1 || lastSecond == pair.first ||
                   (lastSecond + 1) % s == pair.first,
               logicErr, "expects monotone normal angle");
#endif
        lastSecond = pair.second;
      }
#if MANIFOLD_DEBUG
      // check normal projection forms a convex polygon
      for (int j = 0; j < normals.size(); ++j) {
        vec3 a = normals[(j + 1) % normals.size()] - normals[j];
        vec3 b = normals[(j + 2) % normals.size()] -
                 normals[(j + 1) % normals.size()];
        ASSERT(dot(cross(a, b), vertNormal) <= 0, logicErr,
               "expects convex projection");
      }
#endif
    }

    // impl.vertPos_[impl.halfedge_[i].startVert] += normals.front() * offset;

    // two special cases
    // 1. edge v->u contributes two vertices v1 v2
    //    new triangle u v1 v2, u -> v2 and v1 -> u
    // 2. adjacent edges v -> u1, v -> u2 contribute to different vertices
    //    we split the triangle v u2 u1 into two triangles
  }
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
    stable_sort(policy, zip(mortonCodes.begin(), new2old.begin()),
                zip(mortonCodes.end(), new2old.end()),
                [](const thrust::tuple<uint32_t, int>& a,
                   const thrust::tuple<uint32_t, int>& b) {
                  return thrust::get<0>(a) < thrust::get<0>(b);
                });
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

Manifold Manifold::NaiveOffset(float offset) const {
  auto pImpl_ = GetCsgLeafNode().GetImpl();
  MultiNormalOffset(*pImpl_, offset);
  return *this;
}
}  // namespace manifold
