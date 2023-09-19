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
#include <unordered_map>

#include "QuickHull.hpp"
#include "csg_tree.h"
#include "glm/ext.hpp"
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
      vec3 axis =
          normalize(cross(b[(j + 1) % 3] - b[j], a[(i + 1) % 3] - a[i]));
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
dvec3 averageNormal(const dvec3& a, const dvec3& b) {
  double ab = dot(a, b);
  if (ab >= 1 - kCollinear) return a;

  dmat2 m = {1, ab,  //
             ab, 1};
  dmat2 invM = inverse(m);
  dvec2 weights = invM * vec2(1, 1);
  return a * weights.x + b * weights.y;
}

dvec3 averageNormal(const dvec3& a, const dvec3& b, const dvec3& c) {
  double ab = dot(a, b);
  double ac = dot(a, c);
  double bc = dot(b, c);

  if (ab >= 1 - kCollinear) return averageNormal(b, c);
  if (ac >= 1 - kCollinear) return averageNormal(a, b);
  if (bc >= 1 - kCollinear) return averageNormal(a, c);

  dmat3 m = {1,  ab, ac,  //
             ab, 1,  bc,  //
             ac, bc, 1};
  dmat3 invM = inverse(m);
  dvec3 weights = invM * vec3(1, 1, 1);
  return a * weights.x + b * weights.y + c * weights.z;
}

// reference: Offset triangular mesh using multiple normal vectors of a vertex
void MultiNormalOffset(Manifold::Impl& impl, double offset) {
  std::vector<std::vector<int>> vertEdges(impl.NumVert());
  std::vector<std::vector<std::pair<dvec3, double>>> vertNormals(
      impl.NumVert());
  std::vector<std::pair<int, int>> edgeNormals(impl.halfedge_.size(), {-1, -1});

  auto fn = [&](int i) {
    int startVert = impl.halfedge_[i].startVert;
    std::vector<int> edges;
    std::vector<std::pair<dvec3, double>> normals;
    auto addNormal = [&](dvec3 normal) {
      double factor = length(normal);
      normal = normalize(normal);
      for (int j = 0; j < normals.size(); ++j)
        if (dot(normal, normals[j].first) >= 1 - kCollinear) return j;
      normals.push_back({normal, factor});
      return static_cast<int>(normals.size() - 1);
    };
    // entry[i] is the edge where the two neighboring faces have normals
    // normals[i] and normals[i+1]
    std::vector<int> normalEdge;
    std::vector<std::pair<int, int>> edgeNormalsLocal;
    // orbit startVert, get all outgoing halfedges
    int current = i;
    do {
      edges.push_back(current);
      int a = addNormal(impl.faceNormal_[current / 3]);
      int b = addNormal(
          impl.faceNormal_[impl.halfedge_[current].pairedHalfedge / 3]);
      edgeNormalsLocal.push_back({a, b});
      if (a != b) normalEdge.push_back(current);
      current = NextHalfedge(impl.halfedge_[current].pairedHalfedge);

      // we only process if the edge is the smallest out-going edge
      if (current < i) return;
    } while (current != i);

    ASSERT(normals.size() == 1 || normalEdge.size() == normals.size(), logicErr,
           "normalEdge().size() should be equal to normals.size()");

    int originalNormalSize = normals.size();
    // handle concave normals
    if (originalNormalSize >= 2) {
      std::vector<int> normalMap(normals.size());
      sequence(ExecutionPolicy::Seq, normalMap.begin(), normalMap.end());
      std::vector<std::set<int>> consecutives;
      for (int j = 0; j < originalNormalSize; ++j) consecutives.push_back({j});

      for (int j = 0; j < originalNormalSize; ++j) {
        dvec3 a = normals[j].first;
        dvec3 b = normals[(j + 1) % originalNormalSize].first;
        dvec3 out = impl.vertPos_[impl.halfedge_[normalEdge[j]].endVert] -
                    impl.vertPos_[impl.halfedge_[normalEdge[j]].startVert];
        if (dot(cross(a, b), out) <= kCollinear) {
          // concave
          dvec3 newNormal = averageNormal(a, b);
          int additional = -1;
          if (originalNormalSize >= 3) {
            dvec3 c = normals[(j + 2) % originalNormalSize].first;
            dvec3 c1 =
                normals[(j + originalNormalSize - 1) % originalNormalSize]
                    .first;
            Halfedge e =
                impl.halfedge_[normalEdge[(j + 1) % originalNormalSize]];
            Halfedge e1 =
                impl.halfedge_[normalEdge[(j + originalNormalSize - 1) %
                                          originalNormalSize]];
            out = impl.vertPos_[e.endVert] - impl.vertPos_[e.startVert];
            dvec3 out1 =
                impl.vertPos_[e1.endVert] - impl.vertPos_[e1.startVert];
            if (dot(cross(b, c), out) <= kCollinear) {
              newNormal = averageNormal(a, b, c);
              additional = (j + 2) % originalNormalSize;
            } else if (dot(cross(c1, a), out1) <= kCollinear) {
              newNormal = averageNormal(c1, a, b);
              additional = (j + originalNormalSize - 1) % originalNormalSize;
            }
          }
          normalMap[j] = addNormal(newNormal);
          auto iter = std::find_if(
              consecutives.begin(), consecutives.end(), [&](auto i) {
                return std::find(i.begin(), i.end(), j) != i.end();
              });
          auto iter1 = std::find_if(
              consecutives.begin(), consecutives.end(), [&](auto i) {
                return std::find(i.begin(), i.end(),
                                 (j + 1) % originalNormalSize) != i.end();
              });
          ASSERT(iter != consecutives.end(), logicErr, "cannot find set");
          ASSERT(iter1 != consecutives.end(), logicErr, "cannot find set");
          if (additional == -1) {
            iter->insert(iter1->begin(), iter1->end());
            int a = std::distance(consecutives.begin(), iter);
            int b = std::distance(consecutives.begin(), iter1);
            if (a != b) consecutives.erase(consecutives.begin() + b);
          } else {
            auto iter2 = std::find_if(
                consecutives.begin(), consecutives.end(), [&](auto i) {
                  return std::find(i.begin(), i.end(), additional) != i.end();
                });
            ASSERT(iter2 != consecutives.end(), logicErr, "cannot find set");
            iter->insert(iter1->begin(), iter1->end());
            iter->insert(iter2->begin(), iter2->end());
            int a = std::distance(consecutives.begin(), iter);
            int b = std::distance(consecutives.begin(), iter1);
            int c = std::distance(consecutives.begin(), iter2);
            if (b > c) std::swap(b, c);
            if (a != c) consecutives.erase(consecutives.begin() + c);
            if (b != c && a != b) consecutives.erase(consecutives.begin() + b);
          }
        }
      }
      // note that this just approximates the position
      for (auto& r : consecutives) {
        if (r.size() > 1) {
          dvec3 sum(0);
          std::vector<int> rr(r.begin(), r.end());
          rr.erase(rr.end() - 1);
          if (rr.size() > 1) rr.erase(rr.begin());
          for (int index : r) normalMap[index] = normalMap[rr.front()];
        }
      }
      // remove unused normals
      std::vector<std::pair<dvec3, double>> oldnormals;
      std::swap(normals, oldnormals);
      for (int& index : normalMap) {
        dvec3 v = oldnormals[index].first * oldnormals[index].second;
        index = addNormal(v);
      }
      for (std::pair<int, int>& edge : edgeNormalsLocal) {
        edge.first = normalMap[edge.first];
        edge.second = normalMap[edge.second];
      }
    }

    dvec3 vertNormal = impl.vertNormal_[startVert];

    if (normals.size() >= 3) {
      // project them onto a plane with normal = vertex normal,
      // and x axis defined by edge normal furthest from vertex normal
      // and order them in CW direction.
      // this requires at least 3 non-colinear normals...
      float minCos = 2.0;
      int maxIndex = -1;
      for (int j = 0; j < normals.size(); ++j) {
        float angle = abs(dot(normals[j].first, vertNormal));
        if (angle < minCos) minCos = angle;
        maxIndex = j;
      }
      dvec3 xaxis =
          normalize(normals[maxIndex].first -
                    vertNormal * dot(normals[maxIndex].first, vertNormal));
      dvec3 yaxis = normalize(cross(vertNormal, xaxis));
      Vec<double> angles;
      Vec<int> normalMap(normals.size());
      sequence(ExecutionPolicy::Seq, normalMap.begin(), normalMap.end());
      for (const auto& normal : normals) {
        double angle =
            atan2(dot(normal.first, yaxis), dot(normal.first, xaxis));
        angles.push_back(angle);
      }
      std::stable_sort(
          zip(angles.begin(), normals.begin(), normalMap.begin()),
          zip(angles.end(), normals.end(), normalMap.end()),
          [](const thrust::tuple<double, std::pair<dvec3, double>, int>& a,
             const thrust::tuple<double, std::pair<dvec3, double>, int>& b) {
            return thrust::get<0>(a) > thrust::get<0>(b);
          });
      int lastSecond = -1;
      int s = normalMap.size();

      for (std::pair<int, int>& pair : edgeNormalsLocal) {
        pair.first = std::distance(
            normalMap.begin(),
            std::find(normalMap.begin(), normalMap.end(), pair.first));
        pair.second = std::distance(
            normalMap.begin(),
            std::find(normalMap.begin(), normalMap.end(), pair.second));
      }
#if MANIFOLD_DEBUG
      // check normal projection forms a convex polygon
      for (int j = 0; j < normals.size(); ++j) {
        dvec3 a = normals[(j + 1) % normals.size()].first - normals[j].first;
        dvec3 b = normals[(j + 2) % normals.size()].first -
                  normals[(j + 1) % normals.size()].first;
        ASSERT(dot(cross(a, b), vertNormal) <= 0, logicErr,
               "expects convex projection");
      }
#endif
    }

    int lastSecond = -1;
    int s = normals.size();
    for (int j = 0; j < edges.size(); ++j) {
      std::pair<int, int>& pair = edgeNormalsLocal[j];
      if (pair.second == lastSecond) std::swap(pair.first, pair.second);
      ASSERT(lastSecond == -1 || lastSecond == pair.first ||
                 (lastSecond + 1) % s == pair.first,
             logicErr, "expects monotone normal angle");
      // somehow some unused normal is not being removed...
      ASSERT(pair.first == pair.second || (pair.first + 1) % s == pair.second,
             logicErr, "expects monotone normal angle");
      lastSecond = pair.second;
    }
    vertEdges[startVert] = edges;
    vertNormals[startVert] = normals;
    for (int j = 0; j < edges.size(); ++j)
      edgeNormals[edges[j]] = edgeNormalsLocal[j];
  };
  for_each(ExecutionPolicy::Seq, countAt(0), countAt(impl.halfedge_.size()),
           fn);
  // preallocate vertices and triangles
  std::vector<int> newVertsBefore(impl.NumVert() + 1, 0);
  std::vector<int> newTrisBefore(impl.NumVert() + 1, 0);
  for_each(autoPolicy(impl.NumVert()), countAt(0), countAt(impl.NumVert()),
           [&](int i) {
             newVertsBefore[i] = vertNormals[i].size() - 1;
             newTrisBefore[i] =
                 vertNormals[i].size() >= 3 ? vertNormals[i].size() - 2 : 0;
           });
  exclusive_scan(autoPolicy(impl.NumVert()), newVertsBefore.begin(),
                 newVertsBefore.end(), newVertsBefore.begin(), 0);
  exclusive_scan(autoPolicy(impl.NumVert()), newTrisBefore.begin(),
                 newTrisBefore.end(), newTrisBefore.begin(), 0);
  int oldVertNum = impl.NumVert();
  int oldHalfedgeSize = impl.halfedge_.size();
  impl.vertPos_.resize(oldVertNum + newVertsBefore.back());
  impl.vertNormal_.resize(oldVertNum + newVertsBefore.back());
  impl.halfedge_.resize(oldHalfedgeSize + newTrisBefore.back() * 3);
  for (int i = 0; i < oldVertNum; ++i) {
    dvec3 oldPos = impl.vertPos_[i];
    impl.vertNormal_[i] = vertNormals[i].front().first;
    impl.vertPos_[i] = oldPos + vertNormals[i].front().first *
                                    vertNormals[i].front().second * offset;
    int base = oldVertNum + newVertsBefore[i];
    for (int j = 1; j < vertNormals[i].size(); ++j) {
      impl.vertNormal_[base + j - 1] = vertNormals[i][j].first;
      impl.vertPos_[base + j - 1] =
          oldPos + vertNormals[i][j].first * vertNormals[i][j].second * offset;
    }
    int halfedgeBase = oldHalfedgeSize + 3 * newTrisBefore[i];
    // triangles: 1 0 i; 2 1 i; 3 2 i; ...
    for (int j = 0; j < static_cast<int>(vertNormals[i].size()) - 2; ++j) {
      impl.halfedge_[halfedgeBase + j * 3] = {j + 1 + base, j + base, -1,
                                              halfedgeBase / 3 + j};
      impl.halfedge_[halfedgeBase + j * 3 + 1] = {j + base, i, -1,
                                                  halfedgeBase / 3 + j};
      impl.halfedge_[halfedgeBase + j * 3 + 2] = {i, j + 1 + base, -1,
                                                  halfedgeBase / 3 + j};
    }
    // fix matching halfedges
    for (int j = 0; j < static_cast<int>(vertNormals[i].size()) - 3; ++j) {
      impl.halfedge_[halfedgeBase + j * 3 + 2].pairedHalfedge =
          halfedgeBase + j * 3 + 4;
      impl.halfedge_[halfedgeBase + j * 3 + 4].pairedHalfedge =
          halfedgeBase + j * 3 + 2;
    }
  }
  // only for special case in which one vert is split into 2
  std::unordered_map<int, int> vertPairHalfedge;
  auto normalIdToVertId = [&](int v, int idx) {
    if (idx == 0) return v;
    return oldVertNum + newVertsBefore[v] + idx - 1;
  };
  auto pairHalfedge = [&](int idx, int oldvert) {
    if (vertNormals[oldvert].size() == 2) {
      auto iter = vertPairHalfedge.find(oldvert);
      if (iter == vertPairHalfedge.end()) {
        vertPairHalfedge[oldvert] = idx;
      } else {
        int idx2 = iter->second;
        ASSERT(
            impl.halfedge_[idx].startVert == impl.halfedge_[idx2].endVert &&
                impl.halfedge_[idx].endVert == impl.halfedge_[idx2].startVert,
            logicErr, "halfedge does not match");
        impl.halfedge_[idx].pairedHalfedge = idx2;
        impl.halfedge_[idx2].pairedHalfedge = idx;
      }
    } else if (impl.halfedge_[idx].endVert < oldVertNum) {
      int paired = (vertNormals[oldvert].size() - 2) * 3 - 1 +
                   newTrisBefore[oldvert] * 3 + oldHalfedgeSize;
      ASSERT(
          impl.halfedge_[idx].startVert == impl.halfedge_[paired].endVert &&
              impl.halfedge_[idx].endVert == impl.halfedge_[paired].startVert,
          logicErr, "halfedge does not match");
      impl.halfedge_[paired].pairedHalfedge = idx;
      impl.halfedge_[idx].pairedHalfedge = paired;
    } else {
      int edge =
          impl.halfedge_[idx].endVert - newVertsBefore[oldvert] - oldVertNum;
      ASSERT(edge >= 0, logicErr, "should be >=0 due to CCW");
      int paired = (edge == 0 ? 1 : ((edge - 1) * 3)) +
                   newTrisBefore[oldvert] * 3 + oldHalfedgeSize;
      ASSERT(
          impl.halfedge_[idx].startVert == impl.halfedge_[paired].endVert &&
              impl.halfedge_[idx].endVert == impl.halfedge_[paired].startVert,
          logicErr, "halfedge does not match");
      impl.halfedge_[paired].pairedHalfedge = idx;
      impl.halfedge_[idx].pairedHalfedge = paired;
    }
  };
  for (int i = 0; i < oldHalfedgeSize; ++i) {
    Halfedge& he = impl.halfedge_[i];
    int oldStart = he.startVert;
    int oldEnd = he.endVert;
    if (!he.IsForward() || oldStart >= oldVertNum || oldEnd >= oldVertNum)
      continue;
    Halfedge& eh = impl.halfedge_[he.pairedHalfedge];
    int ehIdx = he.pairedHalfedge;
    int newStart1 = normalIdToVertId(oldStart, edgeNormals[i].first);
    int newStart2 = normalIdToVertId(oldStart, edgeNormals[i].second);
    int newEnd1 =
        normalIdToVertId(oldEnd, edgeNormals[he.pairedHalfedge].first);
    int newEnd2 =
        normalIdToVertId(oldEnd, edgeNormals[he.pairedHalfedge].second);
    he.startVert = newStart1;
    he.endVert = newEnd2;
    eh.startVert = newEnd1;
    eh.endVert = newStart2;
    int diagonal = he.pairedHalfedge;
    // note that he and eh are invalidated after push_back
    if (newStart1 != newStart2) {
      int tri = impl.halfedge_.size() / 3;
      impl.halfedge_.push_back({newStart2, newEnd1, ehIdx, tri});
      impl.halfedge_.push_back({newEnd1, newStart1, -1, tri});
      impl.halfedge_.push_back({newStart1, newStart2, -1, tri});
      pairHalfedge(impl.halfedge_.size() - 1, oldStart);
      impl.halfedge_[ehIdx].pairedHalfedge = impl.halfedge_.size() - 3;
      diagonal = impl.halfedge_.size() - 2;
    }
    if (newEnd1 == newEnd2) {
      impl.halfedge_[i].pairedHalfedge = diagonal;
      impl.halfedge_[diagonal].pairedHalfedge = i;
    } else {
      int tri = impl.halfedge_.size() / 3;
      impl.halfedge_.push_back({newStart1, newEnd1, diagonal, tri});
      impl.halfedge_.push_back({newEnd1, newEnd2, -1, tri});
      impl.halfedge_.push_back({newEnd2, newStart1, i, tri});
      impl.halfedge_[diagonal].pairedHalfedge = impl.halfedge_.size() - 3;
      impl.halfedge_[i].pairedHalfedge = impl.halfedge_.size() - 1;
      pairHalfedge(impl.halfedge_.size() - 2, oldEnd);
    }
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
  auto pImpl_ = std::make_shared<Manifold::Impl>();
  *pImpl_ = *GetCsgLeafNode().GetImpl();
  MultiNormalOffset(*pImpl_, offset);
  // pImpl_->Finish();
  return Manifold(pImpl_);
}
}  // namespace manifold
