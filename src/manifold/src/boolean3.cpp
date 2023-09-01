// Copyright 2021 The Manifold Authors.
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

#include "boolean3.h"

#include <iostream>
#include <limits>

#include "par.h"

using namespace manifold;

namespace {

// These two functions (Interpolate and Intersect) are the only places where
// floating-point operations take place in the whole Boolean function. These are
// carefully designed to minimize rounding error and to eliminate it at edge
// cases to ensure consistency.

glm::vec2 Interpolate(glm::vec3 pL, glm::vec3 pR, float x) {
  float dxL = x - pL.x;
  float dxR = x - pR.x;
#ifdef MANIFOLD_DEBUG
  if (dxL * dxR > 0) printf("Not in domain!\n");
#endif
  bool useL = fabs(dxL) < fabs(dxR);
  float lambda = (useL ? dxL : dxR) / (pR.x - pL.x);
  if (!isfinite(lambda)) return glm::vec2(pL.y, pL.z);
  glm::vec2 yz;
  yz[0] = (useL ? pL.y : pR.y) + lambda * (pR.y - pL.y);
  yz[1] = (useL ? pL.z : pR.z) + lambda * (pR.z - pL.z);
  return yz;
}

glm::vec4 Intersect(const glm::vec3 &pL, const glm::vec3 &pR,
                    const glm::vec3 &qL, const glm::vec3 &qR) {
  float dyL = qL.y - pL.y;
  float dyR = qR.y - pR.y;
#ifdef MANIFOLD_DEBUG
  if (dyL * dyR > 0) printf("No intersection!\n");
#endif
  bool useL = fabs(dyL) < fabs(dyR);
  float dx = pR.x - pL.x;
  float lambda = (useL ? dyL : dyR) / (dyL - dyR);
  if (!isfinite(lambda)) lambda = 0.0f;
  glm::vec4 xyzz;
  xyzz.x = (useL ? pL.x : pR.x) + lambda * dx;
  float pDy = pR.y - pL.y;
  float qDy = qR.y - qL.y;
  bool useP = fabs(pDy) < fabs(qDy);
  xyzz.y = (useL ? (useP ? pL.y : qL.y) : (useP ? pR.y : qR.y)) +
           lambda * (useP ? pDy : qDy);
  xyzz.z = (useL ? pL.z : pR.z) + lambda * (pR.z - pL.z);
  xyzz.w = (useL ? qL.z : qR.z) + lambda * (qR.z - qL.z);
  return xyzz;
}

template <const bool inverted>
struct CopyFaceEdges {
  // x can be either vert or edge (0 or 1).
  const SparseIndices &pXq1;
  SparseEdgeEdgeMap::Store p1q1;
  VecView<const Halfedge> halfedgesQ;

  void operator()(thrust::tuple<int, int> in) {
    int idx = 3 * thrust::get<0>(in);
    int i = thrust::get<1>(in);
    int pX = pXq1.Get(i, inverted);
    int q2 = pXq1.Get(i, !inverted);

    for (const int j : {0, 1, 2}) {
      const int q1 = 3 * q2 + j;
      const Halfedge edge = halfedgesQ[q1];
      int a = pX;
      int b = edge.IsForward() ? q1 : edge.pairedHalfedge;
      if (inverted) std::swap(a, b);
      uint64_t key =
          (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
      p1q1.Insert(key, std::make_pair(0, glm::vec4(0)));
    }
  }
};

SparseEdgeEdgeMap Filter11(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                           const SparseIndices &p1q2,
                           const SparseIndices &p2q1) {
  SparseEdgeEdgeMap p1q1(3 * p1q2.size() + 3 * p2q1.size());
  for_each_n(autoPolicy(p1q2.size()), zip(countAt(0), countAt(0)), p1q2.size(),
             CopyFaceEdges<false>({p1q2, p1q1.D(), inQ.halfedge_}));
  for_each_n(autoPolicy(p2q1.size()), zip(countAt(p1q2.size()), countAt(0)),
             p2q1.size(), CopyFaceEdges<true>({p2q1, p1q1.D(), inP.halfedge_}));
  p1q1.Resize();
  return p1q1;
}

inline bool Shadows(float p, float q, float dir) {
  return p == q ? dir < 0 : p < q;
}

/**
 * Since this function is called from two different places, it is necessary that
 * it returns identical results for identical input to keep consistency.
 * Normally this is trivial as computers make floating-point errors, but are
 * at least deterministic. However, in the case of CUDA, these functions can be
 * compiled by two different compilers (for the CPU and GPU). We have found that
 * the different compilers can cause slightly different rounding errors, so it
 * is critical that the two places this function is called both use the same
 * compiled function (they must agree on CPU or GPU). This is now taken care of
 * by the shared policy_ member.
 */
inline thrust::pair<int, glm::vec2> Shadow01(
    const int p0, const int q1, VecView<const glm::vec3> vertPosP,
    VecView<const glm::vec3> vertPosQ, VecView<const Halfedge> halfedgeQ,
    const float expandP, VecView<const glm::vec3> normalP, const bool reverse) {
  const int q1s = halfedgeQ[q1].startVert;
  const int q1e = halfedgeQ[q1].endVert;
  const float p0x = vertPosP[p0].x;
  const float q1sx = vertPosQ[q1s].x;
  const float q1ex = vertPosQ[q1e].x;
  int s01 = reverse ? Shadows(q1sx, p0x, expandP * normalP[q1s].x) -
                          Shadows(q1ex, p0x, expandP * normalP[q1e].x)
                    : Shadows(p0x, q1ex, expandP * normalP[p0].x) -
                          Shadows(p0x, q1sx, expandP * normalP[p0].x);
  glm::vec2 yz01(NAN);

  if (s01 != 0) {
    yz01 = Interpolate(vertPosQ[q1s], vertPosQ[q1e], vertPosP[p0].x);
    if (reverse) {
      glm::vec3 diff = vertPosQ[q1s] - vertPosP[p0];
      const float start2 = glm::dot(diff, diff);
      diff = vertPosQ[q1e] - vertPosP[p0];
      const float end2 = glm::dot(diff, diff);
      const float dir = start2 < end2 ? normalP[q1s].y : normalP[q1e].y;
      if (!Shadows(yz01[0], vertPosP[p0].y, expandP * dir)) s01 = 0;
    } else {
      if (!Shadows(vertPosP[p0].y, yz01[0], expandP * normalP[p0].y)) s01 = 0;
    }
  }
  return thrust::make_pair(s01, yz01);
}

struct Kernel11 {
  VecView<const glm::vec3> vertPosP;
  VecView<const glm::vec3> vertPosQ;
  VecView<const Halfedge> halfedgeP;
  VecView<const Halfedge> halfedgeQ;
  float expandP;
  VecView<const glm::vec3> normalP;
  SparseEdgeEdgeMap::Store p1q1;

  void operator()(int idx) {
    uint64_t key = p1q1.KeyAt(idx);
    if (key == SparseEdgeEdgeMap::Open() ||
        key == SparseEdgeEdgeMap::Tombstone())
      return;
    const int p1 = key >> 32;
    const int q1 = key;

    glm::vec4 &xyzz11 = p1q1.At(idx).second;
    int &s11 = p1q1.At(idx).first;

    // For pRL[k], qRL[k], k==0 is the left and k==1 is the right.
    int k = 0;
    glm::vec3 pRL[2], qRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    s11 = 0;

    const int p0[2] = {halfedgeP[p1].startVert, halfedgeP[p1].endVert};
    for (int i : {0, 1}) {
      const auto syz01 = Shadow01(p0[i], q1, vertPosP, vertPosQ, halfedgeQ,
                                  expandP, normalP, false);
      const int s01 = syz01.first;
      const glm::vec2 yz01 = syz01.second;
      // If the value is NaN, then these do not overlap.
      if (isfinite(yz01[0])) {
        s11 += s01 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          pRL[k] = vertPosP[p0[i]];
          qRL[k] = glm::vec3(pRL[k].x, yz01);
          ++k;
        }
      }
    }

    const int q0[2] = {halfedgeQ[q1].startVert, halfedgeQ[q1].endVert};
    for (int i : {0, 1}) {
      const auto syz10 = Shadow01(q0[i], p1, vertPosQ, vertPosP, halfedgeP,
                                  expandP, normalP, true);
      const int s10 = syz10.first;
      const glm::vec2 yz10 = syz10.second;
      // If the value is NaN, then these do not overlap.
      if (isfinite(yz10[0])) {
        s11 += s10 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s10 != 0) != shadows)) {
          shadows = s10 != 0;
          qRL[k] = vertPosQ[q0[i]];
          pRL[k] = glm::vec3(qRL[k].x, yz10);
          ++k;
        }
      }
    }

    if (s11 == 0) {  // No intersection
      p1q1.RemoveKey(idx);
    } else {
#ifdef MANIFOLD_DEBUG
      // Assert left and right were both found
      if (k != 2) {
        printf("k = %d\n", k);
      }
#endif
      xyzz11 = Intersect(pRL[0], pRL[1], qRL[0], qRL[1]);

      const int p1s = halfedgeP[p1].startVert;
      const int p1e = halfedgeP[p1].endVert;
      glm::vec3 diff = vertPosP[p1s] - glm::vec3(xyzz11);
      const float start2 = glm::dot(diff, diff);
      diff = vertPosP[p1e] - glm::vec3(xyzz11);
      const float end2 = glm::dot(diff, diff);
      const float dir = start2 < end2 ? normalP[p1s].z : normalP[p1e].z;

      if (!Shadows(xyzz11.z, xyzz11.w, expandP * dir)) s11 = 0;
      if (!isfinite(xyzz11.x)) p1q1.RemoveKey(idx);
    }
  }
};

void Shadow11(SparseEdgeEdgeMap &p1q1, const Manifold::Impl &inP,
              const Manifold::Impl &inQ, float expandP) {
  for_each_n(autoPolicy(p1q1.Size()), countAt(0), p1q1.Size(),
             Kernel11({inP.vertPos_, inQ.vertPos_, inP.halfedge_, inQ.halfedge_,
                       expandP, inP.vertNormal_, p1q1.D()}));
  p1q1.Resize();
};

struct Kernel02 {
  VecView<const glm::vec3> vertPosP;
  VecView<const Halfedge> halfedgeQ;
  VecView<const glm::vec3> vertPosQ;
  const float expandP;
  VecView<const glm::vec3> vertNormalP;
  SparseVertexFaceMap::Store p0q2;
  const bool forward;

  void operator()(int idx) {
    uint64_t key = p0q2.KeyAt(idx);
    if (key == SparseVertexFaceMap::Open() ||
        key == SparseVertexFaceMap::Tombstone())
      return;
    uint32_t p0 = key >> 32;
    uint32_t q2 = key;
    if (!forward) std::swap(p0, q2);

    int &s02 = p0q2.At(idx).first;
    float &z02 = p0q2.At(idx).second;

    // For yzzLR[k], k==0 is the left and k==1 is the right.
    int k = 0;
    glm::vec3 yzzRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    int closestVert = -1;
    float minMetric = std::numeric_limits<float>::infinity();
    s02 = 0;

    const glm::vec3 posP = vertPosP[p0];
    for (const int i : {0, 1, 2}) {
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgeQ[q1];
      const int q1F = edge.IsForward() ? q1 : edge.pairedHalfedge;

      if (!forward) {
        const int qVert = halfedgeQ[q1F].startVert;
        const glm::vec3 diff = posP - vertPosQ[qVert];
        const float metric = glm::dot(diff, diff);
        if (metric < minMetric) {
          minMetric = metric;
          closestVert = qVert;
        }
      }

      const auto syz01 = Shadow01(p0, q1F, vertPosP, vertPosQ, halfedgeQ,
                                  expandP, vertNormalP, !forward);
      const int s01 = syz01.first;
      const glm::vec2 yz01 = syz01.second;
      // If the value is NaN, then these do not overlap.
      if (isfinite(yz01[0])) {
        s02 += s01 * (forward == edge.IsForward() ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          yzzRL[k++] = glm::vec3(yz01[0], yz01[1], yz01[1]);
        }
      }
    }

    if (s02 == 0) {  // No intersection
      p0q2.RemoveKey(idx);
    } else {
#ifdef MANIFOLD_DEBUG
      // Assert left and right were both found
      if (k != 2) {
        printf("k = %d\n", k);
      }
#endif
      glm::vec3 vertPos = vertPosP[p0];
      z02 = Interpolate(yzzRL[0], yzzRL[1], vertPos.y)[1];
      if (forward) {
        if (!Shadows(vertPos.z, z02, expandP * vertNormalP[p0].z)) s02 = 0;
      } else {
        // ASSERT(closestVert != -1, topologyErr, "No closest vert");
        if (!Shadows(z02, vertPos.z, expandP * vertNormalP[closestVert].z))
          s02 = 0;
      }
      if (!isfinite(z02)) p0q2.RemoveKey(idx);
    }
  }
};

void Shadow02(const Manifold::Impl &inP, const Manifold::Impl &inQ,
              SparseVertexFaceMap &p0q2, bool forward, float expandP) {
  auto vertNormalP = forward ? inP.vertNormal_ : inQ.vertNormal_;
  for_each_n(autoPolicy(p0q2.Size()), countAt(0), p0q2.Size(),
             Kernel02({inP.vertPos_, inQ.halfedge_, inQ.vertPos_, expandP,
                       vertNormalP, p0q2.D(), forward}));
  p0q2.Resize();
};

struct Kernel12 {
  const SparseVertexFaceMap::Store p0q2;
  const SparseEdgeEdgeMap::Store p1q1;
  VecView<const Halfedge> halfedgesP;
  VecView<const Halfedge> halfedgesQ;
  VecView<const glm::vec3> vertPosP;
  const bool forward;
  const SparseIndices &p1q2;

  void operator()(thrust::tuple<int, int &, glm::vec3 &> inout) {
    int p1 = p1q2.Get(thrust::get<0>(inout), !forward);
    int q2 = p1q2.Get(thrust::get<0>(inout), forward);
    int &x12 = thrust::get<1>(inout);
    glm::vec3 &v12 = thrust::get<2>(inout);

    // For xzyLR-[k], k==0 is the left and k==1 is the right.
    int k = 0;
    glm::vec3 xzyLR0[2];
    glm::vec3 xzyLR1[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    x12 = 0;

    const Halfedge edge = halfedgesP[p1];

    for (int vert : {edge.startVert, edge.endVert}) {
      const uint64_t key = forward ? SparseIndices::EncodePQ(vert, q2)
                                   : SparseIndices::EncodePQ(q2, vert);
      const uint32_t idx = p0q2.GetIdx(key);
      const uint64_t foundKey = p0q2.KeyAt(idx);
      if (foundKey == key) {
        const int s = p0q2.At(idx).first;
        x12 += s * ((vert == edge.startVert) == forward ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          xzyLR0[k] = vertPosP[vert];
          thrust::swap(xzyLR0[k].y, xzyLR0[k].z);
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = p0q2.At(idx).second;
          k++;
        }
      }
    }

    for (const int i : {0, 1, 2}) {
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgesQ[q1];
      const int q1F = edge.IsForward() ? q1 : edge.pairedHalfedge;
      const uint64_t key = forward ? SparseIndices::EncodePQ(p1, q1F)
                                   : SparseIndices::EncodePQ(q1F, p1);
      const uint32_t idx = p1q1.GetIdx(key);
      const uint64_t foundKey = p1q1.KeyAt(idx);
      // s is implicitly zero for anything not found
      if (foundKey == key) {
        const int s = p1q1.At(idx).first;
        x12 -= s * (edge.IsForward() ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          const glm::vec4 xyzz = p1q1.At(idx).second;
          xzyLR0[k][0] = xyzz.x;
          xzyLR0[k][1] = xyzz.z;
          xzyLR0[k][2] = xyzz.y;
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = xyzz.w;
          if (!forward) thrust::swap(xzyLR0[k][1], xzyLR1[k][1]);
          k++;
        }
      }
    }

    if (x12 == 0) {  // No intersection
      v12 = glm::vec3(NAN);
    } else {
#ifdef MANIFOLD_DEBUG
      // Assert left and right were both found
      if (k != 2) {
        printf("k = %d\n", k);
      }
#endif
      const glm::vec4 xzyy =
          Intersect(xzyLR0[0], xzyLR0[1], xzyLR1[0], xzyLR1[1]);
      v12.x = xzyy[0];
      v12.y = xzyy[2];
      v12.z = xzyy[1];
    }
  }
};

std::tuple<Vec<int>, Vec<glm::vec3>> Intersect12(
    const Manifold::Impl &inP, const Manifold::Impl &inQ,
    const SparseVertexFaceMap &p0q2, const SparseEdgeEdgeMap &p1q1,
    SparseIndices &p1q2, bool forward) {
  Vec<int> x12(p1q2.size());
  Vec<glm::vec3> v12(p1q2.size());

  for_each_n(autoPolicy(p1q2.size()), zip(countAt(0), x12.begin(), v12.begin()),
             p1q2.size(),
             Kernel12({p0q2.D(), p1q1.D(), inP.halfedge_, inQ.halfedge_,
                       inP.vertPos_, forward, p1q2}));

  p1q2.KeepFinite(v12, x12);

  return std::make_tuple(x12, v12);
};

Vec<int> Winding03(const Manifold::Impl &inP, Vec<int> &vertices, Vec<int> &s02,
                   bool reverse) {
  // verts that are not shadowed (not in p0q2) have winding number zero.
  Vec<int> w03(inP.NumVert(), 0);
  auto policy = autoPolicy(vertices.size());
  Vec<int> w03val(w03.size());
  Vec<int> w03vert(w03.size());
  // sum known s02 values into w03 (winding number)
  auto endPair = reduce_by_key<
      thrust::pair<decltype(w03val.begin()), decltype(w03val.begin())>>(
      policy, vertices.begin(), vertices.end(), s02.begin(), w03vert.begin(),
      w03val.begin());
  scatter(policy, w03val.begin(), endPair.second, w03vert.begin(), w03.begin());

  if (reverse)
    transform(policy, w03.begin(), w03.end(), w03.begin(),
              thrust::negate<int>());
  return w03;
};
}  // namespace

namespace manifold {
Boolean3::Boolean3(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                   OpType op)
    : inP_(inP), inQ_(inQ), expandP_(op == OpType::Add ? 1.0 : -1.0) {
  // Symbolic perturbation:
  // Union -> expand inP
  // Difference, Intersection -> contract inP

#ifdef MANIFOLD_DEBUG
  Timer broad;
  broad.Start();
#endif

  if (inP.IsEmpty() || inQ.IsEmpty() || !inP.bBox_.DoesOverlap(inQ.bBox_)) {
    PRINT("No overlap, early out");
    w03_.resize(inP.NumVert(), 0);
    w30_.resize(inQ.NumVert(), 0);
    return;
  }

  // Level 3
  // Find edge-triangle overlaps (broad phase)
  p1q2_ = inQ_.EdgeCollisions(inP_);
  p2q1_ = inP_.EdgeCollisions(inQ_, true);  // inverted

  p1q2_.Sort();
  PRINT("p1q2 size = " << p1q2_.size());

  p2q1_.Sort();
  PRINT("p2q1 size = " << p2q1_.size());

  // Level 2
  // Find vertices that overlap faces in XY-projection
  SparseVertexFaceMap p0q2 = inQ.VertexCollisionsZ(inP.vertPos_);
  PRINT("p0q2 size = " << p0q2.Entries());

  SparseVertexFaceMap p2q0 =
      inP.VertexCollisionsZ(inQ.vertPos_, true);  // inverted
  PRINT("p2q0 size = " << p2q0.Entries());

  // Find involved edge pairs from Level 3
  SparseEdgeEdgeMap p1q1 = Filter11(inP_, inQ_, p1q2_, p2q1_);
  // {
  //   SparseIndices tmp;
  //   for (int i = 0; i < p1q1.Size(); ++i) {
  //     uint64_t key = p1q1.D().KeyAt(i);
  //     if (key == SparseEdgeEdgeMap::Open() ||
  //         key == SparseEdgeEdgeMap::Tombstone())
  //       continue;
  //     tmp.Add(static_cast<int32_t>(key >> 32), static_cast<int32_t>(key));
  //   }
  //   sort(autoPolicy(tmp.size()), tmp.AsVec64().begin(), tmp.AsVec64().end());
  //   for (int i = 0; i < tmp.size(); ++i) {
  //     printf("%d %d\n", tmp.Get(i, false), tmp.Get(i, true));
  //   }
  // }
  PRINT("p1q1 size = " << p1q1.Entries());

#ifdef MANIFOLD_DEBUG
  broad.Stop();
  Timer intersections;
  intersections.Start();
#endif

  // Level 2
  // Build up XY-projection intersection of two edges, including the z-value for
  // each edge, keeping only those whose intersection exists.
  Shadow11(p1q1, inP, inQ, expandP_);

  // {
  //   SparseIndices tmp;
  //   Vec<int> tmp1;
  //   Vec<glm::vec4> tmp2;
  //   for (int i = 0; i < p1q1.Size(); ++i) {
  //     uint64_t key = p1q1.D().KeyAt(i);
  //     if (key == SparseEdgeEdgeMap::Open() ||
  //         key == SparseEdgeEdgeMap::Tombstone())
  //       continue;
  //     tmp.Add(static_cast<int32_t>(key >> 32), static_cast<int32_t>(key));
  //     tmp1.push_back(p1q1.D().At(i).first);
  //     tmp2.push_back(p1q1.D().At(i).second);
  //   }
  //   sort_by_key(autoPolicy(tmp.size()), tmp.AsVec64().begin(),
  //               tmp.AsVec64().end(), zip(tmp1.begin(), tmp2.begin()));
  //   for (int i = 0; i < tmp.size(); ++i) {
  //     printf("%d %d -> %d (%.3f, %.3f, %.3f, %.3f)\n", tmp.Get(i, false),
  //            tmp.Get(i, true), tmp1[i], tmp2[i].x, tmp2[i].y, tmp2[i].z,
  //            tmp2[i].w);
  //   }
  // }
  PRINT("s11 size = " << p1q1.Entries());

  // Build up Z-projection of vertices onto triangles, keeping only those that
  // fall inside the triangle.
  Shadow02(inP, inQ, p0q2, true, expandP_);
  // {
  //   SparseIndices tmp;
  //   Vec<int> tmp1;
  //   Vec<float> tmp2;
  //   for (int i = 0; i < p0q2.Size(); ++i) {
  //     uint64_t key = p0q2.D().KeyAt(i);
  //     if (key == SparseEdgeEdgeMap::Open() ||
  //         key == SparseEdgeEdgeMap::Tombstone())
  //       continue;
  //     tmp.Add(static_cast<int32_t>(key >> 32), static_cast<int32_t>(key));
  //     tmp1.push_back(p0q2.D().At(i).first);
  //     tmp2.push_back(p0q2.D().At(i).second);
  //   }
  //   sort_by_key(autoPolicy(tmp.size()), tmp.AsVec64().begin(),
  //               tmp.AsVec64().end(), zip(tmp1.begin(), tmp2.begin()));
  //   for (int i = 0; i < tmp.size(); ++i) {
  //     printf("%d %d -> %d %.3f\n", tmp.Get(i, false), tmp.Get(i, true),
  //     tmp1[i],
  //            tmp2[i]);
  //   }
  // }
  PRINT("s02 size = " << p0q2.Entries());
  Shadow02(inQ, inP, p2q0, false, expandP_);
  // {
  //   SparseIndices tmp;
  //   Vec<int> tmp1;
  //   Vec<float> tmp2;
  //   for (int i = 0; i < p2q0.Size(); ++i) {
  //     uint64_t key = p2q0.D().KeyAt(i);
  //     if (key == SparseEdgeEdgeMap::Open() ||
  //         key == SparseEdgeEdgeMap::Tombstone())
  //       continue;
  //     tmp.Add(static_cast<int32_t>(key >> 32), static_cast<int32_t>(key));
  //     tmp1.push_back(p2q0.D().At(i).first);
  //     tmp2.push_back(p2q0.D().At(i).second);
  //   }
  //   sort_by_key(autoPolicy(tmp.size()), tmp.AsVec64().begin(),
  //               tmp.AsVec64().end(), zip(tmp1.begin(), tmp2.begin()));
  //   for (int i = 0; i < tmp.size(); ++i) {
  //     printf("%d %d -> %d %.3f\n", tmp.Get(i, false), tmp.Get(i, true),
  //     tmp1[i],
  //            tmp2[i]);
  //   }
  // }
  PRINT("s20 size = " << p2q0.Entries());

  // Level 3
  // Build up the intersection of the edges and triangles, keeping only those
  // that intersect, and record the direction the edge is passing through the
  // triangle.
  // TODO: bug here
  std::tie(x12_, v12_) = Intersect12(inP, inQ, p0q2, p1q1, p1q2_, true);
  // for (int i = 0; i < x12_.size(); ++i) {
  //   printf("%d %d -> %d (%.3f, %.3f, %.3f)\n", p1q2_.Get(i, false),
  //          p1q2_.Get(i, true), x12_[i], v12_[i].x, v12_[i].y, v12_[i].z);
  // }
  PRINT("x12 size = " << x12_.size());

  std::tie(x21_, v21_) = Intersect12(inQ, inP, p2q0, p1q1, p2q1_, false);
  // for (int i = 0; i < p2q1_.size(); ++i) {
  //   printf("%d %d -> %d (%.3f, %.3f, %.3f)\n", p2q1_.Get(i, false),
  //          p2q1_.Get(i, true), x21_[i], v21_[i].x, v21_[i].y, v21_[i].z);
  // }
  PRINT("x21 size = " << x21_.size());

  // Sum up the winding numbers of all vertices.
  {
    Vec<int> p0(p0q2.Entries());
    Vec<int> s02(p0q2.Entries());
    Vec<Uint64> keys;
    Vec<std::pair<int, float>> values;
    std::tie(keys, values) = p0q2.Move();
    sort_by_key(autoPolicy(keys.size()), keys.begin(), keys.end(),
                values.begin());
    for_each_n(autoPolicy(p0.size()), countAt(0), p0.size(), [&](int i) {
      p0[i] = keys[i] >> 32;
      s02[i] = values[i].first;
    });
    keys.resize(0);
    values.resize(0);
    // for (int i = 0; i < p0.size(); ++i) {
    //   printf("%d -> %d\n", p0[i], s02[i]);
    // }
    w03_ = Winding03(inP, p0, s02, false);
  }

  {
    // printf("p2q0.Entries() = %d\n", p2q0.Entries());
    Vec<int> q0(p2q0.Entries());
    Vec<int> s20(p2q0.Entries());
    Vec<Uint64> keys;
    Vec<std::pair<int, float>> values;
    std::tie(keys, values) = p2q0.Move();
    for_each_n(autoPolicy(keys.size()), countAt(0), keys.size(), [&](int i) {
      uint64_t key = keys[i];
      keys[i] =
          ((key & std::numeric_limits<uint32_t>::max()) << 32) | key >> 32;
    });
    sort_by_key(autoPolicy(keys.size()), keys.begin(), keys.end(),
                values.begin());
    for_each_n(autoPolicy(q0.size()), countAt(0), q0.size(), [&](int i) {
      q0[i] = keys[i] >> 32;
      s20[i] = values[i].first;
    });
    keys.resize(0);
    values.resize(0);
    // for (int i = 0; i < q0.size(); ++i) {
    //   printf("%d -> %d\n", q0[i], s20[i]);
    // }
    // std::cout.flush();
    w30_ = Winding03(inQ, q0, s20, true);
  }

#ifdef MANIFOLD_DEBUG
  intersections.Stop();

  if (ManifoldParams().verbose) {
    broad.Print("Broad phase");
    intersections.Print("Intersections");
    MemUsage();
  }
#endif
}
}  // namespace manifold
