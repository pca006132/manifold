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

#include <glm/gtc/integer.hpp>

#include "hashtable.h"
#include "impl.h"
#include "manifold.h"
#include "par.h"
#include "utils.h"
#include "vec.h"

namespace {
using namespace manifold;

ivec3 TetTri0(int i) {
  constexpr ivec3 tetTri0[16] = {{-1, -1, -1},  //
                                 {0, 3, 4},     //
                                 {0, 1, 5},     //
                                 {1, 5, 3},     //
                                 {1, 4, 2},     //
                                 {1, 0, 3},     //
                                 {2, 5, 0},     //
                                 {5, 3, 2},     //
                                 {2, 3, 5},     //
                                 {0, 5, 2},     //
                                 {3, 0, 1},     //
                                 {2, 4, 1},     //
                                 {3, 5, 1},     //
                                 {5, 1, 0},     //
                                 {4, 3, 0},     //
                                 {-1, -1, -1}};
  return tetTri0[i];
}

ivec3 TetTri1(int i) {
  constexpr ivec3 tetTri1[16] = {{-1, -1, -1},  //
                                 {-1, -1, -1},  //
                                 {-1, -1, -1},  //
                                 {3, 4, 1},     //
                                 {-1, -1, -1},  //
                                 {3, 2, 1},     //
                                 {0, 4, 2},     //
                                 {-1, -1, -1},  //
                                 {-1, -1, -1},  //
                                 {2, 4, 0},     //
                                 {1, 2, 3},     //
                                 {-1, -1, -1},  //
                                 {1, 4, 3},     //
                                 {-1, -1, -1},  //
                                 {-1, -1, -1},  //
                                 {-1, -1, -1}};
  return tetTri1[i];
}

ivec4 Neighbors(int i) {
  constexpr ivec4 neighbors[7] = {{0, 0, 0, 1},   //
                                  {1, 0, 0, 0},   //
                                  {0, 1, 0, 0},   //
                                  {0, 0, 1, 0},   //
                                  {-1, 0, 0, 1},  //
                                  {0, -1, 0, 1},  //
                                  {0, 0, -1, 1}};
  return neighbors[i];
}

Uint64 EncodeIndex(ivec4 gridPos, ivec3 gridPow) {
  return static_cast<Uint64>(gridPos.w) | static_cast<Uint64>(gridPos.z) << 1 |
         static_cast<Uint64>(gridPos.y) << (1 + gridPow.z) |
         static_cast<Uint64>(gridPos.x) << (1 + gridPow.z + gridPow.y);
}

ivec4 DecodeIndex(Uint64 idx, ivec3 gridPow) {
  ivec4 gridPos;
  gridPos.w = idx & 1;
  idx = idx >> 1;
  gridPos.z = idx & ((1 << gridPow.z) - 1);
  idx = idx >> gridPow.z;
  gridPos.y = idx & ((1 << gridPow.y) - 1);
  idx = idx >> gridPow.y;
  gridPos.x = idx & ((1 << gridPow.x) - 1);
  return gridPos;
}

vec3 Position(ivec4 gridIndex, vec3 origin, vec3 spacing) {
  return origin + spacing * (vec3(gridIndex) + (gridIndex.w == 1 ? 0.0 : -0.5));
}

double BoundedSDF(ivec4 gridIndex, vec3 origin, vec3 spacing, ivec3 gridSize,
                  double level, std::function<double(vec3)> sdf) {
  auto Min = [](ivec3 p) { return std::min(p.x, std::min(p.y, p.z)); };

  const ivec3 xyz(gridIndex);
  const int lowerBoundDist = Min(xyz);
  const int upperBoundDist = Min(gridSize - xyz);
  const int boundDist = std::min(lowerBoundDist, upperBoundDist - gridIndex.w);

  if (boundDist < 0) {
    return 0.0;
  }
  const double d = sdf(Position(gridIndex, origin, spacing)) - level;
  return boundDist == 0 ? std::min(d, 0.0) : d;
}

struct GridVert {
  double distance = NAN;
  int edgeVerts[7] = {-1, -1, -1, -1, -1, -1, -1};

  int Inside() const { return distance > 0 ? 1 : -1; }

  int NeighborInside(int i) const {
    return Inside() * (edgeVerts[i] < 0 ? 1 : -1);
  }
};

struct ComputeVerts {
  VecView<vec3> vertPos;
  VecView<int> vertIndex;
  HashTableD<GridVert> gridVerts;
  VecView<const double> voxels;
  const std::function<double(vec3)> sdf;
  const vec3 origin;
  const ivec3 gridSize;
  const ivec3 gridPow;
  const vec3 spacing;
  const double level;
  const double tol;

  // Simplified ITP root finding algorithm - same worst-case performance as
  // bisection, better average performance.
  inline vec3 FindSurface(vec3 pos0, double d0, vec3 pos1, double d1) const {
    if (d0 == 0) {
      return pos0;
    } else if (d1 == 0) {
      return pos1;
    }

    // Sole tuning parameter, k: (0, 1) - smaller value gets better median
    // performance, but also hits the worst case more often.
    const double k = 0.1;
    const double check = 2 * tol / glm::length(pos0 - pos1);
    double frac = 1;
    double biFrac = 1;
    while (frac > check) {
      const double t = glm::mix(d0 / (d0 - d1), 0.5, k);
      const double r = biFrac / frac - 0.5;
      const double x = std::abs(t - 0.5) < r ? t : 0.5 - r * (t < 0.5 ? 1 : -1);

      const vec3 mid = glm::mix(pos0, pos1, x);
      const double d = sdf(mid) - level;

      if ((d > 0) == (d0 > 0)) {
        d0 = d;
        pos0 = mid;
        frac *= 1 - x;
      } else {
        d1 = d;
        pos1 = mid;
        frac *= x;
      }
      biFrac /= 2;
    }

    return glm::mix(pos0, pos1, d0 / (d0 - d1));
  }

  inline void operator()(Uint64 index) {
    ZoneScoped;
    if (gridVerts.Full()) return;

    const ivec4 gridIndex = DecodeIndex(index, gridPow);

    if (glm::any(glm::greaterThan(ivec3(gridIndex), gridSize))) return;

    const vec3 position = Position(gridIndex, origin, spacing);

    GridVert gridVert;
    gridVert.distance =
        voxels[EncodeIndex(gridIndex + ivec4(1, 1, 1, 0), gridPow)];

    bool keep = false;
    // These seven edges are uniquely owned by this gridVert; any of them
    // which intersect the surface create a vert.
    for (int i = 0; i < 7; ++i) {
      ivec4 neighborIndex = gridIndex + Neighbors(i);
      if (neighborIndex.w == 2) {
        neighborIndex += 1;
        neighborIndex.w = 0;
      }
      const double val =
          voxels[EncodeIndex(neighborIndex + ivec4(1, 1, 1, 0), gridPow)];
      if ((val > 0) == (gridVert.distance > 0)) continue;
      keep = true;

      const int idx = AtomicAdd(vertIndex[0], 1);
      vertPos[idx] = FindSurface(position, gridVert.distance,
                                 Position(neighborIndex, origin, spacing), val);
      gridVert.edgeVerts[i] = idx;
    }

    if (keep) gridVerts.Insert(index, gridVert);
  }
};

struct BuildTris {
  VecView<ivec3> triVerts;
  VecView<int> triIndex;
  const HashTableD<GridVert> gridVerts;
  const ivec3 gridPow;

  void CreateTri(const ivec3& tri, const int edges[6]) {
    if (tri[0] < 0) return;
    int idx = AtomicAdd(triIndex[0], 1);
    triVerts[idx] = {edges[tri[0]], edges[tri[1]], edges[tri[2]]};
  }

  void CreateTris(const ivec4& tet, const int edges[6]) {
    const int i = (tet[0] > 0 ? 1 : 0) + (tet[1] > 0 ? 2 : 0) +
                  (tet[2] > 0 ? 4 : 0) + (tet[3] > 0 ? 8 : 0);
    CreateTri(TetTri0(i), edges);
    CreateTri(TetTri1(i), edges);
  }

  void operator()(int idx) {
    ZoneScoped;
    Uint64 basekey = gridVerts.KeyAt(idx);
    if (basekey == kOpen) return;

    const GridVert& base = gridVerts.At(idx);
    const ivec4 baseIndex = DecodeIndex(basekey, gridPow);

    ivec4 leadIndex = baseIndex;
    if (leadIndex.w == 0)
      leadIndex.w = 1;
    else {
      leadIndex += 1;
      leadIndex.w = 0;
    }

    // This GridVert is in charge of the 6 tetrahedra surrounding its edge in
    // the (1,1,1) direction (edge 0).
    ivec4 tet(base.NeighborInside(0), base.Inside(), -2, -2);
    ivec4 thisIndex = baseIndex;
    thisIndex.x += 1;

    GridVert thisVert = gridVerts[EncodeIndex(thisIndex, gridPow)];

    tet[2] = base.NeighborInside(1);
    for (const int i : {0, 1, 2}) {
      thisIndex = leadIndex;
      --thisIndex[Prev3(i)];
      // indices take unsigned input, so check for negatives, given the
      // decrement.
      GridVert nextVert = thisIndex[Prev3(i)] < 0
                              ? GridVert()
                              : gridVerts[EncodeIndex(thisIndex, gridPow)];
      tet[3] = base.NeighborInside(Prev3(i) + 4);

      const int edges1[6] = {base.edgeVerts[0],
                             base.edgeVerts[i + 1],
                             nextVert.edgeVerts[Next3(i) + 4],
                             nextVert.edgeVerts[Prev3(i) + 1],
                             thisVert.edgeVerts[i + 4],
                             base.edgeVerts[Prev3(i) + 4]};
      thisVert = nextVert;
      CreateTris(tet, edges1);

      thisIndex = baseIndex;
      ++thisIndex[Next3(i)];
      nextVert = gridVerts[EncodeIndex(thisIndex, gridPow)];
      tet[2] = tet[3];
      tet[3] = base.NeighborInside(Next3(i) + 1);

      const int edges2[6] = {base.edgeVerts[0],
                             edges1[5],
                             thisVert.edgeVerts[i + 4],
                             nextVert.edgeVerts[Next3(i) + 4],
                             edges1[3],
                             base.edgeVerts[Next3(i) + 1]};
      thisVert = nextVert;
      CreateTris(tet, edges2);

      tet[2] = tet[3];
    }
  }
};
}  // namespace

namespace manifold {

/** @addtogroup Core
 *  @{
 */

/**
 * Constructs a level-set MeshGL from the input Signed-Distance Function (SDF).
 * This uses a form of Marching Tetrahedra (akin to Marching Cubes, but better
 * for manifoldness). Instead of using a cubic grid, it uses a body-centered
 * cubic grid (two shifted cubic grids). This means if your function's interior
 * exceeds the given bounds, you will see a kind of egg-crate shape closing off
 * the manifold, which is due to the underlying grid.
 *
 * @param sdf The signed-distance functor, containing this function signature:
 * `double operator()(vec3 point)`, which returns the
 * signed distance of a given point in R^3. Positive values are inside,
 * negative outside.
 * @param bounds An axis-aligned box that defines the extent of the grid.
 * @param edgeLength Approximate maximum edge length of the triangles in the
 * final result. This affects grid spacing, and hence has a strong effect on
 * performance.
 * @param level You can inset your Mesh by using a positive value, or outset
 * it with a negative value.
 * @param precision Ensure each vertex is within this distance of the true
 * surface. Defaults to -1, which will return the interpolated
 * crossing-point based on the two nearest grid points. Small positive values
 * will require more sdf evaluations per output vertex.
 * @param canParallel Parallel policies violate will crash language runtimes
 * with runtime locks that expect to not be called back by unregistered threads.
 * This allows bindings use LevelSet despite being compiled with MANIFOLD_PAR
 * active.
 * @return MeshGL This mesh is guaranteed to be manifold and so can always be
 * used as input to the Manifold constructor for further operations.
 */
MeshGL MeshGL::LevelSet(std::function<double(vec3)> sdf, Box bounds,
                        double edgeLength, double level, double precision,
                        bool canParallel) {
  if (precision <= 0) {
    precision = std::numeric_limits<double>::infinity();
  }
  const vec3 dim = bounds.Size();
  const ivec3 gridSize(dim / edgeLength + 1.0);
  const vec3 spacing = dim / (vec3(gridSize - 1));

  const ivec3 gridPow(glm::log2(gridSize + 2) + 1);
  const Uint64 maxIndex = EncodeIndex(ivec4(gridSize + 2, 1), gridPow);

  // Parallel policies violate will crash language runtimes with runtime locks
  // that expect to not be called back by unregistered threads. This allows
  // bindings use LevelSet despite being compiled with MANIFOLD_PAR
  // active.
  const auto pol = canParallel ? autoPolicy(maxIndex) : ExecutionPolicy::Seq;

  const vec3 origin = bounds.min;
  Vec<double> voxels(maxIndex);
  for_each_n(
      pol, countAt(0_uz), maxIndex,
      [&voxels, sdf, level, origin, spacing, gridSize, gridPow](Uint64 idx) {
        voxels[idx] = BoundedSDF(DecodeIndex(idx, gridPow) - ivec4(1, 1, 1, 0),
                                 origin, spacing, gridSize, level, sdf);
      });

  size_t tableSize = std::min(
      2 * maxIndex, static_cast<Uint64>(10 * glm::pow(maxIndex, 0.667)));
  HashTable<GridVert> gridVerts(tableSize);
  Vec<vec3> vertPos(gridVerts.Size() * 7);

  while (1) {
    Vec<int> index(1, 0);
    for_each_n(pol, countAt(0_uz), EncodeIndex(ivec4(gridSize, 1), gridPow),
               ComputeVerts({vertPos, index, gridVerts.D(), voxels, sdf, origin,
                             gridSize, gridPow, spacing, level, precision}));

    if (gridVerts.Full()) {  // Resize HashTable
      const vec3 lastVert = vertPos[index[0] - 1];
      const Uint64 lastIndex =
          EncodeIndex(ivec4((lastVert - origin) / spacing, 1), gridPow);
      const double ratio = static_cast<double>(maxIndex) / lastIndex;

      if (ratio > 1000)  // do not trust the ratio if it is too large
        tableSize *= 2;
      else
        tableSize *= ratio;
      gridVerts = HashTable<GridVert>(tableSize);
      vertPos = Vec<vec3>(gridVerts.Size() * 7);
    } else {  // Success
      vertPos.resize(index[0]);
      break;
    }
  }

  Vec<ivec3> triVerts(gridVerts.Entries() * 12);  // worst case

  Vec<int> index(1, 0);
  for_each_n(pol, countAt(0), gridVerts.Size(),
             BuildTris({triVerts, index, gridVerts.D(), gridPow}));
  triVerts.resize(index[0]);

  MeshGL out;
  out.numProp = 3;
  out.vertProperties.resize(out.numProp * vertPos.size());
  for (size_t i = 0; i < vertPos.size(); ++i) {
    for (int j : {0, 1, 2}) out.vertProperties[3 * i + j] = vertPos[i][j];
  }
  out.triVerts.resize(3 * triVerts.size());
  for (size_t i = 0; i < triVerts.size(); ++i) {
    for (int j : {0, 1, 2}) out.triVerts[3 * i + j] = triVerts[i][j];
  }
  return out;
}
/** @} */
}  // namespace manifold
