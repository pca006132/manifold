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

#include "polygon.h"
#if MANIFOLD_PAR == 'T'
#include "tbb/tbb.h"
#endif

#include <algorithm>
#include <numeric>
#if MANIFOLD_PAR == 'T' && TBB_INTERFACE_VERSION >= 10000 && \
    __has_include(<pstl/glue_execution_defs.h>)
#include <execution>
#endif
#include <list>
#include <map>
#if __has_include(<memory_resource>)
#include <memory_resource>
#endif
#include <queue>
#include <set>
#include <stack>

#include "optional_assert.h"

namespace {
using namespace manifold;

static ExecutionParams params;

constexpr float kBest = -std::numeric_limits<float>::infinity();

#ifdef MANIFOLD_DEBUG
struct PolyEdge {
  int startVert, endVert;
};

std::vector<PolyEdge> Polygons2Edges(const PolygonsIdx &polys) {
  std::vector<PolyEdge> halfedges;
  for (const auto &poly : polys) {
    for (int i = 1; i < poly.size(); ++i) {
      halfedges.push_back({poly[i - 1].idx, poly[i].idx});
    }
    halfedges.push_back({poly.back().idx, poly[0].idx});
  }
  return halfedges;
}

std::vector<PolyEdge> Triangles2Edges(
    const std::vector<glm::ivec3> &triangles) {
  std::vector<PolyEdge> halfedges;
  halfedges.reserve(triangles.size() * 3);
  for (const glm::ivec3 &tri : triangles) {
    halfedges.push_back({tri[0], tri[1]});
    halfedges.push_back({tri[1], tri[2]});
    halfedges.push_back({tri[2], tri[0]});
  }
  return halfedges;
}

void CheckTopology(const std::vector<PolyEdge> &halfedges) {
  ASSERT(halfedges.size() % 2 == 0, topologyErr, "Odd number of halfedges.");
  size_t n_edges = halfedges.size() / 2;
  std::vector<PolyEdge> forward(halfedges.size()), backward(halfedges.size());

  auto end = std::copy_if(halfedges.begin(), halfedges.end(), forward.begin(),
                          [](PolyEdge e) { return e.endVert > e.startVert; });
  ASSERT(std::distance(forward.begin(), end) == n_edges, topologyErr,
         "Half of halfedges should be forward.");
  forward.resize(n_edges);

  end = std::copy_if(halfedges.begin(), halfedges.end(), backward.begin(),
                     [](PolyEdge e) { return e.endVert < e.startVert; });
  ASSERT(std::distance(backward.begin(), end) == n_edges, topologyErr,
         "Half of halfedges should be backward.");
  backward.resize(n_edges);

  std::for_each(backward.begin(), backward.end(),
                [](PolyEdge &e) { std::swap(e.startVert, e.endVert); });
  auto cmp = [](const PolyEdge &a, const PolyEdge &b) {
    return a.startVert < b.startVert ||
           (a.startVert == b.startVert && a.endVert < b.endVert);
  };
  std::stable_sort(forward.begin(), forward.end(), cmp);
  std::stable_sort(backward.begin(), backward.end(), cmp);
  for (int i = 0; i < n_edges; ++i) {
    ASSERT(forward[i].startVert == backward[i].startVert &&
               forward[i].endVert == backward[i].endVert,
           topologyErr, "Forward and backward edge do not match.");
    // if (i > 0) {
    //   ASSERT(forward[i - 1].startVert != forward[i].startVert ||
    //              forward[i - 1].endVert != forward[i].endVert,
    //          topologyErr, "Not a 2-manifold.");
    //   ASSERT(backward[i - 1].startVert != backward[i].startVert ||
    //              backward[i - 1].endVert != backward[i].endVert,
    //          topologyErr, "Not a 2-manifold.");
    // }
  }
}

void CheckTopology(const std::vector<glm::ivec3> &triangles,
                   const PolygonsIdx &polys) {
  std::vector<PolyEdge> halfedges = Triangles2Edges(triangles);
  std::vector<PolyEdge> openEdges = Polygons2Edges(polys);
  for (PolyEdge e : openEdges) {
    halfedges.push_back({e.endVert, e.startVert});
  }
  CheckTopology(halfedges);
}

void CheckGeometry(const std::vector<glm::ivec3> &triangles,
                   const PolygonsIdx &polys, float precision) {
  std::unordered_map<int, glm::vec2> vertPos;
  for (const auto &poly : polys) {
    for (int i = 0; i < poly.size(); ++i) {
      vertPos[poly[i].idx] = poly[i].pos;
    }
  }
  ASSERT(std::all_of(triangles.begin(), triangles.end(),
                     [&vertPos, precision](const glm::ivec3 &tri) {
                       return CCW(vertPos[tri[0]], vertPos[tri[1]],
                                  vertPos[tri[2]], precision) >= 0;
                     }),
         geometryErr, "triangulation is not entirely CCW!");
}

void Dump(const PolygonsIdx &polys) {
  for (auto poly : polys) {
    std::cout << "polys.push_back({" << std::setprecision(9) << std::endl;
    for (auto v : poly) {
      std::cout << "    {" << v.pos.x << ", " << v.pos.y << "},  //"
                << std::endl;
    }
    std::cout << "});" << std::endl;
  }
  for (auto poly : polys) {
    std::cout << "show(array([" << std::endl;
    for (auto v : poly) {
      std::cout << "  [" << v.pos.x << ", " << v.pos.y << "]," << std::endl;
    }
    std::cout << "]))" << std::endl;
  }
}

void PrintFailure(const std::exception &e, const PolygonsIdx &polys,
                  std::vector<glm::ivec3> &triangles, float precision) {
  std::cout << "-----------------------------------" << std::endl;
  std::cout << "Triangulation failed! Precision = " << precision << std::endl;
  std::cout << e.what() << std::endl;
  Dump(polys);
  std::cout << "produced this triangulation:" << std::endl;
  for (int j = 0; j < triangles.size(); ++j) {
    std::cout << triangles[j][0] << ", " << triangles[j][1] << ", "
              << triangles[j][2] << std::endl;
  }
}

#define PRINT(msg) \
  if (params.verbose) std::cout << msg << std::endl;
#else
#define PRINT(msg)
#endif

/**
 * Ear-clipping triangulator based on David Eberly's approach from Geometric
 * Tools, but adjusted to handle epsilon-valid polygons, and including a
 * fallback that ensures a manifold triangulation even for overlapping polygons.
 * This is an O(n^2) algorithm, but hopefully this is not a big problem as the
 * number of edges in a given polygon is generally much less than the number of
 * triangles in a mesh, and relatively few faces even need triangulation.
 *
 * The main adjustments for robustness involve clipping the sharpest ears first
 * (a known technique to get higher triangle quality), and doing an exhaustive
 * search to determine ear convexity exactly if the first geometric result is
 * within precision.
 */

class EarClip {
 public:
  EarClip(const PolygonsIdx &polys, float precision) : precision_(precision) {
    int numVert = 0;
    for (const SimplePolygonIdx &poly : polys) {
      numVert += poly.size();
    }
    polygon_.reserve(numVert + 2 * polys.size());

    Initialize(polys);
  }

  std::vector<glm::ivec3> Triangulate() {
    for (VertItr v = polygon_.begin(); v != polygon_.end(); ++v) {
      ClipIfDegenerate(v);
    }

    FindStarts();

    CutKeyholes();

    for (const VertItr start : starts_) {
      TriangulatePoly(start);
    }

    return triangles_;
  }

  float GetPrecision() const { return precision_; }

 private:
  struct Vert;
  typedef std::vector<Vert>::iterator VertItr;
  struct MaxX {
    bool operator()(const VertItr &a, const VertItr &b) const {
      return a->pos.x > b->pos.x;
    }
  };
  struct MinCost {
    bool operator()(const VertItr &a, const VertItr &b) const {
      return a->cost < b->cost;
    }
  };
  typedef std::set<VertItr, MinCost>::iterator qItr;

  // The flat list where all the Verts are stored. Not used much for traversal.
  std::vector<Vert> polygon_;
  // The set of right-most starting points, one for each polygon.
  std::multiset<VertItr, MaxX> starts_;
  // Maps each polygon (by way of starting point) to its bounding box.
  std::map<VertItr, Rect> start2BBox_;
  // A priority queue of valid ears - the multiset allows them to be updated.
  std::multiset<VertItr, MinCost> earsQueue_;
  // The output triangulation.
  std::vector<glm::ivec3> triangles_;
  // Working precision: max of float error and input value.
  float precision_;

  // A circularly-linked list representing the polygon(s) that still need to be
  // triangulated. This gets smaller as ears are clipped until it degenerates to
  // two points and terminates.
  struct Vert {
    int mesh_idx;
    qItr ear;
    glm::vec2 pos, rightDir;
    VertItr left, right;
    float cost;

    bool IsShort(float precision) const {
      const glm::vec2 edge = right->pos - pos;
      return glm::dot(edge, edge) < precision * precision;
    }

    // A major key to robustness is to only clip convex ears, but this is
    // difficult to determine when an edge is folded back on itself. This
    // function walks down the kinks in a degenerate portion of a polygon until
    // it finds a clear geometric result. In the vast majority of cases the loop
    // will never run, and when it does, it usually only needs one iteration.
    bool IsConvex(float precision) const {
      const float p2 = precision * precision;
      VertItr nextL = left;
      VertItr nextR = right;
      VertItr center = left->right;
      VertItr last = center;

      while (nextL != nextR) {
        const glm::vec2 edgeL = nextL->pos - center->pos;
        const float l2 = glm::dot(edgeL, edgeL);
        if (l2 <= p2) {
          nextL = nextL->left;
          continue;
        }

        const glm::vec2 edgeR = nextR->pos - center->pos;
        const float r2 = glm::dot(edgeR, edgeR);
        if (r2 <= p2) {
          nextR = nextR->right;
          continue;
        }

        const glm::vec2 vecLR = nextR->pos - nextL->pos;
        const float lr2 = glm::dot(vecLR, vecLR);
        if (lr2 <= p2) {
          last = center;
          center = nextL;
          nextL = nextL->left;
          if (nextL == nextR) break;
          nextR = nextR->right;
          continue;
        }

        int convexity = CCW(nextL->pos, center->pos, nextR->pos, precision);
        if (center != last) {
          convexity += CCW(last->pos, center->pos, nextL->pos, precision) +
                       CCW(nextR->pos, center->pos, last->pos, precision);
        }
        if (convexity != 0) return convexity > 0;

        if (l2 < r2) {
          center = nextL;
          nextL = nextL->left;
        } else {
          center = nextR;
          nextR = nextR->right;
        }
        last = center;
      }
      // The whole polygon is degenerate - consider this to be convex.
      return true;
    }

    // This function is the core of finding a proper place to keyhole. It runs
    // on this Vert, which is represents the edge from this to right. It returns
    // an iterator to the vert to connect to (either this or right) and the
    // x-value of the edge at the given y-level. If the edge is not a valid
    // option for a keyhole (must be upwards and cross the y-value), the x-value
    // is inf.
    //
    // If the edge terminates within the precision band, it checks the next edge
    // to ensure validity. No while loop is necessary because short edges have
    // already been removed. The onTop value is 1 if the y-value is at the top
    // of the polygon's bounding box, -1 if it's at the bottom, and 0 otherwise.
    // This allows proper handling of horizontal edges.
    std::pair<VertItr, float> InterpY2X(float y, int onTop,
                                        float precision) const {
      const float p2 = precision * precision;
      if (pos.y < right->pos.y) {  // Edge goes up
        if (glm::abs(pos.y - y) <= precision) {
          if (glm::abs(right->pos.y - y) > precision) {
            // Tail is at y
            VertItr prev = left;
            if (!(prev->pos.y > y + precision && IsConvex(precision)) &&
                !(onTop == 1 && prev->pos.y > y - precision)) {
              return std::make_pair(left->right, pos.x);
            }
          }  // Edges within the precision band are skipped
        } else {
          if (glm::abs(right->pos.y - y) <= precision) {
            // Head is at y
            VertItr next = right->right;
            if (!(next->pos.y < y - precision && right->IsConvex(precision)) &&
                !(onTop == -1 && next->pos.y <= y + precision)) {
              return std::make_pair(right, right->pos.x);
            }
          } else if (pos.y < y && right->pos.y > y) {
            // Edge crosses y
            float a =
                glm::clamp((y - pos.y) / (right->pos.y - pos.y), 0.0f, 1.0f);
            const float x = glm::mix(pos.x, right->pos.x, a);
            const VertItr p = pos.x < right->pos.x ? right : left->right;
            return std::make_pair(p, x);
          }
        }
      }
      // Edge does not cross y going up
      return std::make_pair(left, std::numeric_limits<float>::infinity());
    }

    // This finds the cost of this vert relative to one of the two closed sides
    // of the ear. Points are valid even when they touch, so long as their edge
    // goes to the outside. No need to check the other side, since all verts are
    // processed in the EarCost loop.
    float SignedDist(VertItr v, glm::vec2 unit, float precision) const {
      float d = glm::determinant(glm::mat2(unit, v->pos - pos));
      if (glm::abs(d) < precision) {
        d = glm::determinant(glm::mat2(unit, v->right->pos - pos));
        return d < precision ? kBest : 0;
      }
      return d;
    }

    // Find the cost of Vert v within this ear, where openSide is the unit
    // vector from Verts right to left - passed in for reuse.
    float Cost(VertItr v, glm::vec2 openSide, float precision) const {
      const glm::vec2 offset = v->pos - pos;
      float cost = SignedDist(v, rightDir, precision);
      if (!glm::isfinite(cost)) {
        return cost;  // Not inside the ear
      }

      cost = glm::min(cost, SignedDist(v, left->rightDir, precision));
      if (!glm::isfinite(cost)) {
        return cost;  // Not inside the ear
      }

      const float openCost =
          glm::determinant(glm::mat2(openSide, v->pos - right->pos));
      return (cost == 0 && glm::abs(openCost) < precision)
                 ? kBest  // Not inside the ear
                 : cost = glm::min(cost, openCost);
    }

    // For verts outside the ear, apply a cost based on the Delaunay condition
    // to aid in prioritization and produce cleaner triangulations. This doesn't
    // affect robustness, but may be adjusted to improve output.
    float DelaunayCost(glm::vec2 diff, float scale, float precision) const {
      return -precision - scale * glm::dot(diff, diff);
    }

    // This is the O(n^2) part of the algorithm, checking this ear against every
    // Vert to ensure none are inside. It may be possible to improve performance
    // by using the Collider to get it down to nlogn or doing some
    // parallelization, but that may be more trouble than it's worth.
    //
    // Think of a cost as vaguely a distance metric - 0 is right on the edge of
    // being invalid. cost > precision is definitely invalid. Cost < -precision
    // is definitely valid, so all improvement costs are designed to always give
    // values < -precision so they will never affect validity. The first
    // totalCost is designed to give priority to sharper angles. Any cost < (-1
    // - precision) has satisfied the Delaunay condition.
    float EarCost(float precision) const {
      glm::vec2 openSide = left->pos - right->pos;
      const glm::vec2 center = 0.5f * (left->pos + right->pos);
      const float scale = 4 / glm::dot(openSide, openSide);
      openSide = glm::normalize(openSide);

      float totalCost = glm::dot(left->rightDir, rightDir) - 1 - precision;
      if (CCW(pos, left->pos, right->pos, precision) == 0) {
        return totalCost < -1 ? kBest : 0;
      }
      VertItr test = right;
      while (test != left) {
        float cost = Cost(test, openSide, precision);
        if (cost < -precision) {
          cost = DelaunayCost(test->pos - center, scale, precision);
        }
        totalCost = glm::max(totalCost, cost);

        test = test->right;
      }
      return totalCost;
    }

    void PrintVert() const {
#ifdef MANIFOLD_DEBUG
      if (!params.verbose) return;
      std::cout << "vert: " << mesh_idx << ", left: " << left->mesh_idx
                << ", right: " << right->mesh_idx << ", cost: " << cost
                << std::endl;
#endif
    }
  };

  // This function and JoinPolygons are the only functions that affect the
  // circular list data structure. This helps ensure it remains circular.
  void Link(VertItr left, VertItr right) const {
    left->right = right;
    right->left = left;
    left->rightDir = glm::normalize(right->pos - left->pos);
    if (!glm::isfinite(left->rightDir.x)) left->rightDir = {0, 0};
  }

  // When an ear vert is clipped, its neighbors get linked, so they get unlinked
  // from it, but it is still linked to them.
  bool Clipped(VertItr v) { return v->right->left != v; }

  // Remove this vert from the circular list and output a corresponding
  // triangle.
  void ClipEar(VertItr ear) {
    Link(ear->left, ear->right);
    if (ear->left->mesh_idx != ear->mesh_idx &&
        ear->mesh_idx != ear->right->mesh_idx &&
        ear->right->mesh_idx != ear->left->mesh_idx) {
      // Filter out topological degenerates, which can form in bad
      // triangulations of polygons with holes, due to vert duplication.
      triangles_.push_back(
          {ear->left->mesh_idx, ear->mesh_idx, ear->right->mesh_idx});
#ifdef MANIFOLD_DEBUG
      if (params.verbose) {
        std::cout << "output tri: " << ear->mesh_idx << ", "
                  << ear->right->mesh_idx << ", " << ear->left->mesh_idx
                  << std::endl;
      }
#endif
    } else {
      PRINT("Topological degenerate!");
    }
  }

  // If an ear will make a degenerate triangle, clip it early to avoid
  // difficulty in key-holing. This function is recursive, as the process of
  // clipping may cause the neighbors to degenerate. Reflex degenerates *must
  // not* be clipped, unless they have a short edge.
  void ClipIfDegenerate(VertItr ear) {
    if (Clipped(ear)) {
      return;
    }
    if (ear->left == ear->right) {
      return;
    }
    if (ear->IsShort(precision_) ||
        (CCW(ear->left->pos, ear->pos, ear->right->pos, precision_) == 0 &&
         glm::dot(ear->left->pos - ear->pos, ear->right->pos - ear->pos) > 0 &&
         ear->IsConvex(precision_))) {
      ClipEar(ear);
      ClipIfDegenerate(ear->left);
      ClipIfDegenerate(ear->right);
    }
  }

  // Build the circular list polygon structures.
  void Initialize(const PolygonsIdx &polys) {
    float bound = 0;
    for (const SimplePolygonIdx &poly : polys) {
      auto vert = poly.begin();
      polygon_.push_back({vert->idx, earsQueue_.end(), vert->pos});
      const VertItr first = std::prev(polygon_.end());
      VertItr last = first;
      // This is not the real rightmost start, but just an arbitrary vert for
      // now to identify each polygon.
      starts_.insert(first);

      for (++vert; vert != poly.end(); ++vert) {
        polygon_.push_back({vert->idx, earsQueue_.end(), vert->pos});
        VertItr next = std::prev(polygon_.end());

        bound = glm::max(
            bound, glm::max(glm::abs(next->pos.x), glm::abs(next->pos.y)));

        Link(last, next);
        last = next;
      }
      Link(last, first);
    }

    if (precision_ < 0) precision_ = bound * kTolerance;

    // Slightly more than enough, since each hole can cause two extra triangles.
    triangles_.reserve(polygon_.size() + 2 * starts_.size());
  }

  // Find the actual rightmost starts after degenerate removal. Also calculate
  // the polygon bounding boxes.
  void FindStarts() {
    std::multiset<VertItr, MaxX> starts;
    for (auto startItr = starts_.begin(); startItr != starts_.end();
         ++startItr) {
      VertItr first = *startItr;
      // This vert may have been clipped during the key-holing process.
      VertItr start = first;
      VertItr v = first;
      float maxX = -std::numeric_limits<float>::infinity();
      Rect bBox;
      do {
        if (Clipped(v)) {
          // Update first to an un-clipped vert so we will return to it instead
          // of infinite-looping.
          first = v->right->left;
          if (!Clipped(first)) {
            bBox.Union(first->pos);
            if (first->pos.x > maxX) {
              maxX = first->pos.x;
              start = first;
            }
          }
        } else {
          bBox.Union(v->pos);
          if (v->pos.x > maxX) {
            maxX = v->pos.x;
            start = v;
          }
        }
        v = v->right;
      } while (v != first);

      // No polygon left if all ears were degenerate and already clipped.
      if (glm::isfinite(maxX)) {
        starts.insert(start);
        start2BBox_.insert({start, bBox});
      }
    }
    starts_ = starts;
  }

  // All holes must be key-holed (attached to an outer polygon) before ear
  // clipping can commence. A polygon is a hole if and only if its start vert is
  // reflex. Instead of relying on sorting, which may be incorrect due to
  // precision, we check for polygon edges both ahead and behind to ensure all
  // valid options are found.
  void CutKeyholes() {
    auto startItr = starts_.begin();
    while (startItr != starts_.end()) {
      const VertItr start = *startItr;

      if (start->IsConvex(precision_)) {  // Outer
        ++startItr;
        continue;
      }

      // Hole
      const float startX = start->pos.x;
      const Rect bBox = start2BBox_[start];
      const int onTop = start->pos.y >= bBox.max.y - precision_   ? 1
                        : start->pos.y <= bBox.min.y + precision_ ? -1
                                                                  : 0;
      float minX = std::numeric_limits<float>::infinity();
      VertItr connector = polygon_.end();
      for (auto poly = starts_.begin(); poly != starts_.end(); ++poly) {
        if (poly == startItr) continue;
        VertItr edge = *poly;
        do {
          const std::pair<VertItr, float> pair =
              edge->InterpY2X(start->pos.y, onTop, precision_);
          const float x = pair.second;
          // This ensures we capture all valid edges, but will choose the same
          // edge as precision == 0 would, if possible.
          if (glm::isfinite(x) && x > startX - precision_ &&
              (!glm::isfinite(minX) || (x >= startX && x < minX) ||
               (minX < startX && x > minX))) {
            minX = x;
            connector = pair.first;
          }
          edge = edge->right;
        } while (edge != *poly);
      }

      if (connector == polygon_.end()) {
        PRINT("hole did not find an outer contour!");
        ++startItr;
        continue;
      }

      connector = FindBridge(start, connector, glm::vec2(minX, start->pos.y));

      JoinPolygons(start, connector);

      // Remove this hole polygon by erasing its start.
      startItr = starts_.erase(startItr);
    }
  }

  // This converts the initial guess for the keyhole location into the final one
  // and returns it. It does so by finding any reflex verts inside the triangle
  // containing the guessed connection and the initial horizontal line, and
  // returning the closest one to the start vert. This function doesn't
  // currently make much use of precision, but it remains to be seen if this may
  // be necessary.
  VertItr FindBridge(VertItr start, VertItr guess,
                     glm::vec2 intersection) const {
    const float above = guess->pos.y > start->pos.y ? 1 : -1;
    VertItr best = guess;
    VertItr vert = guess->right;
    const glm::vec2 left = start->pos - guess->pos;
    const glm::vec2 right = intersection - guess->pos;
    float minD2 = glm::dot(left, left);
    while (vert != guess) {
      const glm::vec2 offset = vert->pos - guess->pos;
      const glm::vec2 diff = vert->pos - start->pos;
      const float d2 = glm::dot(diff, diff);
      if (d2 < minD2 && vert->pos.y * above > start->pos.y * above &&
          above * glm::determinant(glm::mat2(left, offset)) > -precision_ &&
          above * glm::determinant(glm::mat2(offset, right)) > -precision_) {
        const glm::vec2 diffN = d2 > precision_ * precision_
                                    ? glm::normalize(diff)
                                    : glm::vec2(1, 0);
        if (CCW(vert->rightDir, -diffN, -vert->left->rightDir, precision_) >=
            0) {
          minD2 = d2;
          best = vert;
        }
      }
      vert = vert->right;
    }
#ifdef MANIFOLD_DEBUG
    if (params.verbose) {
      std::cout << "connected " << start->mesh_idx << " to " << best->mesh_idx
                << std::endl;
    }
#endif
    return best;
  }

  // Creates a keyhole between the start vert of a hole and the connector vert
  // of an outer polygon. To do this, both verts are duplicated and reattached.
  // This process may create degenerate ears, so these are clipped if necessary
  // to keep from confusing subsequent key-holing operations.
  void JoinPolygons(VertItr start, VertItr connector) {
    polygon_.push_back(*start);
    const VertItr newStart = std::prev(polygon_.end());
    polygon_.push_back(*connector);
    const VertItr newConnector = std::prev(polygon_.end());

    start->right->left = newStart;
    connector->left->right = newConnector;
    Link(start, connector);
    Link(newConnector, newStart);

    ClipIfDegenerate(start);
    ClipIfDegenerate(newStart);
    ClipIfDegenerate(connector);
    ClipIfDegenerate(newConnector);
  }

  // Recalculate the cost of the Vert v ear, updating it in the queue by
  // removing and reinserting it.
  void ProcessEar(VertItr v) {
    if (v->ear != earsQueue_.end()) {
      earsQueue_.erase(v->ear);
      v->ear = earsQueue_.end();
    }
    if (v->IsShort(precision_)) {
      v->cost = kBest;
      v->ear = earsQueue_.insert(v);
    } else if (v->IsConvex(precision_)) {
      v->cost = v->EarCost(precision_);
      v->ear = earsQueue_.insert(v);
    }
  }

  // The main ear-clipping loop. This is called once for each outer polygon -
  // all holes have already been key-holed and joined to an outer polygon.
  void TriangulatePoly(VertItr start) {
    // A simple polygon always creates two fewer triangles than it has verts.
    int numTri = -2;
    earsQueue_.clear();
    VertItr v = start;
    do {
      if (v->left == v->right) {
        return;
      }
      if (Clipped(v)) {
        start = v->right->left;
        if (!Clipped(start)) {
          ProcessEar(start);
          ++numTri;
          start->PrintVert();
        }
      } else {
        ProcessEar(v);
        ++numTri;
        v->PrintVert();
      }
      v = v->right;
    } while (v != start);
    Dump(v);

    while (numTri > 0) {
      const qItr ear = earsQueue_.begin();
      if (ear != earsQueue_.end()) {
        v = *ear;
        // Cost should always be negative, generally < -precision.
        v->PrintVert();
        earsQueue_.erase(ear);
      } else {
        PRINT("No ear found!");
      }

      ClipEar(v);
      --numTri;

      ProcessEar(v->left);
      ProcessEar(v->right);
      // This is a backup vert that is used if the queue is empty (geometrically
      // invalid polygon), to ensure manifoldness.
      v = v->right;
    }

    ASSERT(v->right == v->left, logicErr, "Triangulator error!");
    PRINT("Finished poly");
  }

  void Dump(VertItr start) const {
#ifdef MANIFOLD_DEBUG
    if (!params.verbose) return;
    VertItr v = start;
    std::cout << "show(array([" << std::endl;
    do {
      std::cout << "  [" << v->pos.x << ", " << v->pos.y << "],# "
                << v->mesh_idx << ", cost: " << v->cost << std::endl;
      v = v->right;
    } while (v != start);
    std::cout << "  [" << v->pos.x << ", " << v->pos.y << "],# " << v->mesh_idx
              << std::endl;
    std::cout << "]))" << std::endl;
#endif
  }
};
}  // namespace

namespace manifold {

/**
 * @brief Triangulates a set of &epsilon;-valid polygons. If the input is not
 * &epsilon;-valid, the triangulation may overlap, but will always return a
 * manifold result that matches the input edge directions.
 *
 * @param polys The set of polygons, wound CCW and representing multiple
 * polygons and/or holes. These have 2D-projected positions as well as
 * references back to the original vertices.
 * @param precision The value of &epsilon;, bounding the uncertainty of the
 * input.
 * @return std::vector<glm::ivec3> The triangles, referencing the original
 * vertex indicies.
 */
std::vector<glm::ivec3> TriangulateIdx(const PolygonsIdx &polys,
                                       float precision) {
  std::vector<glm::ivec3> triangles;
  try {
    EarClip triangulator(polys, precision);
    triangles = triangulator.Triangulate();
#ifdef MANIFOLD_DEBUG
    if (params.intermediateChecks) {
      CheckTopology(triangles, polys);
      if (!params.processOverlaps) {
        CheckGeometry(triangles, polys, 2 * triangulator.GetPrecision());
      }
    }
  } catch (const geometryErr &e) {
    if (!params.suppressErrors) {
      PrintFailure(e, polys, triangles, precision);
    }
    throw;
  } catch (const std::exception &e) {
    PrintFailure(e, polys, triangles, precision);
    throw;
#else
  } catch (const std::exception &e) {
#endif
  }
  return triangles;
}

/**
 * @brief Triangulates a set of &epsilon;-valid polygons. If the input is not
 * &epsilon;-valid, the triangulation may overlap, but will always return a
 * manifold result that matches the input edge directions.
 *
 * @param polygons The set of polygons, wound CCW and representing multiple
 * polygons and/or holes.
 * @param precision The value of &epsilon;, bounding the uncertainty of the
 * input.
 * @return std::vector<glm::ivec3> The triangles, referencing the original
 * polygon points in order.
 */
std::vector<glm::ivec3> Triangulate(const Polygons &polygons, float precision) {
  int idx = 0;
  PolygonsIdx polygonsIndexed;
  for (const auto &poly : polygons) {
    SimplePolygonIdx simpleIndexed;
    for (const glm::vec2 &polyVert : poly) {
      simpleIndexed.push_back({polyVert, idx++});
    }
    polygonsIndexed.push_back(simpleIndexed);
  }
  return TriangulateIdx(polygonsIndexed, precision);
}

ExecutionParams &PolygonParams() { return params; }

}  // namespace manifold
