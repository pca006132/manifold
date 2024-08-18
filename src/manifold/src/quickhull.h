// Copyright 2024 The Manifold Authors.
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
//
// Derived from the public domain work of Antti Kuukka at
// https://github.com/akuukka/quickhull

/*
 * INPUT:  a list of points in 3D space (for example, vertices of a 3D mesh)
 *
 * OUTPUT: a ConvexHull object which provides vertex and index buffers of the
 *generated convex hull as a triangle mesh.
 *
 *
 *
 * The implementation is thread-safe if each thread is using its own QuickHull
 *object.
 *
 *
 * SUMMARY OF THE ALGORITHM:
 *         - Create initial simplex (tetrahedron) using extreme points. We have
 *four faces now and they form a convex mesh M.
 *         - For each point, assign them to the first face for which they are on
 *the positive side of (so each point is assigned to at most one face). Points
 *inside the initial tetrahedron are left behind now and no longer affect the
 *calculations.
 *         - Add all faces that have points assigned to them to Face Stack.
 *         - Iterate until Face Stack is empty:
 *              - Pop topmost face F from the stack
 *              - From the points assigned to F, pick the point P that is
 *farthest away from the plane defined by F.
 *              - Find all faces of M that have P on their positive side. Let us
 *call these the "visible faces".
 *              - Because of the way M is constructed, these faces are
 *connected. Solve their horizon edge loop.
 *				- "Extrude to P": Create new faces by connecting
 *P with the points belonging to the horizon edge. Add the new faces to M and
 *remove the visible faces from M.
 *              - Each point that was assigned to visible faces is now assigned
 *to at most one of the newly created faces.
 *              - Those new faces that have points assigned to them are added to
 *the top of Face Stack.
 *          - M is now the convex hull.
 *
 * */
#pragma once
#include <array>
#include <deque>
#include <limits>
#include <vector>

#include "shared.h"
#include "vec.h"

namespace manifold {

class ConvexHull {
 public:
  Vec<Halfedge> halfEdges;
  std::vector<glm::dvec3> vertices;
  ConvexHull(Vec<Halfedge> halfEdges, std::vector<glm::dvec3> vertices)
      : halfEdges(halfEdges), vertices(vertices) {}
  ConvexHull() = default;
};

// Pool.hpp
class Pool {
  std::vector<std::unique_ptr<Vec<size_t>>> data;

 public:
  void clear() { data.clear(); }

  void reclaim(std::unique_ptr<Vec<size_t>>& ptr) {
    data.push_back(std::move(ptr));
  }

  std::unique_ptr<Vec<size_t>> get() {
    if (data.size() == 0) {
      return std::make_unique<Vec<size_t>>();
    }
    auto it = data.end() - 1;
    std::unique_ptr<Vec<size_t>> r = std::move(*it);
    data.erase(it);
    return r;
  }
};

// Plane.hpp

class Plane {
 public:
  glm::dvec3 N;

  // Signed distance (if normal is of length 1) to the plane from origin
  double D;

  // Normal length squared
  double sqrNLength;

  bool isPointOnPositiveSide(const glm::dvec3& Q) const {
    double d = glm::dot(N, Q) + D;
    if (d >= 0) return true;
    return false;
  }

  Plane() = default;

  // Construct a plane using normal N and any point P on the plane
  Plane(const glm::dvec3& N, const glm::dvec3& P)
      : N(N), D(glm::dot(-N, P)), sqrNLength(glm::dot(N, N)) {}
};

// Ray.hpp

struct Ray {
  const glm::dvec3 S;
  const glm::dvec3 V;
  const double VInvLengthSquared;

  Ray(const glm::dvec3& S, const glm::dvec3& V)
      : S(S), V(V), VInvLengthSquared(1 / (glm::dot(V, V))) {}
};

// Mesh.hpp

class MeshBuilder {
 public:
  struct HalfEdge {
    size_t endVertex;
    size_t opp;
    size_t face;
    size_t next;

    HalfEdge(size_t endVertex, size_t opp, size_t face, size_t next)
        : endVertex(endVertex), opp(opp), face(face), next(next) {}
    HalfEdge() = default;

    void disable() { endVertex = std::numeric_limits<size_t>::max(); }

    bool isDisabled() const {
      return endVertex == std::numeric_limits<size_t>::max();
    }
  };

  struct Face {
    size_t he;
    Plane P{};
    double mostDistantPointDist = 0.0;
    size_t mostDistantPoint = 0;
    size_t visibilityCheckedOnIteration = 0;
    std::uint8_t isVisibleFaceOnCurrentIteration : 1;
    std::uint8_t inFaceStack : 1;
    // Bit for each half edge assigned to this face, each being 0 or 1 depending
    // on whether the edge belongs to horizon edge
    std::uint8_t horizonEdgesOnCurrentIteration : 3;
    std::unique_ptr<Vec<size_t>> pointsOnPositiveSide;

    Face(size_t he)
        : he(he),
          isVisibleFaceOnCurrentIteration(0),
          inFaceStack(0),
          horizonEdgesOnCurrentIteration(0) {}

    Face()
        : he(std::numeric_limits<size_t>::max()),
          isVisibleFaceOnCurrentIteration(0),
          inFaceStack(0),
          horizonEdgesOnCurrentIteration(0) {}

    void disable() { he = std::numeric_limits<size_t>::max(); }

    bool isDisabled() const { return he == std::numeric_limits<size_t>::max(); }
  };

  // Mesh data
  std::vector<Face> faces;
  std::vector<HalfEdge> halfEdges;

  // When the mesh is modified and faces and half edges are removed from it, we
  // do not actually remove them from the container vectors. Insted, they are
  // marked as disabled which means that the indices can be reused when we need
  // to add new faces and half edges to the mesh. We store the free indices in
  // the following vectors.
  std::vector<size_t> disabledFaces, disabledHalfEdges;

  size_t addFace();

  size_t addHalfEdge();

  // Mark a face as disabled and return a pointer to the points that were on the
  // positive of it.
  std::unique_ptr<Vec<size_t>> disableFace(size_t faceIndex) {
    auto& f = faces[faceIndex];
    f.disable();
    disabledFaces.push_back(faceIndex);
    return std::move(f.pointsOnPositiveSide);
  }

  void disableHalfEdge(size_t heIndex) {
    HalfEdge& he = halfEdges[heIndex];
    he.disable();
    disabledHalfEdges.push_back(heIndex);
  }

  MeshBuilder() = default;

  // Create a mesh with initial tetrahedron ABCD. Dot product of AB with the
  // normal of triangle ABC should be negative.
  void setup(size_t a, size_t b, size_t c, size_t d);

  std::array<size_t, 3> getVertexIndicesOfFace(const Face& f) const;

  std::array<size_t, 2> getVertexIndicesOfHalfEdge(const HalfEdge& he) const {
    return {halfEdges[he.opp].endVertex, he.endVertex};
  }

  std::array<size_t, 3> getHalfEdgeIndicesOfFace(const Face& f) const {
    return {f.he, halfEdges[f.he].next, halfEdges[halfEdges[f.he].next].next};
  }
};

// HalfEdgeMesh.hpp

class HalfEdgeMesh {
 public:
  struct HalfEdge {
    size_t endVertex;
    size_t opp;
    size_t face;
    size_t next;
    HalfEdge(size_t endVertex, size_t opp, size_t face, size_t next)
        : endVertex(endVertex), opp(opp), face(face), next(next) {}
    HalfEdge() = default;
  };

  struct Face {
    // Index of one of the half edges of this face
    size_t halfEdgeIndex;

    Face(size_t halfEdgeIndex) : halfEdgeIndex(halfEdgeIndex) {}
  };

  std::vector<glm::dvec3> vertices;
  std::vector<Face> faces;
  std::vector<HalfEdge> halfEdges;

  HalfEdgeMesh(const MeshBuilder& builderObject,
               const VecView<glm::dvec3>& vertexData);
};

// QuickHull.hpp

struct DiagnosticsData {
  // How many times QuickHull failed to solve the horizon edge. Failures lead
  // to degenerated convex hulls.
  size_t failedHorizonEdges;

  DiagnosticsData() : failedHorizonEdges(0) {}
};

double defaultEps();

class QuickHull {
  using vec3 = glm::dvec3;

  double m_epsilon, epsilonSquared, scale;
  bool planar;
  std::vector<vec3> planarPointCloudTemp;
  VecView<glm::dvec3> originalVertexData;
  MeshBuilder mesh;
  std::array<size_t, 6> extremeValues;
  DiagnosticsData diagnostics;

  // Temporary variables used during iteration process
  std::vector<size_t> newFaceIndices;
  std::vector<size_t> newHalfEdgeIndices;
  std::vector<std::unique_ptr<Vec<size_t>>> disabledFacePointVectors;
  std::vector<size_t> visibleFaces;
  std::vector<size_t> horizonEdgesData;
  struct FaceData {
    size_t faceIndex;
    // If the face turns out not to be visible, this half edge will be marked as
    // horizon edge
    size_t enteredFromHalfEdge;
    FaceData(size_t fi, size_t he) : faceIndex(fi), enteredFromHalfEdge(he) {}
  };
  std::vector<FaceData> possiblyVisibleFaces;
  std::deque<size_t> faceList;

  // Create a half edge mesh representing the base tetrahedron from which the
  // QuickHull iteration proceeds. extremeValues must be properly set up when
  // this is called.
  void setupInitialTetrahedron();

  // Given a list of half edges, try to rearrange them so that they form a loop.
  // Return true on success.
  bool reorderHorizonEdges(std::vector<size_t>& horizonEdges);

  // Find indices of extreme values (max x, min x, max y, min y, max z, min z)
  // for the given point cloud
  std::array<size_t, 6> getExtremeValues();

  // Compute scale of the vertex data.
  double getScale(const std::array<size_t, 6>& extremeValuesInput);

  // Each face contains a unique pointer to a vector of indices. However, many -
  // often most - faces do not have any points on the positive side of them
  // especially at the the end of the iteration. When a face is removed from the
  // mesh, its associated point vector, if such exists, is moved to the index
  // vector pool, and when we need to add new faces with points on the positive
  // side to the mesh, we reuse these vectors. This reduces the amount of
  // std::vectors we have to deal with, and impact on performance is remarkable.
  Pool indexVectorPool;
  inline std::unique_ptr<Vec<size_t>> getIndexVectorFromPool();
  inline void reclaimToIndexVectorPool(std::unique_ptr<Vec<size_t>>& ptr);

  // Associates a point with a face if the point resides on the positive side of
  // the plane. Returns true if the points was on the positive side.
  inline bool addPointToFace(typename MeshBuilder::Face& f, size_t pointIndex);

  // This will update mesh from which we create the ConvexHull object that
  // getConvexHull function returns
  void createConvexHalfEdgeMesh();

  // Constructs the convex hull into halfEdges and NewVerts
  ConvexHull buildMesh(const VecView<glm::dvec3>& pointCloud, bool CCW,
                       double eps);

 public:
  QuickHull(const Vec<glm::dvec3>& pointCloudVec)
      : originalVertexData(VecView(pointCloudVec)) {}

  // Computes convex hull for a given point cloud. This function assumes that
  // the vertex data resides in memory in the following format:
  // x_0,y_0,z_0,x_1,y_1,z_1,... Params:
  //   vertexData: pointer to the X component of the first point of the point
  //   cloud. vertexCount: number of vertices in the point cloud CCW: whether
  //   the output mesh triangles should have CCW orientation eps: minimum
  //   distance to a plane to consider a point being on positive side of it (for
  //   a point cloud with scale 1)
  // Returns:
  //   Convex hull of the point cloud as halfEdge vector and vertex vector
  ConvexHull getConvexHullAsMesh(const Vec<glm::dvec3>& pointCloud, bool CCW,
                                 double epsilon = defaultEps()) {
    Vec<glm::dvec3> pointCloudVec(pointCloud);
    QuickHull qh(pointCloudVec);
    return qh.buildMesh(pointCloudVec, CCW, epsilon);
  }

  // Get diagnostics about last generated convex hull
  const DiagnosticsData& getDiagnostics() { return diagnostics; }
};
}  // namespace manifold
