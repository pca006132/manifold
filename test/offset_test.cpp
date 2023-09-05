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

#include <algorithm>

#include "manifold.h"
#include "meshIO.h"
#include "samples.h"
#include "test.h"

TEST(Offset, Sphere) {
  Manifold sphere = Manifold::Sphere(5);
  sphere += sphere.Translate({12, 0, 0});
  std::cout << sphere.OffsetDecomposition(2).size() << std::endl;
}

TEST(Offset, Cube) {
  Manifold Cube = Manifold::Cube(glm::vec3(10.0f));
  Cube += Cube.Translate({11.5, 0, 0});
  Cube += Cube.Translate({0, 11.5, 0});
  std::cout << Cube.OffsetDecomposition(1).size() << std::endl;
}

TEST(Offset, CubeC) {
  Manifold Cube = Manifold::Cube(glm::vec3(20.0f, 20.0f, 10.0f));
  Cube -= Manifold::Cube(glm::vec3(10.0f)).Translate({10.0f, 5.0f, 0});
  std::cout << Cube.OffsetDecomposition(1).size() << std::endl;
}

TEST(Offset, CubeC2) {
  Manifold Cube = Manifold::Cube(glm::vec3(20.0f, 20.0f, 10.0f));
  Cube -=
      Manifold::Cube(glm::vec3(10.0f, 2.0f, 10.0f)).Translate({10.0f, 9.0f, 0});
  std::cout << Cube.OffsetDecomposition(1).size() << std::endl;
}
