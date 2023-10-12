// Copyright 2020 The Manifold Authors.
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

#include <chrono>
#include <iostream>

#include "manifold.h"
#include "vec.h"

using namespace manifold;

constexpr int BTREE_WAYS = 4;

constexpr int BTREE_LUT_COMPUTE(int level) {
  if (level == 0) return 0;
  return BTREE_LUT_COMPUTE(level - 1) * BTREE_WAYS + BTREE_WAYS;
}

constexpr int BTREE_LUT[] = {
    BTREE_LUT_COMPUTE(0),  BTREE_LUT_COMPUTE(1),  BTREE_LUT_COMPUTE(2),
    BTREE_LUT_COMPUTE(3),  BTREE_LUT_COMPUTE(4),  BTREE_LUT_COMPUTE(5),
    BTREE_LUT_COMPUTE(6),  BTREE_LUT_COMPUTE(7),  BTREE_LUT_COMPUTE(8),
    BTREE_LUT_COMPUTE(9),  BTREE_LUT_COMPUTE(10), BTREE_LUT_COMPUTE(11),
    BTREE_LUT_COMPUTE(12), BTREE_LUT_COMPUTE(13),
};

void btree_construct_rec(int level, VecView<int64_t> dest,
                         VecView<const int64_t> sorted) {
  int partitionSize = BTREE_LUT[level - 1];
  int low = 0;
  int high = partitionSize;
  for (int i = 0; i < BTREE_WAYS; ++i) {
    if (level > 1)
      btree_construct_rec(
          level - 1, dest.view(BTREE_WAYS + i * partitionSize),
          sorted.view(low, std::min(high, sorted.size()) - low));
    if (high >= sorted.size()) {
      dest[i] = std::numeric_limits<int64_t>::max();
      return;
    }
    dest[i] = sorted[high];
    low = high + 1;
    high += partitionSize + 1;
  }
}

std::pair<Vec<int64_t>, int> btree_construct(VecView<const int64_t> sorted) {
  int levels = 1;
  while (BTREE_LUT[levels] < sorted.size()) {
    levels += 1;
  }
  // we just overapproximate the size needed
  Vec<int64_t> result(sorted.size() + (BTREE_WAYS - 1) * levels);
  btree_construct_rec(levels, result, sorted);
  return std::make_pair(result, levels);
}

// this returns the index in the original sorted array
inline int btree_search(int64_t key, VecView<const int64_t> tree, int levels) {
  int low = 0;
  const int64_t *treeptr = tree.cbegin();
  int offset = 0;
  while (1) {
    int i = 0;
#pragma GCC unroll 4
    for (; i < BTREE_WAYS; ++i) {
      if (key <= treeptr[i + offset]) break;
      low += BTREE_LUT[levels - 1] + 1;
    }
    if (i < BTREE_WAYS && key == treeptr[i + offset])
      return low + BTREE_LUT[levels - 1];
    if (levels-- == 1 || i >= BTREE_WAYS) return -1;
    offset += BTREE_WAYS + i * BTREE_LUT[levels];
  }
}

int main(int argc, char **argv) {
  Vec<int64_t> test(100);
  std::iota(test.begin(), test.end(), 0);
  auto btree = btree_construct(test);
  for (int i = 0; i < 101; i++)
    std::cout << btree_search(i, btree.first, btree.second) << std::endl;
  return 0;

  // for (int i = 0; i < 8; ++i) {
  //   Manifold sphere = Manifold::Sphere(1, (8 << i) * 4);
  //   Manifold sphere2 = sphere.Translate(glm::vec3(0.5));
  //   auto start = std::chrono::high_resolution_clock::now();
  //   Manifold diff = sphere - sphere2;
  //   diff.NumTri();
  //   auto end = std::chrono::high_resolution_clock::now();
  //   std::chrono::duration<double> elapsed = end - start;
  //   std::cout << "nTri = " << sphere.NumTri() << ", time = " <<
  //   elapsed.count()
  //             << " sec" << std::endl;
  // }
}
