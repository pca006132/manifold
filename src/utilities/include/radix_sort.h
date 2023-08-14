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

#pragma once

#include <tbb/tbb.h>

#include <algorithm>

namespace {
constexpr size_t kSeqThreshold = 1 << 16;

template <typename N, const int K>
struct Hist {
  using SizeType = N;
  static constexpr int k = K;
  N hist[k][256] = {{0}};
  void merge(const Hist<N, K>& other) {
    for (int i = 0; i < k; ++i)
      for (int j = 0; j < 256; ++j) hist[i][j] += other.hist[i][j];
  }
  void prefixSum(int total, bool* canSkip) {
    for (int i = 0; i < k; ++i) {
      size_t count = 0;
      for (int j = 0; j < 256; ++j) {
        size_t tmp = hist[i][j];
        hist[i][j] = count;
        count += tmp;
        if (tmp == total) canSkip[i] = true;
      }
    }
  }
};

template <typename T, typename H>
void histogram(T* ptr, typename H::SizeType n, H& hist) {
  auto worker = [](T* ptr, typename H::SizeType n, H& hist) {
    for (typename H::SizeType i = 0; i < n; ++i)
      for (int k = 0; k < hist.k; ++k)
        ++hist.hist[k][(ptr[i] >> (8 * k)) & 0xFF];
  };
  if (n < kSeqThreshold) {
    worker(ptr, n, hist);
  } else {
    tbb::combinable<H> store;
    tbb::parallel_for(
        tbb::blocked_range<typename H::SizeType>(0, n, kSeqThreshold),
        [&worker, &store, ptr](const auto& r) {
          worker(ptr + r.begin(), r.end() - r.begin(), store.local());
        });
    store.combine_each([&hist](const H& h) { hist.merge(h); });
  }
}

template <typename T, typename H>
void shuffle(T* src, T* target, typename H::SizeType n, H& hist, int k) {
  int prev = -1;
  int consecutive = 0;
  typename H::SizeType i = 0;
#pragma GCC unroll 8
  while (i < n) {
    int byte = (src[i] >> (8 * k)) & 0xFF;
    target[hist.hist[k][byte]++] = src[i];
    i++;
    // only trigger if we got a long sequence
    if (__builtin_expect(byte == prev && ++consecutive == 64, 0)) {
      typename H::SizeType j = i;
      while (j < n && ((src[j] >> (8 * k)) & 0xFF) == byte) ++j;
      if (j - i > kSeqThreshold)
        std::copy(std::execution::par, src + i, src + j,
                  target + hist.hist[k][byte]);
      else
        std::copy(src + i, src + j, target + hist.hist[k][byte]);
      hist.hist[k][byte] += j - i;
      i = j;
      consecutive = 0;
    } else if (byte != prev) {
      consecutive = 0;
      prev = byte;
    }
  }
}

template <typename T, typename SizeType>
bool LSB_radix_sort(T* input, T* tmp, SizeType n) {
  Hist<SizeType, sizeof(T) / sizeof(char)> hist;
  if (std::is_sorted(input, input + n)) return false;
  histogram(input, n, hist);
  bool canSkip[hist.k] = {0};
  hist.prefixSum(n, canSkip);
  T *a = input, *b = tmp;
  for (int k = 0; k < hist.k; ++k) {
    if (!canSkip[k]) {
      shuffle(a, b, n, hist, k);
      std::swap(a, b);
    }
  }
  return a == tmp;
}

// LSB radix sort with merge
template <typename T, typename SizeType>
struct SortedRange {
  T *input, *tmp;
  SizeType offset = 0, length = 0;
  bool inTmp = false;

  SortedRange(T* input, T* tmp, SizeType offset = 0, SizeType length = 0)
      : input(input), tmp(tmp), offset(offset), length(length) {}
  SortedRange(SortedRange<T, SizeType>& r, tbb::split)
      : input(r.input), tmp(r.tmp) {}
  void operator()(const tbb::blocked_range<SizeType>& range) {
    SortedRange<T, SizeType> rhs(input, tmp, range.begin(),
                                 range.end() - range.begin());
    rhs.inTmp =
        LSB_radix_sort(input + rhs.offset, tmp + rhs.offset, rhs.length);
    if (length == 0)
      *this = rhs;
    else
      join(rhs);
  }
  bool swapBuffer() const {
    T *src = input, *target = tmp;
    if (inTmp) std::swap(src, target);
    std::copy(std::execution::par_unseq, src + offset, src + offset + length,
              target + offset);
    return !inTmp;
  }
  void join(const SortedRange<T, SizeType>& rhs) {
    if (inTmp != rhs.inTmp) {
      if (length < rhs.length)
        inTmp = swapBuffer();
      else
        rhs.swapBuffer();
    }
    T *src = input, *target = tmp;
    if (inTmp) std::swap(src, target);
    if (src[offset + length - 1] > src[rhs.offset]) {
      // we depends on the performance of this merge function
      // can probably optimize a bit more...
      std::merge(std::execution::par_unseq, src + offset, src + offset + length,
                 src + rhs.offset, src + rhs.offset + rhs.length,
                 target + offset);
      inTmp = !inTmp;
    }
    length += rhs.length;
  }
};
}  // namespace

namespace manifold {
// btw, this is a stable sort
template <typename T, typename SizeTy>
void radix_sort(T* input, SizeTy n) {
  T* aux = new T[n];
  SizeTy blockSize = std::max(n / tbb::this_task_arena::max_concurrency() / 4,
                              static_cast<SizeTy>(kSeqThreshold / sizeof(T)));
  SortedRange<T, SizeTy> result = SortedRange<T, SizeTy>(input, aux);
  tbb::parallel_reduce(tbb::blocked_range<SizeTy>(0, n, blockSize), result);
  if (result.inTmp) std::copy(aux, aux + n, input);
  delete[] aux;
}
}  // namespace manifold
