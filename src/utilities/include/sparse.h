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

#pragma once
#include <math.h>

#include "optional_assert.h"
#include "par.h"
#include "public.h"
#include "utils.h"
#include "vec.h"

namespace manifold {

/** @ingroup Private */
class SparseIndices {
  // sparse indices where {p1: q1, p2: q2, ...} are laid out as
  // p1 q1 p2 q2 or q1 p1 q2 p2, depending on endianness
  // such that the indices are sorted by (p << 32) | q
 public:
#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN ||                 \
    defined(__BIG_ENDIAN__) || defined(__ARMEB__) || defined(__THUMBEB__) || \
    defined(__AARCH64EB__) || defined(_MIBSEB) || defined(__MIBSEB) ||       \
    defined(__MIBSEB__)
  static constexpr size_t pOffset = 0;
#elif defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN ||          \
    defined(__LITTLE_ENDIAN__) || defined(__ARMEL__) ||                    \
    defined(__THUMBEL__) || defined(__AARCH64EL__) || defined(_MIPSEL) ||  \
    defined(__MIPSEL) || defined(__MIPSEL__) || defined(__EMSCRIPTEN__) || \
    defined(_WIN32)
  static constexpr size_t pOffset = 1;
#else
#error "unknown architecture"
#endif
  static constexpr int64_t EncodePQ(int p, int q) {
    return (int64_t(p) << 32) | q;
  }

  SparseIndices() = default;
  SparseIndices(size_t size) { data_ = Vec<char>(size * sizeof(int64_t)); }

  size_t size() const { return data_.size() / sizeof(int64_t); }

  Vec<int> Copy(bool use_q) const {
    Vec<int> out(size());
    size_t offset = pOffset;
    if (use_q) offset = 1 - offset;
    const int* p = ptr();
    for_each(autoPolicy(out.size()), countAt(0_z), countAt(out.size()),
             [&](size_t i) { out[i] = p[i * 2 + offset]; });
    return out;
  }

  void Sort() {
    VecView<int64_t> view = AsVec64();
    stable_sort(autoPolicy(size()), view.begin(), view.end());
  }

  void Resize(size_t size) { data_.resize(size * sizeof(int64_t), -1); }

  inline int& Get(size_t i, bool use_q) {
    if (use_q)
      return ptr()[2 * i + 1 - pOffset];
    else
      return ptr()[2 * i + pOffset];
  }

  inline int Get(size_t i, bool use_q) const {
    if (use_q)
      return ptr()[2 * i + 1 - pOffset];
    else
      return ptr()[2 * i + pOffset];
  }

  inline int64_t GetPQ(size_t i) const {
    VecView<const int64_t> view = AsVec64();
    return view[i];
  }

  inline void Set(size_t i, int p, int q) {
    VecView<int64_t> view = AsVec64();
    view[i] = EncodePQ(p, q);
  }

  inline void SetPQ(size_t i, int64_t pq) {
    VecView<int64_t> view = AsVec64();
    view[i] = pq;
  }

  VecView<int64_t> AsVec64() {
    return VecView<int64_t>(reinterpret_cast<int64_t*>(data_.data()),
                            data_.size() / sizeof(int64_t));
  }

  VecView<const int64_t> AsVec64() const {
    return VecView<const int64_t>(
        reinterpret_cast<const int64_t*>(data_.data()),
        data_.size() / sizeof(int64_t));
  }

  VecView<int32_t> AsVec32() {
    return VecView<int32_t>(reinterpret_cast<int32_t*>(data_.data()),
                            data_.size() / sizeof(int32_t));
  }

  VecView<const int32_t> AsVec32() const {
    return VecView<const int32_t>(
        reinterpret_cast<const int32_t*>(data_.data()),
        data_.size() / sizeof(int32_t));
  }

  inline void Add(int p, int q) {
    for (unsigned int i = 0; i < sizeof(int64_t); ++i) data_.push_back(-1);
    Set(size() - 1, p, q);
  }

  void Unique() {
    Sort();
    VecView<int64_t> view = AsVec64();
    size_t newSize = std::unique(view.begin(), view.end()) - view.begin();
    Resize(newSize);
  }

  size_t RemoveZeros(Vec<int>& S) {
    DEBUG_ASSERT(S.size() == size(), userErr,
                 "Different number of values than indicies!");

    Vec<int> new2Old(S.size());
    auto policy = autoPolicy(S.size());
    sequence(policy, new2Old.begin(), new2Old.end());

    size_t size = copy_if<decltype(new2Old.begin())>(
                      policy, countAt(0_z), countAt(S.size()), new2Old.begin(),
                      [&S](const int i) { return S[i] != 0; }) -
                  new2Old.begin();
    new2Old.resize(size);

    Permute(S, new2Old);
    Vec<char> tmp(std::move(data_));
    Resize(size);
    gather(policy, new2Old.begin(), new2Old.end(),
           reinterpret_cast<int64_t*>(tmp.data()),
           reinterpret_cast<int64_t*>(data_.data()));

    return size;
  }

  template <typename T>
  struct firstFinite {
    VecView<T> v;

    bool Finite(float v) const { return isfinite(v); }
    bool Finite(glm::vec2 v) const { return isfinite(v[0]); }
    bool Finite(glm::vec3 v) const { return isfinite(v[0]); }
    bool Finite(glm::vec4 v) const { return isfinite(v[0]); }

    bool operator()(const int i) const { return Finite(v[i]); }
  };

  template <typename T>
  size_t KeepFinite(Vec<T>& v, Vec<int>& x) {
    DEBUG_ASSERT(x.size() == size(), userErr,
                 "Different number of values than indicies!");

    Vec<int> new2Old(v.size());
    auto policy = autoPolicy(v.size());

    size_t size = copy_if<decltype(new2Old.begin())>(
                      policy, countAt(0_z), countAt(v.size()), new2Old.begin(),
                      firstFinite<T>({v})) -
                  new2Old.begin();
    new2Old.resize(size);

    Permute(v, new2Old);
    Permute(x, new2Old);
    Vec<char> tmp(std::move(data_));
    Resize(size);
    gather(policy, new2Old.begin(), new2Old.end(),
           reinterpret_cast<int64_t*>(tmp.data()),
           reinterpret_cast<int64_t*>(data_.data()));

    return size;
  }

#ifdef MANIFOLD_DEBUG
  void Dump() const {
    std::cout << "SparseIndices = " << std::endl;
    const int* p = ptr();
    for (size_t i = 0; i < size(); ++i) {
      std::cout << i << ", p = " << Get(i, false) << ", q = " << Get(i, true)
                << std::endl;
    }
    std::cout << std::endl;
  }
#endif

 private:
  Vec<char> data_;
  inline int* ptr() { return reinterpret_cast<int32_t*>(data_.data()); }
  inline const int* ptr() const {
    return reinterpret_cast<const int32_t*>(data_.data());
  }
};

}  // namespace manifold
