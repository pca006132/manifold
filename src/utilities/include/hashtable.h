// Copyright 2022 The Manifold Authors.
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
#include <stdint.h>

#include <atomic>

#include "par.h"
#include "public.h"
#include "utils.h"
#include "vec.h"

namespace {
typedef unsigned long long int Uint64;
typedef Uint64 (*hash_fun_t)(Uint64);
inline constexpr Uint64 kOpen = std::numeric_limits<Uint64>::max();
inline constexpr Uint64 kTombstone = std::numeric_limits<Uint64>::max() - 1;

template <typename T>
inline T AtomicCAS(T& target, T compare, T val) {
  std::atomic<T>& tar = reinterpret_cast<std::atomic<T>&>(target);
  tar.compare_exchange_strong(compare, val, std::memory_order_acq_rel);
  return compare;
}

template <typename T>
inline void AtomicStore(T& target, T val) {
  std::atomic<T>& tar = reinterpret_cast<std::atomic<T>&>(target);
  // release is good enough, although not really something general
  tar.store(val, std::memory_order_release);
}

template <typename T>
inline T AtomicLoad(const T& target) {
  const std::atomic<T>& tar = reinterpret_cast<const std::atomic<T>&>(target);
  // acquire is good enough, although not general
  return tar.load(std::memory_order_acquire);
}
}  // namespace

namespace manifold {
/** @addtogroup Private
 *  @{
 */

// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
inline Uint64 hash64bit(Uint64 x) {
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  x = x ^ (x >> 31);
  return x;
}

template <typename V, hash_fun_t H = hash64bit>
class HashTableD {
 public:
  HashTableD(Vec<Uint64>& keys, Vec<V>& values, Vec<uint32_t>& used,
             uint32_t step = 1)
      : step_{step}, keys_{keys}, values_{values}, used_{used} {}

  int Size() const { return keys_.size(); }

  bool Full() const { return AtomicLoad(used_[0]) * 2 > Size(); }

  void Insert(Uint64 key, const V& val) {
    uint32_t idx = H(key) % (Size() - 1);
    int count = 0;
    while (1) {
      // if (Full()) return;
      Uint64& k = keys_[idx];
      const Uint64 found = AtomicCAS(k, kOpen, key);
      if (__builtin_expect(found == kOpen, 1)) {
        std::atomic<unsigned int>& tar =
            reinterpret_cast<std::atomic<unsigned int>&>(used_[0]);
        tar.fetch_add(1, std::memory_order_relaxed);
        values_[idx] = val;
        return;
      }
      if (found == key) {
        return;
      }
      idx = (idx + step_) % (Size() - 1);
    }
  }

  V& operator[](Uint64 key) {
    uint32_t idx = H(key) % (Size() - 1);
    while (1) {
      const Uint64 k = AtomicLoad(keys_[idx]);
      if (k == key || k == kOpen) {
        return values_[idx];
      }
      idx = (idx + step_) % (Size() - 1);
    }
  }

  const V& operator[](Uint64 key) const {
    uint32_t idx = H(key) % (Size() - 1);
    while (1) {
      const Uint64 k = AtomicLoad(keys_[idx]);
      if (k == key || k == kOpen) {
        return values_[idx];
      }
      idx = (idx + step_) % (Size() - 1);
    }
  }

  inline int GetIdx(Uint64 key) const {
    uint32_t idx = H(key) % (Size() - 1);
    while (1) {
      const Uint64 k = AtomicLoad(keys_[idx]);
      if (__builtin_expect(k == key || k == kOpen, 1)) {
        return idx;
      }
      idx = (idx + step_) % (Size() - 1);
    }
  }

  inline void RemoveKey(int idx) {
    keys_[idx] = kTombstone;
    // AtomicStore(keys_[idx], kTombstone);
    std::atomic<unsigned int>& tar =
        reinterpret_cast<std::atomic<unsigned int>&>(used_[0]);
    tar.fetch_sub(1, std::memory_order_relaxed);
  }

  inline Uint64 KeyAt(int idx) const { return keys_[idx]; }
  V& At(int idx) { return values_[idx]; }
  const V& At(int idx) const { return values_[idx]; }

 private:
  uint32_t step_;
  VecView<Uint64> keys_;
  VecView<V> values_;
  VecView<uint32_t> used_;
};

template <typename V, hash_fun_t H = hash64bit>
class HashTable {
 public:
  using Store = HashTableD<V, H>;

  HashTable(uint32_t size, uint32_t step = 1)
      : step_(step), table_{keys_, values_, used_, step} {
    int upperbound = std::max(1 << (int)ceil(log2(size) + 0.5), 64) - 1;
    keys_ = Vec<Uint64>(upperbound, -1);
    values_ = Vec<V>(upperbound, {});
    table_ = HashTableD<V, H>{keys_, values_, used_, step};
  }

  HashTableD<V, H> D() { return table_; }

  const HashTableD<V, H> D() const { return table_; }

  int Entries() const { return used_[0]; }

  int Size() const { return table_.Size(); }

  bool Full() const { return used_[0] * 2 > Size(); }

  float FilledFraction() const { return static_cast<float>(used_[0]) / Size(); }

  void Resize() {
    if (used_[0] < Size() / 8) {
      int oldSize = Size();
      int newSize = std::max(1 << (int)ceil(log2(used_[0]) + 1), 64) - 1;
      used_[0] = 0;
      Vec<Uint64> newKeys(newSize, kOpen);
      Vec<V> newValues(newSize, {});
      HashTableD<V, H> newTable(newKeys, newValues, used_, step_);
      for_each(autoPolicy(oldSize), countAt(0), countAt(oldSize), [&](int idx) {
        Uint64 key = table_.KeyAt(idx);
        if (key == kOpen || key == kTombstone) return;
        newTable.Insert(key, std::move(table_.At(idx)));
      });
      keys_ = std::move(newKeys);
      values_ = std::move(newValues);
      table_ = HashTableD<V, H>(keys_, values_, used_, step_);
    }
  }

  // destroy this hashtable and get the underlying store
  std::pair<Vec<Uint64>, Vec<V>> Move() {
    used_[0] = 0;
    return std::make_pair(std::move(keys_), std::move(values_));
  }

  Vec<V>& GetValueStore() { return values_; }

  static constexpr Uint64 Open() { return kOpen; }
  static constexpr Uint64 Tombstone() { return kTombstone; }

 private:
  int step_;
  Vec<Uint64> keys_;
  Vec<V> values_;
  Vec<uint32_t> used_ = Vec<uint32_t>(1, 0);
  HashTableD<V, H> table_;
};

/** @} */
}  // namespace manifold
