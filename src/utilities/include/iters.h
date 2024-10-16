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
#pragma once

#include <iterator>
#include <type_traits>

namespace manifold {

template <typename Iter, typename = void>
struct InnerIter {
  using pointer = typename std::iterator_traits<Iter>::pointer;
  using reference = typename std::iterator_traits<Iter>::reference;
  using difference_type = typename std::iterator_traits<Iter>::difference_type;
  using value_type = typename std::iterator_traits<Iter>::value_type;
  using iterator_category =
      typename std::iterator_traits<Iter>::iterator_category;
};

template <typename Iter>
struct InnerIter<Iter, typename std::enable_if_t<std::is_pointer_v<Iter>>> {
  using pointer = Iter;
  using reference = std::remove_pointer_t<Iter>&;
  using difference_type = std::ptrdiff_t;
  using value_type = std::remove_pointer_t<Iter>;
  using iterator_category = std::random_access_iterator_tag;
};

template <typename F, typename Iter>
struct TransformIterator {
 private:
  Iter iter;
  F f;

 public:
  // users are not suppposed to take pointer/reference of the iterator.
  using pointer = void;
  using reference = void;
  using difference_type = typename InnerIter<Iter>::difference_type;
  using value_type =
      std::invoke_result_t<F, typename InnerIter<Iter>::value_type>;
  using iterator_category = typename InnerIter<Iter>::iterator_category;

  TransformIterator(Iter iter, F f) : iter(iter), f(f) {}

  value_type operator*() const { return f(*iter); }

  value_type operator[](size_t i) const { return f(iter[i]); }

  // prefix increment
  TransformIterator& operator++() {
    iter += 1;
    return *this;
  }

  // postfix
  TransformIterator operator++(int) {
    auto old = *this;
    operator++();
    return old;
  }

  TransformIterator operator+(size_t n) const {
    return TransformIterator(iter + n, f);
  }

  TransformIterator& operator+=(size_t n) {
    iter += n;
    return *this;
  }

  friend bool operator==(TransformIterator a, TransformIterator b) {
    return a.iter == b.iter;
  }

  friend bool operator!=(TransformIterator a, TransformIterator b) {
    return !(a.iter == b.iter);
  }

  friend bool operator<(TransformIterator a, TransformIterator b) {
    return a.iter < b.iter;
  }

  friend difference_type operator-(TransformIterator a, TransformIterator b) {
    return a.iter - b.iter;
  }

  operator TransformIterator<F, const Iter>() const {
    return TransformIterator(f, iter);
  }
};

template <typename T>
struct CountingIterator {
 private:
  T counter;

 public:
  using pointer = void;
  using reference = T;
  using difference_type = std::make_signed_t<T>;
  using value_type = T;
  using iterator_category = std::random_access_iterator_tag;

  CountingIterator(T counter) : counter(counter) {}

  value_type operator*() const { return counter; }
  value_type operator[](T i) const { return counter + i; }

  // prefix increment
  CountingIterator& operator++() {
    counter += 1;
    return *this;
  }

  // postfix
  CountingIterator operator++(int) {
    auto old = *this;
    operator++();
    return old;
  }

  CountingIterator operator+(T n) const {
    return CountingIterator(counter + n);
  }

  CountingIterator& operator+=(T n) {
    counter += n;
    return *this;
  }

  friend bool operator==(CountingIterator a, CountingIterator b) {
    return a.counter == b.counter;
  }

  friend bool operator!=(CountingIterator a, CountingIterator b) {
    return a.counter != b.counter;
  }

  friend bool operator<(CountingIterator a, CountingIterator b) {
    return a.counter < b.counter;
  }

  friend difference_type operator-(CountingIterator a, CountingIterator b) {
    return a.counter - b.counter;
  }

  operator CountingIterator<const T>() const {
    return CountingIterator(counter);
  }
};

template <typename T>
CountingIterator<T> countAt(T i) {
  return CountingIterator(i);
}

template <typename Iter>
struct StridedRange {
 private:
  struct StridedRangeIter {
   private:
    Iter iter;
    size_t stride;

   public:
    using pointer = typename InnerIter<Iter>::pointer;
    using reference = typename InnerIter<Iter>::reference;
    using difference_type = typename InnerIter<Iter>::difference_type;
    using value_type = typename InnerIter<Iter>::value_type;
    using iterator_category = typename InnerIter<Iter>::iterator_category;

    StridedRangeIter(Iter iter, int stride) : iter(iter), stride(stride) {}

    value_type& operator*() { return *iter; }

    const value_type& operator*() const { return *iter; }

    value_type& operator[](size_t i) { return iter[i * stride]; }

    const value_type& operator[](size_t i) const { return iter[i * stride]; }

    // prefix increment
    StridedRangeIter& operator++() {
      iter += stride;
      return *this;
    }

    // postfix
    StridedRangeIter operator++(int) {
      auto old = *this;
      operator++();
      return old;
    }

    StridedRangeIter operator+(size_t n) const {
      return StridedRangeIter(iter + n * stride, stride);
    }

    StridedRangeIter& operator+=(size_t n) {
      iter += n * stride;
      return *this;
    }

    friend bool operator==(StridedRangeIter a, StridedRangeIter b) {
      return a.iter == b.iter;
    }

    friend bool operator!=(StridedRangeIter a, StridedRangeIter b) {
      return !(a.iter == b.iter);
    }

    friend bool operator<(StridedRangeIter a, StridedRangeIter b) {
      return a.iter < b.iter;
    }

    friend difference_type operator-(StridedRangeIter a, StridedRangeIter b) {
      // note that this is not well-defined if a.stride != b.stride...
      return (a.iter - b.iter) / a.stride;
    }
  };
  Iter _start, _end;
  const size_t stride;

 public:
  StridedRange(Iter start, Iter end, size_t stride)
      : _start(start), _end(end), stride(stride) {}

  StridedRangeIter begin() const { return StridedRangeIter(_start, stride); }

  StridedRangeIter end() const {
    return StridedRangeIter(_start, stride) +
           ((std::distance(_start, _end) + (stride - 1)) / stride);
  }
};

}  // namespace manifold
