#include "thrust/sort.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <execution>
#include <iostream>
#include <random>

#include "radix_sort.h"
#include "tbb/parallel_for.h"
#include "thrust/system/cpp/execution_policy.h"
#include "thrust/system/tbb/execution_policy.h"

template <typename T>
auto genRandom(size_t kSize) {
  std::vector<T> data(kSize);
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<T> dist;
  for (int i = 0; i < kSize; ++i) {
    data[i] = dist(rng);
  }
  return data;
}

template <typename T, typename F>
int bench(const std::vector<T> &data, F f) {
  std::vector<T> v(data.begin(), data.end());
  auto start = std::chrono::high_resolution_clock::now();
  f(v.begin(), v.end());
  auto end = std::chrono::high_resolution_clock::now();
  if (!std::is_sorted(v.begin(), v.end()))
    std::cerr << "sort failed" << std::endl;
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

template <typename T>
void bench_sort() {
  auto std_sort = [](auto begin, auto end) { std::sort(begin, end); };
  auto std_sort_mt = [](auto begin, auto end) {
    std::sort(std::execution::par_unseq, begin, end);
  };
  auto thrust_sort = [](auto begin, auto end) {
    thrust::sort(thrust::cpp::tag(), begin, end);
  };
  auto thrust_sort_mt = [](auto begin, auto end) {
    thrust::sort(thrust::tbb::tag(), begin, end);
  };
  auto thrust_stable_sort_mt = [](auto begin, auto end) {
    thrust::stable_sort(thrust::tbb::tag(), begin, end);
  };
  auto pca_sort_mt = [](auto begin, auto end) {
    manifold::radix_sort(&*begin, (int)std::distance(begin, end));
  };

  std::cout << "random array test" << std::endl;
  for (size_t size = 8; size <= 26; size += 2) {
    std::cout << std::endl;
    std::cout << "size: 1 << " << size << std::endl;
    int times[6] = {0};
    for (int i = 0; i < 5; i++) {
      auto data = genRandom<T>(1 << size);
      times[0] += bench(data, std_sort);
      times[1] += bench(data, std_sort_mt);
      times[2] += bench(data, thrust_sort);
      times[3] += bench(data, thrust_sort_mt);
      times[4] += bench(data, thrust_stable_sort_mt);
      times[5] += bench(data, pca_sort_mt);
    }
    std::cout << "std::sort(seq)     : " << times[0] / 5 << " us" << std::endl;
    std::cout << "std::sort(par)     : " << times[1] / 5 << " us" << std::endl;
    std::cout << "thrust(seq)        : " << times[2] / 5 << " us" << std::endl;
    std::cout << "thrust(par)        : " << times[3] / 5 << " us" << std::endl;
    std::cout << "thrust(stable, par): " << times[4] / 5 << " us" << std::endl;
    std::cout << "pca (par)          : " << times[5] / 5 << " us" << std::endl;
  }

  std::cout << std::endl << "sorted array with random tail test" << std::endl;
  for (size_t size = 8; size <= 26; size += 2) {
    std::cout << std::endl;
    std::cout << "size: 1 << " << size << std::endl;
    int times[6] = {0};
    for (int i = 0; i < 5; i++) {
      auto data = genRandom<T>((1 << size) - (1 << 5));
      thrust_sort_mt(data.begin(), data.end());
      auto data2 = genRandom<T>(1 << 5);
      data.insert(data.begin(), data2.begin(), data2.end());

      times[0] += bench(data, std_sort);
      times[1] += bench(data, std_sort_mt);
      times[2] += bench(data, thrust_sort);
      times[3] += bench(data, thrust_sort_mt);
      times[4] += bench(data, thrust_stable_sort_mt);
      times[5] += bench(data, pca_sort_mt);
    }
    std::cout << "std::sort(seq)     : " << times[0] / 5 << " us" << std::endl;
    std::cout << "std::sort(par)     : " << times[1] / 5 << " us" << std::endl;
    std::cout << "thrust(seq)        : " << times[2] / 5 << " us" << std::endl;
    std::cout << "thrust(par)        : " << times[3] / 5 << " us" << std::endl;
    std::cout << "thrust(stable, par): " << times[4] / 5 << " us" << std::endl;
    std::cout << "pca (par)          : " << times[5] / 5 << " us" << std::endl;
  }
}

int main(int argc, char **argv) {
  std::cout << "int32_t" << std::endl;
  bench_sort<int32_t>();

  std::cout << std::endl << "=================" << std::endl;
  std::cout << "int64_t" << std::endl;
  std::cout << "=================" << std::endl << std::endl;
  bench_sort<int64_t>();
  return 0;
}
