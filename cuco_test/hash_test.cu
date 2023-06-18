/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuco/dynamic_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <limits>


template <typename T>
std::vector<T> make_unique_vector(size_t gen_num) {
  std::vector<T> result;
  std::random_device rd;  // a seed source for the random number engine
  std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<T> distrib(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

  std::unordered_set<T> u;
  size_t i = 0;
  while (i < gen_num) {
    T value = distrib(gen);
    if (u.count(value) == 0) {
      result.push_back(value);
      u.insert(value);
      ++i;
    }
  }

  return result;
}


/**
 * @file host_bulk_example.cu
 * @brief Demonstrates usage of the static_map "bulk" host APIs.
 *
 * The bulk APIs are only invocable from the host and are used for doing operations like insert or
 * find on a set of keys.
 *
 */

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "usage ./test.exe <init loadfactor>\n";
  }
  double init_load_factor = atof(argv[1]);
  using Key   = int64_t;
  using Value = int;

  // Empty slots are represented by reserved "sentinel" values. These values should be selected such
  // that they never occur in your input data.
  Key constexpr empty_key_sentinel     = -1;
  Value constexpr empty_value_sentinel = -1;

  // static capacity
  std::size_t const capacity = 2550000;

  std::size_t const init_insert_num = capacity * init_load_factor;

  std::size_t const test_num = 406900;

  // Constructs a map with "capacity" slots using -1 and -1 as the empty key/value sentinels.
  cuco::dynamic_map<Key, Value> map{
    capacity, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel}};

  thrust::device_vector<Key> init_keys(init_insert_num);
  thrust::device_vector<Value> init_values(init_insert_num);

  init_keys = make_unique_vector<Key>(init_insert_num);
  init_values = make_unique_vector<Value>(init_insert_num);

  thrust::device_vector<Key> insert_keys(test_num);
  thrust::device_vector<Value> insert_values(test_num);

  insert_keys = make_unique_vector<Key>(test_num);
  insert_values = make_unique_vector<Value>(test_num);


  auto init_zipped =
    thrust::make_zip_iterator(thrust::make_tuple(init_keys.begin(), init_values.begin()));

  // Inserts all pairs into the map
  map.insert(init_zipped, init_zipped + init_insert_num);

  if (init_insert_num > test_num) {
    thrust::device_vector<Value> found_values(test_num);
    map.find(insert_keys.begin(), insert_keys.begin() + test_num, found_values.begin());
  }

  auto test_zipped =
    thrust::make_zip_iterator(thrust::make_tuple(insert_keys.begin(), insert_values.begin()));

  map.insert(test_zipped, test_zipped + test_num);

  return 0;
}
