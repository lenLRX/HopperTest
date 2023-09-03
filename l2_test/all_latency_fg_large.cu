#include <fstream>
#include <iostream>
#include <stdint.h>

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define THREADS_NUM 1024
#define WARP_SIZE 32

uint32_t *MakeChaseBuffer(uint32_t size, uint32_t stride) {
  uint32_t element_count = size / sizeof(uint32_t);
  uint32_t *result = new uint32_t[element_count + WARP_SIZE * 2];
  int round = element_count / WARP_SIZE;
  for (uint32_t i = 0; i < round; ++i) {
    for (uint32_t j = 0; j < WARP_SIZE; ++j) {
      result[i * WARP_SIZE + j] = stride / sizeof(uint32_t);
    }
  }
  return result;
}

__global__ void l2_chase(uint32_t *duration, uint32_t *dsink,
                         uint32_t *posArray, uint32_t *l2_size_input) {
  uint32_t l2_size = l2_size_input[0];
  uint32_t iter_num = 4096;
  // thread index
  uint32_t tid = threadIdx.x;
  // a register to avoid compiler optimization
  uint32_t sink = 0;
  // populate l2 cache to warm up
  /*
  for (uint32_t i = blockIdx.x * blockDim.x + tid; i < l2_size; i += blockDim.x * gridDim.x) {
    uint32_t *ptr = (uint32_t *)posArray + i;
    uint32_t idx;
    asm volatile("{\t\n"
                 "ld.global.cg.u32 %0, [%1];\n\t"
                 "}"
                 : "=r"(idx)
                 : "l"(ptr)
                 : "memory");
    sink += idx;
  }
  */
  // synchronize all threads
  asm volatile("bar.sync 0;");
  if (blockIdx.x > 0) {
    return;
  }

  if (threadIdx.x > 0) {
    return;
  }

  __shared__ uint32_t s_tvalue[4096];
  __shared__ uint32_t s_index[4096];

  posArray = posArray + tid;

  for (uint32_t i = 0; i < iter_num; ++i) {
    // start timing
    uint32_t start = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");
    uint32_t idx;
    asm volatile("{\t\n"
                 "ld.global.ca.u32 %0, [%1];\n\t"
                 "}"
                 : "=r"(idx)
                 : "l"(posArray)
                 : "memory");
    posArray += idx;
    s_index[i] = idx;
    // stop timing
    uint32_t stop = 0;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");
    s_tvalue[i] = stop - start - 35; // 35 is overhead of clock
  }
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // write time and data back to memory
  for (uint32_t i = 0; i < iter_num; ++i) {
    duration[i] = s_tvalue[i];
    dsink[i] = s_index[i] + sink;
  }
}

void TestL2Latency(uint32_t test_cache_size, uint32_t stride, std::ofstream &ofs) {
  uint32_t l2_element_count = test_cache_size / sizeof(uint32_t);

  uint32_t *duration_host = new uint32_t[l2_element_count];
  uint32_t *chaseBuffer_host = MakeChaseBuffer(test_cache_size, stride);

  uint32_t *chaseBuffer_dev;

  cudaMalloc(&chaseBuffer_dev,
             (l2_element_count + WARP_SIZE) * sizeof(uint32_t));
  uint32_t *duration_dev;
  uint32_t *dsink_dev;
  uint32_t *l2_size_dev;
  cudaMalloc(&duration_dev, l2_element_count * sizeof(uint32_t));
  cudaMalloc(&dsink_dev, l2_element_count * sizeof(uint32_t));
  cudaMalloc(&l2_size_dev, sizeof(uint32_t));

  cudaMemcpy(l2_size_dev, &l2_element_count, sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(chaseBuffer_dev, chaseBuffer_host,
             (l2_element_count + WARP_SIZE) * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;

  l2_chase<<<sm_count, THREADS_NUM>>>(duration_dev, dsink_dev,
                               chaseBuffer_dev, l2_size_dev);

  gpuErrchk(cudaDeviceSynchronize());
  cudaMemcpy(duration_host, duration_dev, l2_element_count * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  int offset = 0;

  for (int i = 0; i < 4096; ++i) {
    ofs << offset << "," << duration_host[i] << "\n";
    std::cout << offset << "," << duration_host[i] << "\n";
    offset += stride;
  }

  double clock_rate = prop.clockRate * 1000;
  // std::cout << "Device freq:" << prop.clockRate << "khz" << std::endl;



  cudaFree(chaseBuffer_dev);
  cudaFree(duration_dev);
  cudaFree(dsink_dev);
  cudaFree(l2_size_dev);

  delete[] duration_host;
  delete[] chaseBuffer_host;
}

int main() {
  std::ofstream ofs("mem_fg_all_large.csv");
  ofs << "offset(byte),latency(cycle)\n";
  TestL2Latency(100*1024*1024, 32, ofs);

}
