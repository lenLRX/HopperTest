#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "gmma_desc.h"
#include "mma_sm80.hpp"

// code from
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

template <typename T> void init_data(T *data, int count, std::string method);

template <> void init_data(__half *data, int count, std::string method) {
  if (method == "uniform") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < count; ++i) {
      data[i] = __float2half(dis(gen));
    }
  } else {
    for (int i = 0; i < count; ++i) {
      data[i] = __float2half(0);
    }
  }
}

template <> void init_data(float *data, int count, std::string method) {
  if (method == "uniform") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < count; ++i) {
      data[i] = dis(gen);
    }
  } else {
    for (int i = 0; i < count; ++i) {
      data[i] = 0;
    }
  }
}



using namespace cute;


__global__ void
wmma_m16n8k16_fmix_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
                                      float *gm_c,
                                      uint64_t repeat_time) {                        
  constexpr int M = 16;
  constexpr int N = 8;
  constexpr int K = 16;
  constexpr int loop_inner = 32;
  using WMMA_t = SM80_16x8x16_F32F16F16F32_TN;    
  WMMA_t::CRegisters reg_d;                                                          
  WMMA_t::ARegisters reg_a;
  WMMA_t::BRegisters reg_b;

  int laneid = threadIdx.x % 32;

  int reg_a_id = 0;
  #pragma unroll
  for (int i = laneid; i < M * K / 2; i += 32) {                        
    reinterpret_cast<__half2&>(reg_a[reg_a_id]) = gm_a[i];
    ++reg_a_id;
  }                                                                                  
                                                                                     
  int reg_b_id = 0;                                                                                     
  #pragma unroll
  for (int i = laneid; i < N * K / 2; i += 32) {                        
    reinterpret_cast<__half2&>(reg_b[reg_b_id]) = gm_b[i];
    ++reg_b_id;
  }                                                                                  
  #pragma unroll                                                                                   
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
                                                                                     
  __syncthreads();                                                                   
                                                                                     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    #pragma unroll
    for (int i = 0; i < loop_inner; ++i) {
      WMMA_t::fma(reg_d[0], reg_d[1], reg_d[2], reg_d[3],
                  reg_a[0], reg_a[1], reg_a[2], reg_a[3],
                  reg_b[0], reg_b[1],	       
                  reg_d[0],reg_d[1],reg_d[2],reg_d[3]);
    }
  }


  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m16n8k16_fmix(int argc, char **argv) {
  constexpr int M = 16;
  constexpr int N = 8;
  constexpr int K = 16;

  std::string init_method(argv[1]);

  int mat_a_size = M * K;
  int mat_b_size = N * K;
  int mat_c_size = M * N;

  __half *mat_a_host = new __half[mat_a_size];
  init_data(mat_a_host, mat_a_size, init_method);

  __half *mat_b_host = new __half[mat_b_size];
  init_data(mat_b_host, mat_b_size, init_method);

  float *mat_c_host = new float[mat_c_size];
  init_data(mat_c_host, mat_c_size, init_method);

  float *mat_d_host = new float[mat_c_size];
  init_data(mat_d_host, mat_c_size, init_method);

  __half2 *mat_a_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_a_dev, mat_a_size * sizeof(__half)));
  gpuErrchk(cudaMemcpy(mat_a_dev, mat_a_host, mat_a_size * sizeof(__half),
                       cudaMemcpyHostToDevice));

  __half2 *mat_b_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_b_dev, mat_b_size * sizeof(__half)));
  gpuErrchk(cudaMemcpy(mat_b_dev, mat_b_host, mat_b_size * sizeof(__half),
                       cudaMemcpyHostToDevice));

  float *mat_c_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_c_dev, mat_c_size * sizeof(float)));
  gpuErrchk(cudaMemcpy(mat_c_dev, mat_c_host, mat_c_size * sizeof(float),
                       cudaMemcpyHostToDevice));

  float *mat_d_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_d_dev, mat_c_size * sizeof(float)));
  gpuErrchk(cudaMemcpy(mat_d_dev, mat_d_host, mat_c_size * sizeof(float),
                       cudaMemcpyHostToDevice));


  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  int num_warp = 32;
  constexpr int loop_inner = 32;
  std::cout << "using SM count: " << sm_count
            << std::endl;

  int repeat_time = 102400;
  

  while (1){
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wmma_m16n8k16_fmix_test<<<sm_count, 32 * num_warp>>>(
        mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;
    FLOPS *= num_warp;
    FLOPS *= loop_inner;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
              << std::endl;

  }
}




int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << " usage ./test  <init method>\n";
    return -1;
  }

  test_m16n8k16_fmix(argc, argv);
}

