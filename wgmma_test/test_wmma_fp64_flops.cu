#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "gmma_desc.h"
//#include "mma_sm90.hpp"

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

template <> void init_data(double *data, int count, std::string method) {
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




//using namespace cute;

struct SM90_16x8x16_F64F64F64F64_TN
{
  using DRegisters = double[4];
  using ARegisters = double[8];
  using BRegisters = double[4];
  using CRegisters = double[4];

  __device__ static void
  fma(double      & d0, double      & d1, double      & d2, double      & d3,
      double const& a0, double const& a1, double const& a2, double const& a3,
      double const& a4, double const& a5, double const& a6, double const& a7,
      double const& b0, double const& b1, double const& b2, double const& b3,
      double const& c0, double const& c1, double const& c2, double const& c3)
  {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64"
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7,  %8,  %9,  %10, %11},"
      "{%12, %13, %14, %15},"
      "{%16, %17, %18, %19};\n"
      : "=d"(d0), "=d"(d1), "=d"(d2), "=d"(d3)
      :  "d"(a0),  "d"(a1),  "d"(a2),  "d"(a3),
         "d"(a4),  "d"(a5),  "d"(a6),  "d"(a7),
         "d"(b0),  "d"(b1),  "d"(b2),  "d"(b3),
         "d"(c0),  "d"(c1),  "d"(c2),  "d"(c3));
  }
};


__global__ void
wmma_m16n8k16_fp64_test(double *gm_d, double *gm_a, double *gm_b,         
                                      double *gm_c,
                                      uint64_t repeat_time) {                        
  constexpr int M = 16;
  constexpr int N = 8;
  constexpr int K = 16;
  constexpr int loop_inner = 32;
  using WMMA_t = SM90_16x8x16_F64F64F64F64_TN;
  WMMA_t::CRegisters reg_d;                                                          
  WMMA_t::ARegisters reg_a;
  WMMA_t::BRegisters reg_b;

  int laneid = threadIdx.x % 32;

  int reg_a_id = 0;
  #pragma unroll
  for (int i = laneid; i < M * K; i += 32) {                        
    reg_a[reg_a_id] = gm_a[i];
    ++reg_a_id;
  }                                                                                  
                                                                                     
  int reg_b_id = 0;                                                                                     
  #pragma unroll
  for (int i = laneid; i < N * K; i += 32) {                        
    reg_b[reg_b_id] = gm_b[i];
    ++reg_b_id;
  }                                                                                  
  #pragma unroll                                                                                   
  for (int i = 0; i < sizeof(reg_d)/sizeof(double); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
                                                                                     
  __syncthreads();                                                                   
                                                                                     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    #pragma unroll
    for (int i = 0; i < loop_inner; ++i) {
      WMMA_t::fma(reg_d[0], reg_d[1], reg_d[2], reg_d[3],
                  reg_a[0], reg_a[1], reg_a[2], reg_a[3], reg_a[4], reg_a[5], reg_a[6], reg_a[7],
                  reg_b[0], reg_b[1], reg_b[2], reg_b[3],
                  reg_d[0],reg_d[1], reg_d[2], reg_d[3]);
    }
  }


  for (int i = 0; i < sizeof(reg_d)/sizeof(double); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m16n8k16_double(int argc, char **argv) {
  constexpr int M = 16;
  constexpr int N = 8;
  constexpr int K = 16;

  std::string init_method(argv[1]);

  int mat_a_size = M * K;
  int mat_b_size = N * K;
  int mat_c_size = M * N;

  double *mat_a_host = new double[mat_a_size];
  init_data(mat_a_host, mat_a_size, init_method);

  double *mat_b_host = new double[mat_b_size];
  init_data(mat_b_host, mat_b_size, init_method);

  double *mat_c_host = new double[mat_c_size];
  init_data(mat_c_host, mat_c_size, init_method);

  double *mat_d_host = new double[mat_c_size];
  init_data(mat_d_host, mat_c_size, init_method);

  double *mat_a_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_a_dev, mat_a_size * sizeof(double)));
  gpuErrchk(cudaMemcpy(mat_a_dev, mat_a_host, mat_a_size * sizeof(double),
                       cudaMemcpyHostToDevice));

  double *mat_b_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_b_dev, mat_b_size * sizeof(double)));
  gpuErrchk(cudaMemcpy(mat_b_dev, mat_b_host, mat_b_size * sizeof(double),
                       cudaMemcpyHostToDevice));

  double *mat_c_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_c_dev, mat_c_size * sizeof(double)));
  gpuErrchk(cudaMemcpy(mat_c_dev, mat_c_host, mat_c_size * sizeof(double),
                       cudaMemcpyHostToDevice));

  double *mat_d_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_d_dev, mat_c_size * sizeof(double)));
  gpuErrchk(cudaMemcpy(mat_d_dev, mat_d_host, mat_c_size * sizeof(double),
                       cudaMemcpyHostToDevice));


  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  int num_warp = 8;
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

    wmma_m16n8k16_fp64_test<<<sm_count, 32 * num_warp>>>(
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

  test_m16n8k16_double(argc, argv);
}

