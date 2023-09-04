#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "gmma_desc.h"
#include "mma_sm90_gmma.hpp"

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


// N = 256
__global__ void
wgmma_m64n256k16_fmix_SS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
                                      float *gm_c, uint8_t sm_layout,               
                                      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 256;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x256x16_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
                                                                                    
  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];                                                               
  }                                                                                  
                                                                                     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
                                                                                     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
                                                                                     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
                                                                                     
  __syncthreads();                                                                   
                                                                                     
  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
                                                                                     
  GmmaDescriptor desc_a, desc_b;                                                     
  desc_a.layout_type_ = sm_layout;                                                   
  desc_b.layout_type_ = sm_layout;                                                   
                                                                                     
  desc_a.start_address_ = sm_a_addr >> 4;                                            
  desc_b.start_address_ = sm_b_addr >> 4;                                            
                                                                                     
  desc_a.base_offset_ = 0;                                                           
  desc_b.base_offset_ = 0;                                                           
                                                                                     
  desc_a.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
                                                                                     
  desc_a.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
                                                                                     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(desc_a.desc_, desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3],reg_d[4],reg_d[5],reg_d[6],reg_d[7],
reg_d[8],reg_d[9],reg_d[10],reg_d[11],reg_d[12],reg_d[13],reg_d[14],reg_d[15],
reg_d[16],reg_d[17],reg_d[18],reg_d[19],reg_d[20],reg_d[21],reg_d[22],reg_d[23],
reg_d[24],reg_d[25],reg_d[26],reg_d[27],reg_d[28],reg_d[29],reg_d[30],reg_d[31],
reg_d[32],reg_d[33],reg_d[34],reg_d[35],reg_d[36],reg_d[37],reg_d[38],reg_d[39],
reg_d[40],reg_d[41],reg_d[42],reg_d[43],reg_d[44],reg_d[45],reg_d[46],reg_d[47],
reg_d[48],reg_d[49],reg_d[50],reg_d[51],reg_d[52],reg_d[53],reg_d[54],reg_d[55],
reg_d[56],reg_d[57],reg_d[58],reg_d[59],reg_d[60],reg_d[61],reg_d[62],reg_d[63],
reg_d[64],reg_d[65],reg_d[66],reg_d[67],reg_d[68],reg_d[69],reg_d[70],reg_d[71],
reg_d[72],reg_d[73],reg_d[74],reg_d[75],reg_d[76],reg_d[77],reg_d[78],reg_d[79],
reg_d[80],reg_d[81],reg_d[82],reg_d[83],reg_d[84],reg_d[85],reg_d[86],reg_d[87],
reg_d[88],reg_d[89],reg_d[90],reg_d[91],reg_d[92],reg_d[93],reg_d[94],reg_d[95],
reg_d[96],reg_d[97],reg_d[98],reg_d[99],reg_d[100],reg_d[101],reg_d[102],reg_d[103],
reg_d[104],reg_d[105],reg_d[106],reg_d[107],reg_d[108],reg_d[109],reg_d[110],reg_d[111],
reg_d[112],reg_d[113],reg_d[114],reg_d[115],reg_d[116],reg_d[117],reg_d[118],reg_d[119],
reg_d[120],reg_d[121],reg_d[122],reg_d[123],reg_d[124],reg_d[125],reg_d[126],reg_d[127]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n256k16_fmix_SS(int argc, char **argv) {
  constexpr int M = 64;
  constexpr int N = 256;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n256k16_fmix_SS_test,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
            << " dynamic shared size: " << dyn_shared_size << "bytes"
            << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n256k16_fmix_SS_test<<<sm_count, 128, dyn_shared_size>>>(
        mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Shared "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
              << std::endl;
  }
}

__global__ void
wgmma_m64n256k16_fmix_RS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
                                      float *gm_c, uint8_t sm_layout,               
                                      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 256;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x256x16_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
  GMMA_t::ARegisters reg_a;

  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];
  }                                                                                  
                                                                                     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
                                                                                     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
                                                                                     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
                                                                                     
  __syncthreads();                                                                   

  for (int i = 0; i < sizeof(reg_a)/sizeof(__half2); i += 1) {                        
    reinterpret_cast<__half2&>(reg_a[i]) = sm_a[i];
  }                                                                                  
                                                                                     
  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
                                                                                     
  GmmaDescriptor desc_b;                                                     
  desc_b.layout_type_ = sm_layout;                                                   
                                                                                     
  desc_b.start_address_ = sm_b_addr >> 4;                                            
                                                                                     
  desc_b.base_offset_ = 0;                                                           
                                                                                     
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
                                                                                     
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
                                                                                     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(reg_a[0], reg_a[1], reg_a[2], reg_a[3], desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3],reg_d[4],reg_d[5],reg_d[6],reg_d[7],
reg_d[8],reg_d[9],reg_d[10],reg_d[11],reg_d[12],reg_d[13],reg_d[14],reg_d[15],
reg_d[16],reg_d[17],reg_d[18],reg_d[19],reg_d[20],reg_d[21],reg_d[22],reg_d[23],
reg_d[24],reg_d[25],reg_d[26],reg_d[27],reg_d[28],reg_d[29],reg_d[30],reg_d[31],
reg_d[32],reg_d[33],reg_d[34],reg_d[35],reg_d[36],reg_d[37],reg_d[38],reg_d[39],
reg_d[40],reg_d[41],reg_d[42],reg_d[43],reg_d[44],reg_d[45],reg_d[46],reg_d[47],
reg_d[48],reg_d[49],reg_d[50],reg_d[51],reg_d[52],reg_d[53],reg_d[54],reg_d[55],
reg_d[56],reg_d[57],reg_d[58],reg_d[59],reg_d[60],reg_d[61],reg_d[62],reg_d[63],
reg_d[64],reg_d[65],reg_d[66],reg_d[67],reg_d[68],reg_d[69],reg_d[70],reg_d[71],
reg_d[72],reg_d[73],reg_d[74],reg_d[75],reg_d[76],reg_d[77],reg_d[78],reg_d[79],
reg_d[80],reg_d[81],reg_d[82],reg_d[83],reg_d[84],reg_d[85],reg_d[86],reg_d[87],
reg_d[88],reg_d[89],reg_d[90],reg_d[91],reg_d[92],reg_d[93],reg_d[94],reg_d[95],
reg_d[96],reg_d[97],reg_d[98],reg_d[99],reg_d[100],reg_d[101],reg_d[102],reg_d[103],
reg_d[104],reg_d[105],reg_d[106],reg_d[107],reg_d[108],reg_d[109],reg_d[110],reg_d[111],
reg_d[112],reg_d[113],reg_d[114],reg_d[115],reg_d[116],reg_d[117],reg_d[118],reg_d[119],
reg_d[120],reg_d[121],reg_d[122],reg_d[123],reg_d[124],reg_d[125],reg_d[126],reg_d[127]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n256k16_fmix_RS(int argc, char **argv) {
  constexpr int M = 64;
  constexpr int N = 256;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n256k16_fmix_RS_test,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
            << " dynamic shared size: " << dyn_shared_size << "bytes"
            << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n256k16_fmix_RS_test<<<sm_count, 128, dyn_shared_size>>>(
        mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Reg "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
              << std::endl;

  }
}



// N = 128
__global__ void
wgmma_m64n128k16_fmix_SS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
				      float *gm_c, uint8_t sm_layout,               
				      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 128;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x128x16_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
										    
  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];                                                               
  }                                                                                  
										     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
										     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
										     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
										     
  __syncthreads();                                                                   
										     
  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
										     
  GmmaDescriptor desc_a, desc_b;                                                     
  desc_a.layout_type_ = sm_layout;                                                   
  desc_b.layout_type_ = sm_layout;                                                   
										     
  desc_a.start_address_ = sm_a_addr >> 4;                                            
  desc_b.start_address_ = sm_b_addr >> 4;                                            
										     
  desc_a.base_offset_ = 0;                                                           
  desc_b.base_offset_ = 0;                                                           
										     
  desc_a.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
										     
  desc_a.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
										     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(desc_a.desc_, desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3],reg_d[4],reg_d[5],reg_d[6],reg_d[7],
reg_d[8],reg_d[9],reg_d[10],reg_d[11],reg_d[12],reg_d[13],reg_d[14],reg_d[15],
reg_d[16],reg_d[17],reg_d[18],reg_d[19],reg_d[20],reg_d[21],reg_d[22],reg_d[23],
reg_d[24],reg_d[25],reg_d[26],reg_d[27],reg_d[28],reg_d[29],reg_d[30],reg_d[31],
reg_d[32],reg_d[33],reg_d[34],reg_d[35],reg_d[36],reg_d[37],reg_d[38],reg_d[39],
reg_d[40],reg_d[41],reg_d[42],reg_d[43],reg_d[44],reg_d[45],reg_d[46],reg_d[47],
reg_d[48],reg_d[49],reg_d[50],reg_d[51],reg_d[52],reg_d[53],reg_d[54],reg_d[55],
reg_d[56],reg_d[57],reg_d[58],reg_d[59],reg_d[60],reg_d[61],reg_d[62],reg_d[63]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n128k16_fmix_SS(int argc, char **argv) {
  constexpr int M = 64;
  constexpr int N = 128;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n128k16_fmix_SS_test,
		       cudaFuncAttributeMaxDynamicSharedMemorySize,
		       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
	    << " dynamic shared size: " << dyn_shared_size << "bytes"
	    << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n128k16_fmix_SS_test<<<sm_count, 128, dyn_shared_size>>>(
	mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Shared "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
	      << std::endl;
  }
}

__global__ void
wgmma_m64n128k16_fmix_RS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
				      float *gm_c, uint8_t sm_layout,               
				      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 128;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x128x16_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
  GMMA_t::ARegisters reg_a;

  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];
  }                                                                                  
										     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
										     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
										     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
										     
  __syncthreads();                                                                   

  for (int i = 0; i < sizeof(reg_a)/sizeof(__half2); i += 1) {
    reinterpret_cast<__half2&>(reg_a[i]) = sm_a[i];
  }


  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
										     
  GmmaDescriptor desc_b;                                                     
  desc_b.layout_type_ = sm_layout;                                                   
										     
  desc_b.start_address_ = sm_b_addr >> 4;                                            
										     
  desc_b.base_offset_ = 0;                                                           
										     
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
										     
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
										     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(reg_a[0], reg_a[1], reg_a[2], reg_a[3], desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3],reg_d[4],reg_d[5],reg_d[6],reg_d[7],
reg_d[8],reg_d[9],reg_d[10],reg_d[11],reg_d[12],reg_d[13],reg_d[14],reg_d[15],
reg_d[16],reg_d[17],reg_d[18],reg_d[19],reg_d[20],reg_d[21],reg_d[22],reg_d[23],
reg_d[24],reg_d[25],reg_d[26],reg_d[27],reg_d[28],reg_d[29],reg_d[30],reg_d[31],
reg_d[32],reg_d[33],reg_d[34],reg_d[35],reg_d[36],reg_d[37],reg_d[38],reg_d[39],
reg_d[40],reg_d[41],reg_d[42],reg_d[43],reg_d[44],reg_d[45],reg_d[46],reg_d[47],
reg_d[48],reg_d[49],reg_d[50],reg_d[51],reg_d[52],reg_d[53],reg_d[54],reg_d[55],
reg_d[56],reg_d[57],reg_d[58],reg_d[59],reg_d[60],reg_d[61],reg_d[62],reg_d[63]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n128k16_fmix_RS(int argc, char **argv) {
  constexpr int M = 64;
  constexpr int N = 128;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n128k16_fmix_RS_test,
		       cudaFuncAttributeMaxDynamicSharedMemorySize,
		       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
	    << " dynamic shared size: " << dyn_shared_size << "bytes"
	    << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n128k16_fmix_RS_test<<<sm_count, 128, dyn_shared_size>>>(
	mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Reg "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
	      << std::endl;

  }
}

// N = 64
__global__ void
wgmma_m64n64k16_fmix_SS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
				      float *gm_c, uint8_t sm_layout,               
				      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 64;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x64x16_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
										    
  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];                                                               
  }                                                                                  
										     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
										     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
										     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
										     
  __syncthreads();                                                                   
										     
  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
										     
  GmmaDescriptor desc_a, desc_b;                                                     
  desc_a.layout_type_ = sm_layout;                                                   
  desc_b.layout_type_ = sm_layout;                                                   
										     
  desc_a.start_address_ = sm_a_addr >> 4;                                            
  desc_b.start_address_ = sm_b_addr >> 4;                                            
										     
  desc_a.base_offset_ = 0;                                                           
  desc_b.base_offset_ = 0;                                                           
										     
  desc_a.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
										     
  desc_a.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
										     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(desc_a.desc_, desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3],reg_d[4],reg_d[5],reg_d[6],reg_d[7],
reg_d[8],reg_d[9],reg_d[10],reg_d[11],reg_d[12],reg_d[13],reg_d[14],reg_d[15],
reg_d[16],reg_d[17],reg_d[18],reg_d[19],reg_d[20],reg_d[21],reg_d[22],reg_d[23],
reg_d[24],reg_d[25],reg_d[26],reg_d[27],reg_d[28],reg_d[29],reg_d[30],reg_d[31]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n64k16_fmix_SS(int argc, char **argv) {
  constexpr int M = 64;
  constexpr int N = 64;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n64k16_fmix_SS_test,
		       cudaFuncAttributeMaxDynamicSharedMemorySize,
		       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
	    << " dynamic shared size: " << dyn_shared_size << "bytes"
	    << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n64k16_fmix_SS_test<<<sm_count, 128, dyn_shared_size>>>(
	mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Shared "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
	      << std::endl;
  }
}

__global__ void
wgmma_m64n64k16_fmix_RS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
				      float *gm_c, uint8_t sm_layout,               
				      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 64;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x64x16_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
  GMMA_t::ARegisters reg_a;

  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];
  }                                                                                  
										     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
										     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
										     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
										     
  __syncthreads();                                                                   

  for (int i = 0; i < sizeof(reg_a)/sizeof(__half2); i += 1) {
    reinterpret_cast<__half2&>(reg_a[i]) = sm_a[i];
  }

  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
										     
  GmmaDescriptor desc_b;                                                     
  desc_b.layout_type_ = sm_layout;                                                   
										     
  desc_b.start_address_ = sm_b_addr >> 4;                                            
										     
  desc_b.base_offset_ = 0;                                                           
										     
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
										     
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
										     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(reg_a[0], reg_a[1], reg_a[2], reg_a[3], desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3],reg_d[4],reg_d[5],reg_d[6],reg_d[7],
reg_d[8],reg_d[9],reg_d[10],reg_d[11],reg_d[12],reg_d[13],reg_d[14],reg_d[15],
reg_d[16],reg_d[17],reg_d[18],reg_d[19],reg_d[20],reg_d[21],reg_d[22],reg_d[23],
reg_d[24],reg_d[25],reg_d[26],reg_d[27],reg_d[28],reg_d[29],reg_d[30],reg_d[31]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n64k16_fmix_RS(int argc, char **argv) {
  constexpr int M = 64;
  constexpr int N = 64;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n64k16_fmix_RS_test,
		       cudaFuncAttributeMaxDynamicSharedMemorySize,
		       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
	    << " dynamic shared size: " << dyn_shared_size << "bytes"
	    << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n64k16_fmix_RS_test<<<sm_count, 128, dyn_shared_size>>>(
	mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Reg "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
	      << std::endl;

  }
}

// N = 32
__global__ void
wgmma_m64n32k16_fmix_SS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
				      float *gm_c, uint8_t sm_layout,               
				      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 32;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x32x16_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
										    
  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];                                                               
  }                                                                                  
										     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
										     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
										     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
										     
  __syncthreads();                                                                   
										     
  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
										     
  GmmaDescriptor desc_a, desc_b;                                                     
  desc_a.layout_type_ = sm_layout;                                                   
  desc_b.layout_type_ = sm_layout;                                                   
										     
  desc_a.start_address_ = sm_a_addr >> 4;                                            
  desc_b.start_address_ = sm_b_addr >> 4;                                            
										     
  desc_a.base_offset_ = 0;                                                           
  desc_b.base_offset_ = 0;                                                           
										     
  desc_a.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
										     
  desc_a.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
										     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(desc_a.desc_, desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3],reg_d[4],reg_d[5],reg_d[6],reg_d[7],
reg_d[8],reg_d[9],reg_d[10],reg_d[11],reg_d[12],reg_d[13],reg_d[14],reg_d[15]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n32k16_fmix_SS(int argc, char **argv) {
  constexpr int M = 64;
  constexpr int N = 32;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n32k16_fmix_SS_test,
		       cudaFuncAttributeMaxDynamicSharedMemorySize,
		       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
	    << " dynamic shared size: " << dyn_shared_size << "bytes"
	    << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n32k16_fmix_SS_test<<<sm_count, 128, dyn_shared_size>>>(
	mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Shared "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
	      << std::endl;
  }
}

__global__ void
wgmma_m64n32k16_fmix_RS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
				      float *gm_c, uint8_t sm_layout,               
				      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 32;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x32x16_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
  GMMA_t::ARegisters reg_a;

  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];
  }                                                                                  
										     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
										     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
										     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
										     
  __syncthreads();                                                                   

  for (int i = 0; i < sizeof(reg_a)/sizeof(__half2); i += 1) {
    reinterpret_cast<__half2&>(reg_a[i]) = sm_a[i];
  }

  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
										     
  GmmaDescriptor desc_b;                                                     
  desc_b.layout_type_ = sm_layout;                                                   
										     
  desc_b.start_address_ = sm_b_addr >> 4;                                            
										     
  desc_b.base_offset_ = 0;                                                           
										     
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
										     
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
										     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(reg_a[0], reg_a[1], reg_a[2], reg_a[3], desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3],reg_d[4],reg_d[5],reg_d[6],reg_d[7],
reg_d[8],reg_d[9],reg_d[10],reg_d[11],reg_d[12],reg_d[13],reg_d[14],reg_d[15]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n32k16_fmix_RS(int argc, char **argv) {
  constexpr int M = 64;
  constexpr int N = 32;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n32k16_fmix_RS_test,
		       cudaFuncAttributeMaxDynamicSharedMemorySize,
		       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
	    << " dynamic shared size: " << dyn_shared_size << "bytes"
	    << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n32k16_fmix_RS_test<<<sm_count, 128, dyn_shared_size>>>(
	mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Reg "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
	      << std::endl;

  }
}


// N = 16
__global__ void
wgmma_m64n16k16_fmix_SS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
				      float *gm_c, uint8_t sm_layout,               
				      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 16;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x16x16_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
										    
  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];                                                               
  }                                                                                  
										     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
										     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
										     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
										     
  __syncthreads();                                                                   
										     
  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
										     
  GmmaDescriptor desc_a, desc_b;                                                     
  desc_a.layout_type_ = sm_layout;                                                   
  desc_b.layout_type_ = sm_layout;                                                   
										     
  desc_a.start_address_ = sm_a_addr >> 4;                                            
  desc_b.start_address_ = sm_b_addr >> 4;                                            
										     
  desc_a.base_offset_ = 0;                                                           
  desc_b.base_offset_ = 0;                                                           
										     
  desc_a.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
										     
  desc_a.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
										     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(desc_a.desc_, desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3],reg_d[4],reg_d[5],reg_d[6],reg_d[7]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n16k16_fmix_SS(int argc, char **argv) {
  constexpr int M = 64;
  constexpr int N = 16;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n16k16_fmix_SS_test,
		       cudaFuncAttributeMaxDynamicSharedMemorySize,
		       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
	    << " dynamic shared size: " << dyn_shared_size << "bytes"
	    << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n16k16_fmix_SS_test<<<sm_count, 128, dyn_shared_size>>>(
	mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Shared "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
	      << std::endl;
  }
}

__global__ void
wgmma_m64n16k16_fmix_RS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
				      float *gm_c, uint8_t sm_layout,               
				      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 16;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x16x16_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
  GMMA_t::ARegisters reg_a;

  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];
    reinterpret_cast<__half2&>(reg_a[i]) = gm_a[i];
  }                                                                                  
										     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
										     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
										     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
										     
  __syncthreads();                                                                   

  for (int i = 0; i < sizeof(reg_a)/sizeof(__half2); i += 1) {
    reinterpret_cast<__half2&>(reg_a[i]) = sm_a[i];
  }

  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
										     
  GmmaDescriptor desc_b;                                                     
  desc_b.layout_type_ = sm_layout;                                                   
										     
  desc_b.start_address_ = sm_b_addr >> 4;                                            
										     
  desc_b.base_offset_ = 0;                                                           
										     
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
										     
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
										     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(reg_a[0], reg_a[1], reg_a[2], reg_a[3], desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3],reg_d[4],reg_d[5],reg_d[6],reg_d[7]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n16k16_fmix_RS(int argc, char **argv) {
  constexpr int M = 64;
  constexpr int N = 16;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n16k16_fmix_RS_test,
		       cudaFuncAttributeMaxDynamicSharedMemorySize,
		       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
	    << " dynamic shared size: " << dyn_shared_size << "bytes"
	    << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n16k16_fmix_RS_test<<<sm_count, 128, dyn_shared_size>>>(
	mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Reg "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
	      << std::endl;

  }
}



// N = 8
__global__ void
wgmma_m64n8k16_fmix_SS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
                                      float *gm_c, uint8_t sm_layout,               
                                      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 8;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x8x16_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
                                                                                    
  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];                                                               
  }                                                                                  
                                                                                     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
                                                                                     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
                                                                                     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
                                                                                     
  __syncthreads();                                                                   
                                                                                     
  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
                                                                                     
  GmmaDescriptor desc_a, desc_b;                                                     
  desc_a.layout_type_ = sm_layout;                                                   
  desc_b.layout_type_ = sm_layout;                                                   
                                                                                     
  desc_a.start_address_ = sm_a_addr >> 4;                                            
  desc_b.start_address_ = sm_b_addr >> 4;                                            
                                                                                     
  desc_a.base_offset_ = 0;                                                           
  desc_b.base_offset_ = 0;                                                           
                                                                                     
  desc_a.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
                                                                                     
  desc_a.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
                                                                                     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(desc_a.desc_, desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n8k16_fmix_SS(int argc, char **argv) {
  constexpr int M = 64;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n8k16_fmix_SS_test,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
            << " dynamic shared size: " << dyn_shared_size << "bytes"
            << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n8k16_fmix_SS_test<<<sm_count, 128, dyn_shared_size>>>(
        mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Shared "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
              << std::endl;

  }
}

__global__ void
wgmma_m64n8k16_fmix_RS_test(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
                                      float *gm_c, uint8_t sm_layout,               
                                      uint64_t repeat_time) {                        
  constexpr int M = 64;
  constexpr int N = 8;
  constexpr int K = 16;
  extern __shared__ char sm_buff[];                                                  
  using GMMA_t = SM90_64x8x16_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;    
  GMMA_t::CRegisters reg_d;                                                          
  GMMA_t::ARegisters reg_a;

  __half2 *sm_a = (__half2 *)sm_buff;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    sm_a[i] = gm_a[i];
  }                                                                                  
                                                                                     
  __half2 *sm_b = (__half2 *)(sm_buff + sizeof(__half2) * M * K / 2);                
                                                                                     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    sm_b[i] = gm_b[i];                                                               
  }                                                                                  
                                                                                     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
                                                                                     
  __syncthreads();                                                                   

  for (int i = 0; i < sizeof(reg_a)/sizeof(__half2); i += 1) {
    reinterpret_cast<__half2&>(reg_a[i]) = sm_a[i];
  }

  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sm_b));        
                                                                                     
  GmmaDescriptor desc_b;                                                     
  desc_b.layout_type_ = sm_layout;                                                   
                                                                                     
  desc_b.start_address_ = sm_b_addr >> 4;                                            
                                                                                     
  desc_b.base_offset_ = 0;                                                           
                                                                                     
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
                                                                                     
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__half)) >> 4;                    
                                                                                     
  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {                  
    GMMA_t::fma(reg_a[0], reg_a[1], reg_a[2], reg_a[3], desc_b.desc_,                                   
		    reg_d[0],reg_d[1],reg_d[2],reg_d[3]);
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

void test_m64n8k16_fmix_RS(int argc, char **argv) {
  constexpr int M = 64;
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

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64n8k16_fmix_RS_test,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
            << " dynamic shared size: " << dyn_shared_size << "bytes"
            << std::endl;

  int repeat_time = 102400;
  

  {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n8k16_fmix_RS_test<<<sm_count, 128, dyn_shared_size>>>(
        mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    double FLOPS = M * K * N * 2;
    FLOPS *= repeat_time;
    FLOPS *= sm_count;

    double FLOPS_T = FLOPS / duration / 1000 / 1000 / 1000;

    std::cout << "A Reg "
	      << "M=" << M << ","
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

  test_m64n8k16_fmix_SS(argc, argv);
  test_m64n8k16_fmix_RS(argc, argv);

  test_m64n16k16_fmix_SS(argc, argv);
  test_m64n16k16_fmix_RS(argc, argv);

  test_m64n32k16_fmix_SS(argc, argv);
  test_m64n32k16_fmix_RS(argc, argv);

  test_m64n64k16_fmix_SS(argc, argv);
  test_m64n64k16_fmix_RS(argc, argv);

  test_m64n128k16_fmix_SS(argc, argv);
  test_m64n128k16_fmix_RS(argc, argv);

  test_m64n256k16_fmix_SS(argc, argv);
  test_m64n256k16_fmix_RS(argc, argv);

}

