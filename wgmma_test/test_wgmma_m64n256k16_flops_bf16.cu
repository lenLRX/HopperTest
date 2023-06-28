#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "gmma_desc.h"

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

static const int M = 64;
static const int N = 256;
static const int K = 16;

// I didn't check result since it is just a throughput test
__global__ void
wgmma_m64n256k16_throughput_fmix_test(float *gm_d, __nv_bfloat162 *gm_a, __nv_bfloat162 *gm_b,
                                      float *gm_c, uint8_t sm_layout,
                                      uint64_t repeat_time) {
  extern __shared__ char sm_buff[];
  float RegD[128];

  __nv_bfloat162 *sm_a = (__nv_bfloat162 *)sm_buff;
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {
    sm_a[i] = gm_a[i];
  }

  __nv_bfloat162 *sm_b = (__nv_bfloat162 *)(sm_buff + sizeof(__nv_bfloat162) * M * K / 2);

  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {
    sm_b[i] = gm_b[i];
  }

  for (int i = 0; i < 128; ++i) {
    RegD[i] = gm_c[threadIdx.x + blockDim.x * i];
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

  desc_a.leading_byte_offset_ = (8 * 8 * sizeof(__nv_bfloat16)) >> 4;
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__nv_bfloat16)) >> 4;

  desc_a.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__nv_bfloat16)) >> 4;
  desc_b.stride_byte_offset_ = (2 * 8 * 8 * sizeof(__nv_bfloat16)) >> 4;

  for (uint64_t repeat_i = 0; repeat_i < repeat_time; ++repeat_i) {

    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " 1,   1,  1,  0,  0;\n"
        "}\n"
        : "+f"(RegD[0]), "+f"(RegD[1]), "+f"(RegD[2]), "+f"(RegD[3]),
          "+f"(RegD[4]), "+f"(RegD[5]), "+f"(RegD[6]), "+f"(RegD[7]),
          "+f"(RegD[8]), "+f"(RegD[9]), "+f"(RegD[10]), "+f"(RegD[11]),
          "+f"(RegD[12]), "+f"(RegD[13]), "+f"(RegD[14]), "+f"(RegD[15]),
          "+f"(RegD[16]), "+f"(RegD[17]), "+f"(RegD[18]), "+f"(RegD[19]),
          "+f"(RegD[20]), "+f"(RegD[21]), "+f"(RegD[22]), "+f"(RegD[23]),
          "+f"(RegD[24]), "+f"(RegD[25]), "+f"(RegD[26]), "+f"(RegD[27]),
          "+f"(RegD[28]), "+f"(RegD[29]), "+f"(RegD[30]), "+f"(RegD[31]),
          "+f"(RegD[32]), "+f"(RegD[33]), "+f"(RegD[34]), "+f"(RegD[35]),
          "+f"(RegD[36]), "+f"(RegD[37]), "+f"(RegD[38]), "+f"(RegD[39]),
          "+f"(RegD[40]), "+f"(RegD[41]), "+f"(RegD[42]), "+f"(RegD[43]),
          "+f"(RegD[44]), "+f"(RegD[45]), "+f"(RegD[46]), "+f"(RegD[47]),
          "+f"(RegD[48]), "+f"(RegD[49]), "+f"(RegD[50]), "+f"(RegD[51]),
          "+f"(RegD[52]), "+f"(RegD[53]), "+f"(RegD[54]), "+f"(RegD[55]),
          "+f"(RegD[56]), "+f"(RegD[57]), "+f"(RegD[58]), "+f"(RegD[59]),
          "+f"(RegD[60]), "+f"(RegD[61]), "+f"(RegD[62]), "+f"(RegD[63]),
          "+f"(RegD[64]), "+f"(RegD[65]), "+f"(RegD[66]), "+f"(RegD[67]),
          "+f"(RegD[68]), "+f"(RegD[69]), "+f"(RegD[70]), "+f"(RegD[71]),
          "+f"(RegD[72]), "+f"(RegD[73]), "+f"(RegD[74]), "+f"(RegD[75]),
          "+f"(RegD[76]), "+f"(RegD[77]), "+f"(RegD[78]), "+f"(RegD[79]),
          "+f"(RegD[80]), "+f"(RegD[81]), "+f"(RegD[82]), "+f"(RegD[83]),
          "+f"(RegD[84]), "+f"(RegD[85]), "+f"(RegD[86]), "+f"(RegD[87]),
          "+f"(RegD[88]), "+f"(RegD[89]), "+f"(RegD[90]), "+f"(RegD[91]),
          "+f"(RegD[92]), "+f"(RegD[93]), "+f"(RegD[94]), "+f"(RegD[95]),
          "+f"(RegD[96]), "+f"(RegD[97]), "+f"(RegD[98]), "+f"(RegD[99]),
          "+f"(RegD[100]), "+f"(RegD[101]), "+f"(RegD[102]), "+f"(RegD[103]),
          "+f"(RegD[104]), "+f"(RegD[105]), "+f"(RegD[106]), "+f"(RegD[107]),
          "+f"(RegD[108]), "+f"(RegD[109]), "+f"(RegD[110]), "+f"(RegD[111]),
          "+f"(RegD[112]), "+f"(RegD[113]), "+f"(RegD[114]), "+f"(RegD[115]),
          "+f"(RegD[116]), "+f"(RegD[117]), "+f"(RegD[118]), "+f"(RegD[119]),
          "+f"(RegD[120]), "+f"(RegD[121]), "+f"(RegD[122]), "+f"(RegD[123]),
          "+f"(RegD[124]), "+f"(RegD[125]), "+f"(RegD[126]), "+f"(RegD[127])
        : "l"(desc_a.desc_), "l"(desc_b.desc_));
  }

  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < 128; ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = RegD[i];
  }
}

template <typename T> void init_data(T *data, int count, std::string method);

template <> void init_data(__nv_bfloat16 *data, int count, std::string method) {
  if (method == "uniform") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < count; ++i) {
      data[i] = __float2bfloat16(dis(gen));
    }
  } else {
    for (int i = 0; i < count; ++i) {
      data[i] = __float2bfloat16(0);
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

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << " usage ./test  <init method>\n";
    return -1;
  }

  std::string init_method(argv[1]);

  int mat_a_size = M * K;
  int mat_b_size = N * K;
  int mat_c_size = M * N;

  __nv_bfloat16 *mat_a_host = new __nv_bfloat16[mat_a_size];
  init_data(mat_a_host, mat_a_size, init_method);

  __nv_bfloat16 *mat_b_host = new __nv_bfloat16[mat_b_size];
  init_data(mat_b_host, mat_b_size, init_method);

  float *mat_c_host = new float[mat_c_size];
  init_data(mat_c_host, mat_c_size, init_method);

  float *mat_d_host = new float[mat_c_size];
  init_data(mat_d_host, mat_c_size, init_method);

  __nv_bfloat162 *mat_a_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_a_dev, mat_a_size * sizeof(__nv_bfloat16)));
  gpuErrchk(cudaMemcpy(mat_a_dev, mat_a_host, mat_a_size * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice));

  __nv_bfloat162 *mat_b_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_b_dev, mat_b_size * sizeof(__nv_bfloat16)));
  gpuErrchk(cudaMemcpy(mat_b_dev, mat_b_host, mat_b_size * sizeof(__nv_bfloat16),
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
      mat_a_size * sizeof(__nv_bfloat16) + mat_b_size * sizeof(__nv_bfloat16);

  cudaFuncSetAttribute(wgmma_m64n256k16_throughput_fmix_test,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;
  std::cout << "using SM count: " << sm_count
            << " dynamic shared size: " << dyn_shared_size << "bytes"
            << std::endl;

  int repeat_time = 102400;
  int warmup_times = 0;
  // warm up
  for (int i = 0; i < warmup_times; ++i) {
    wgmma_m64n256k16_throughput_fmix_test<<<sm_count, 128, dyn_shared_size>>>(
        mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, repeat_time);
  }

  while (1) {
    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    wgmma_m64n256k16_throughput_fmix_test<<<sm_count, 128, dyn_shared_size>>>(
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

    std::cout << "duration: " << duration << "ms " << FLOPS_T << "TFLOP/s"
              << std::endl;
  }
}
