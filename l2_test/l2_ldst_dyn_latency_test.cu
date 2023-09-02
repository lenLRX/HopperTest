#include <stdio.h>
#include <cstdint>


inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}


inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__global__ void l2_float4_test(float4* data, float4* output_data, int64_t data_size, int64_t repeat, int* latency_output) {
  int64_t init_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t idx = init_idx;

  constexpr int unroll_size = 8;
  constexpr int ub_buffer_sz = 12288;

  __shared__ int latency_data[ub_buffer_sz];

  float4 temp_data[unroll_size];

  for (int ui = 0; ui < unroll_size; ui++) {
    temp_data[ui] = make_float4(0);
  }

  int stride = blockDim.x;
  if (blockIdx.x > 0) {

    for (int64_t i = 0; i < repeat; i += unroll_size) {
      for (int ui = 0; ui < unroll_size; ui++) {
        temp_data[ui] = temp_data[ui] + data[idx];
        idx += stride;
        if (idx >= data_size) {
          idx = init_idx;
        }
      }
    }
  }
  else {
    for (int i = threadIdx.x;i < ub_buffer_sz; i += blockDim.x) {
      latency_data[i] = 0;
    }
    __syncthreads();
    if (threadIdx.x > 0) {
      return;
    }
    __nanosleep(10000);
    int curr_sm_idx = 0;
    for (int64_t i = 0; i < repeat; i += 20) {
      int start = clock();
      temp_data[0] = temp_data[0] + data[idx];
      __syncthreads();
      int end = clock();
      idx += 128 / sizeof(float);
      if (idx >= data_size) {
        //idx = init_idx;
	break;
      }
      latency_data[curr_sm_idx] = end - start;
      curr_sm_idx = (curr_sm_idx + 1) % ub_buffer_sz;
    }
    __syncthreads();
    for (int i = 0;i < ub_buffer_sz; i += 1) {
      latency_output[i] = latency_data[i];
    }

  }

  float4 all_data = make_float4(0);
  for (int ui = 0; ui < unroll_size; ui++) {
    all_data = all_data + temp_data[ui];
  }

  output_data[threadIdx.x + blockIdx.x * blockDim.x] = all_data;
}


void test_l2_bw_float4(int repeat) {
  int32_t l2_size = 20*1024*1024;
  float4* l2_dev = nullptr;
  float4* output_data = nullptr;

  int latency_data_size = 48*1024;
  int latency_data_count = latency_data_size / sizeof(int);

  int* latency_output = nullptr;
  cudaMalloc(&latency_output, latency_data_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;

  int thread_num = 512;

  cudaMalloc(&l2_dev, l2_size);
  cudaMalloc(&output_data, sm_count * thread_num * sizeof(float4));

  int data_size = l2_size / sizeof(float4);

  float device_time;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  l2_float4_test<<<sm_count, thread_num>>>(l2_dev, output_data, data_size, repeat, latency_output);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&device_time, start, stop);

  cudaDeviceSynchronize();
  int* latency_host = new int[latency_data_count];
  cudaMemcpy(latency_host, latency_output, latency_data_count, cudaMemcpyDeviceToHost);
  
  int total_latency = 0;
  int total_count = 0;

  for (int i = 0;i < latency_data_count; ++i) {
    int cycle = latency_host[i];
    if (cycle > 0) {
      //printf("#%6d latency %6d cycle\n", i, cycle);
      total_latency += cycle;
      total_count += 1;
    }
  }

  double avg_latency = double(total_latency) / total_count;

  double bw = double(sm_count * thread_num) * repeat * sizeof(float4) / double(device_time) / 1000 / 1000;

  printf("test_l2_bw_float4 test time %5.2f bandwidth %6.2fG avg latency %6.2f\n", device_time, bw, avg_latency);
}


__global__ void l2_float_stride_test(float* data, float* output_data, int64_t thread_stride, int64_t data_size, int64_t repeat) {
  int64_t init_idx = threadIdx.x * thread_stride + blockIdx.x * blockDim.x * thread_stride;
  int64_t idx = init_idx;

  constexpr int unroll_size = 8;

  float temp_data[unroll_size];

  for (int ui = 0; ui < unroll_size; ui++) {
    temp_data[ui] = 0.0f;
  }


  int stride = blockDim.x * thread_stride;

  for (int64_t i = 0; i < repeat; i += unroll_size) {
    for (int ui = 0; ui < unroll_size; ui++) {
      temp_data[ui] = temp_data[ui] + data[idx];
      idx += stride;
      if (idx >= data_size) {
        idx = init_idx;
      }
    }
  }
  float all_data = 0.0f;
  for (int ui = 0; ui < unroll_size; ui++) {
    all_data = all_data + temp_data[ui];
  }

  output_data[threadIdx.x + blockIdx.x * blockDim.x] = all_data;
}

void test_l2_bw_stride(int repeat, int thread_stride_byte) {
  int32_t l2_size = 20*1024*1024;
  float* l2_dev = nullptr;
  float* output_data = nullptr;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;

  int thread_num = 1024;

  cudaMalloc(&l2_dev, l2_size);
  cudaMalloc(&output_data, sm_count * thread_num * sizeof(float));

  int data_size = l2_size / sizeof(float);

  float device_time;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  l2_float_stride_test<<<sm_count, thread_num>>>(l2_dev, output_data, thread_stride_byte/sizeof(float), data_size, repeat);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&device_time, start, stop);

  double bw = double(sm_count * thread_num) * repeat * 32 / double(device_time) / 1000 / 1000;

  printf("test_l2_bw_stride test time %5.2f stride %4dB bandwidth %6.2fG\n", device_time, thread_stride_byte, bw);
}



int main(int argc, char** argv) {
  test_l2_bw_float4(100000);
  test_l2_bw_stride(100000, 32);
  test_l2_bw_stride(100000, 64);
  test_l2_bw_stride(100000, 128);
}

