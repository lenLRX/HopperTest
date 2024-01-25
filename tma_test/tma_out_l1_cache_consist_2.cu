#include <iostream>

#include "hopper_util/util.h"

/*
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

*/
#define ACCESS_SIZE 256
#define WARP_NUM 1

__global__ void test_tma_1(float* input, float* output, CUtensorMap* input_desc, int64_t copy_count) {

  extern __shared__ char sm_base[];
  //__shared__ float sm_buffer[WARP_NUM][ACCESS_SIZE];
  float* sm_buffer = (float*)sm_base;
  uint64_t* sm_mbarrier = (uint64_t*)(sm_base + sizeof(float) * WARP_NUM * ACCESS_SIZE);

  // one thread per warp
  if (threadIdx.x % 32 != 0) {
    return;
  }

  float dummy_sum = input[0];

  sm_buffer[0] = 0.12345;



  __syncthreads();



    SM90_TMA_STORE_1D::copy(input_desc,
		    &sm_buffer[0], 0);
    asm volatile("cp.async.bulk.commit_group;");
    asm volatile(
      "cp.async.bulk.wait_group %0;"
      :
      : "n"(1)
      : "memory");

    
    __nanosleep(1000000000);

    float result = input[0];

    // prints 0.12345
    printf("read result of TMA %f\n", result);


  output[threadIdx.y] = dummy_sum;
}



int main(int argc, char** argv) {
  int64_t buffer_size = 1024*1024*1024;
  float* data_input_device = nullptr;
  cudaMalloc(&data_input_device, buffer_size*sizeof(float));

  float* data_output_device = nullptr;
  cudaMalloc(&data_output_device, buffer_size*sizeof(float));

  CUtensorMap input_tma_desc;

  CUtensorMap* input_tma_desc_device;
  cudaMalloc(&input_tma_desc_device, sizeof(CUtensorMap));

  uint64_t input_globalDim[5] = {1,1,1,1,1};
  uint64_t input_globalStride[5] = {0,0,0,0,0};

  input_globalDim[0] = buffer_size;

  uint32_t smem_box_shape[5] = {1,1,1,1,1};
  uint32_t smem_box_stride[5] = {1,1,1,1,1};


  smem_box_shape[0] = ACCESS_SIZE;

  CUresult encode_result =
  cuTensorMapEncodeTiled(&input_tma_desc, 
		         CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
			 1,
			 data_input_device,
			 input_globalDim,
			 input_globalStride + 1,
			 smem_box_shape,
			 smem_box_stride,
			 CU_TENSOR_MAP_INTERLEAVE_NONE,
			 CU_TENSOR_MAP_SWIZZLE_NONE,
			 CU_TENSOR_MAP_L2_PROMOTION_NONE,
			 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (encode_result != CUDA_SUCCESS) {
	  std::cerr << "failed to init TMA desc\n";
	  return -1;
  }

  cudaMemcpy(input_tma_desc_device, &input_tma_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  int grid_dim = 1;
  dim3 block_dim(32);

  size_t sm_size = sizeof(float) * WARP_NUM * ACCESS_SIZE + sizeof(uint64_t) * WARP_NUM;

  cudaFuncSetAttribute(test_tma_1,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sm_size);

  test_tma_1<<<grid_dim, block_dim, sm_size>>>(data_input_device, data_output_device, input_tma_desc_device, buffer_size);
  gpuErrchk(cudaDeviceSynchronize());


}
