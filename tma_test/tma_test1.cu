#include <iostream>

#include "hopper_util/util.h"


#define ACCESS_SIZE 256
#define WARP_NUM 16

__global__ void test_tma_1(float* input, float* output, CUtensorMap* input_desc, int64_t copy_count) {

  __shared__ float sm_buffer[WARP_NUM][ACCESS_SIZE];
  __shared__ uint64_t sm_mbarrier[WARP_NUM];

  // one thread per warp
  if (threadIdx.x % 32 != 0) {
    return;
  }

  float dummy_sum = 0.0;

  initialize_barrier(sm_mbarrier[threadIdx.y]);

    asm volatile(
      "{\n\t"
      "fence.mbarrier_init.release.cluster; \n"
      "}"
      ::);

  int phase = 0;

  for (int64_t i = (blockIdx.x *blockDim.y + threadIdx.y) * ACCESS_SIZE; i < copy_count; i += ACCESS_SIZE * gridDim.x * blockDim.y) {
    set_barrier_transaction_bytes(sm_mbarrier[threadIdx.y], ACCESS_SIZE * sizeof(float));

    SM90_TMA_LOAD_1D::copy(input_desc, sm_mbarrier[threadIdx.y],
		    sm_buffer[threadIdx.y],
		    i);
    wait_barrier(sm_mbarrier[threadIdx.y], phase);
    ++phase;
    dummy_sum += sm_buffer[threadIdx.y][0];
  }
  // dummy output
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

  int grid_dim = 114;
  dim3 block_dim(32, 16);

  test_tma_1<<<grid_dim, block_dim>>>(data_input_device, data_output_device, input_tma_desc_device, buffer_size);
  cudaDeviceSynchronize();


}
