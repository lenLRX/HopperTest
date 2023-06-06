#include <iostream>

#include "hopper_util/util.h"


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


#define ACCESS_SIZE 256
#define WARP_NUM 16

__global__  void test_tma_1(float* input, float* output, CUtensorMap* input_desc, int64_t copy_count) {
  dim3 cluster_id = cluster_id_in_grid();
  dim3 cta_id = block_id_in_cluster();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    printf("blockidx %d, cluster idx %d, cta_id in cluster %d\n", blockIdx.x, cluster_id.x, cta_id.x);
  }

  extern __shared__ char sm_base[];
  //__shared__ float sm_buffer[WARP_NUM][ACCESS_SIZE];
  float* sm_buffer = (float*)sm_base;
  uint64_t* sm_mbarrier = (uint64_t*)(sm_base + sizeof(float) * WARP_NUM * ACCESS_SIZE);

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
		    &sm_buffer[threadIdx.y * ACCESS_SIZE],
		    i);
    wait_barrier(sm_mbarrier[threadIdx.y], phase);
    ++phase;
    dummy_sum += sm_buffer[threadIdx.y*ACCESS_SIZE];
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

  int grid_dim = 112;
  dim3 block_dim(32, 16);

  size_t sm_size = sizeof(float) * WARP_NUM * ACCESS_SIZE + sizeof(uint64_t) * WARP_NUM;

  cudaFuncSetAttribute(test_tma_1,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sm_size);

  cudaFuncSetAttribute(
          test_tma_1, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  // Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim = grid_dim;
        config.blockDim = block_dim;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 16; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
	config.dynamicSmemBytes = sm_size;

        cudaLaunchKernelEx(&config, test_tma_1, data_input_device, data_output_device, input_tma_desc_device, buffer_size);
    }
  //test_tma_1<<<grid_dim, block_dim, sm_size>>>(data_input_device, data_output_device, input_tma_desc_device, buffer_size);
  gpuErrchk(cudaDeviceSynchronize());


}
