#include <iostream>

#include "hopper_util/util.h"

#include <cooperative_groups.h>

using namespace cooperative_groups;

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

  CUTLASS_DEVICE
  static void arrive_remote(uint64_t& smem_bar, uint32_t cta_id, uint32_t pred) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&smem_bar);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "@p mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred));
  }


#define ACCESS_SIZE 256
#define WARP_NUM 16
#define CLUSTER_DIM 4

__global__ void test_tma_1(float* input, float* output, CUtensorMap* input_desc, int64_t copy_count) {
  dim3 cluster_id = cluster_id_in_grid();
  dim3 cta_id_in_cluster = block_id_in_cluster();

  uint16_t mcast_mask = 0xf;
  extern __shared__ char sm_base[];
  float* sm_buffer = (float*)sm_base;
  uint64_t* sm_mbarrier_full = (uint64_t*)(sm_base + sizeof(float) * WARP_NUM * CLUSTER_DIM * ACCESS_SIZE);
  uint64_t* sm_mbarrier_empty = sm_mbarrier_full + WARP_NUM * CLUSTER_DIM;

  // one thread per warp
  if (threadIdx.x % 32 != 0) {
    return;
  }

  float dummy_sum = 0.0;

  // only one empty bar
  // each buffer has one barrier object
  initialize_barrier(sm_mbarrier_empty[threadIdx.y], CLUSTER_DIM);
  for (int i = 0;i < CLUSTER_DIM; ++i) {
    initialize_barrier(sm_mbarrier_full[threadIdx.y * CLUSTER_DIM + i], 1);
  }

    asm volatile(
      "{\n\t"
      "fence.mbarrier_init.release.cluster; \n"
      "}"
      ::);

  cluster_sync();

  int empty_phase = 0;
  int full_phase = 0;

  for (int64_t i = (blockIdx.x *blockDim.y + threadIdx.y) * ACCESS_SIZE; i < copy_count; i += ACCESS_SIZE * gridDim.x * blockDim.y) {
    
    for (int cluster_i = 0; cluster_i < CLUSTER_DIM; ++cluster_i){
      int idx = threadIdx.y * CLUSTER_DIM + cluster_i;
      set_barrier_transaction_bytes(sm_mbarrier_full[idx], ACCESS_SIZE * sizeof(float));
    }


    SM90_TMA_LOAD_MULTICAST_1D::copy(input_desc, sm_mbarrier_full[threadIdx.y * CLUSTER_DIM + cta_id_in_cluster.x],
		                     mcast_mask, &sm_buffer[(threadIdx.y *CLUSTER_DIM + cta_id_in_cluster.x) * ACCESS_SIZE],  i);

    for (int cluster_i = 0; cluster_i < CLUSTER_DIM; ++cluster_i){
      wait_barrier(sm_mbarrier_full[threadIdx.y * CLUSTER_DIM + cluster_i], full_phase);
    }

    for (int cluster_i = 0; cluster_i < CLUSTER_DIM; ++cluster_i){
      arrive_remote(sm_mbarrier_empty[threadIdx.y], cluster_i, 1);
    }

    wait_barrier(sm_mbarrier_empty[threadIdx.y], empty_phase);


    ++empty_phase;
    ++full_phase;
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
  dim3 block_dim(32, WARP_NUM);

  size_t sm_size = sizeof(float) * WARP_NUM * CLUSTER_DIM * ACCESS_SIZE + sizeof(uint64_t) * WARP_NUM * 2 * CLUSTER_DIM + 4096;


  std::cout << "sm_size: " << sm_size << std::endl;
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
        attribute[0].val.clusterDim.x = CLUSTER_DIM; // Cluster size in X-dimension
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
