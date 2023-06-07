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


  CUTLASS_DEVICE
  static void arrive_and_reset_bytes_remote(uint64_t& smem_bar,uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&smem_bar);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes));
  }



template<int STAGE_NUM, int ACCESS_SIZE>
__global__ void test_tma_1(float* input, float* output, CUtensorMap* input_desc, int64_t copy_count) {
  dim3 cluster_id = cluster_id_in_grid();
  dim3 cta_id_in_cluster = block_id_in_cluster();

  int cluster_dim = cluster_shape().x;

  uint16_t mcast_mask = (1 << cluster_dim) - 1;
  extern __shared__ char sm_base[];
  float* sm_buffer = (float*)sm_base;
  uint64_t* sm_mbarrier_full = (uint64_t*)(sm_base + sizeof(float) * STAGE_NUM * cluster_dim * ACCESS_SIZE);
  uint64_t* sm_mbarrier_empty = sm_mbarrier_full + STAGE_NUM * cluster_dim;

  // number of producer thread: STAGE_NUM 
  // number of consumer thread: STAGE_NUM * cluster_dim
  int total_thread_num = STAGE_NUM * (cluster_dim + 1);

  int stage_id = threadIdx.x % STAGE_NUM;

  bool is_producer = threadIdx.x < STAGE_NUM;

  float dummy_sum = 0.0;

  // only one empty bar
  // each buffer has one barrier object
  if (is_producer) {
    initialize_barrier(sm_mbarrier_empty[stage_id], cluster_dim);
  }
  else {
    initialize_barrier(sm_mbarrier_full[threadIdx.x - STAGE_NUM], 1);
  }

    asm volatile(
      "{\n\t"
      "fence.mbarrier_init.release.cluster; \n"
      "}"
      ::);



  cluster_sync();

  int empty_phase = 0;
  int full_phase = 0;

  for (int64_t i = (cta_id_in_cluster.x * STAGE_NUM + stage_id) * ACCESS_SIZE; i < copy_count; i += ACCESS_SIZE * cluster_dim * STAGE_NUM) {
    if (is_producer) {
      // producer threads
      wait_barrier(sm_mbarrier_empty[stage_id], empty_phase);

      int buffer_idx = cta_id_in_cluster.x * STAGE_NUM + stage_id;
      /*
      for (int cluster_i = 0; cluster_i < cluster_dim; ++cluster_i) {
	  arrive_and_reset_bytes_remote(sm_mbarrier_full[buffer_idx], ACCESS_SIZE * sizeof(float), cluster_i, 1);
      }
      */

      SM90_TMA_LOAD_MULTICAST_1D::copy(input_desc, sm_mbarrier_full[buffer_idx],
                 mcast_mask, &sm_buffer[(buffer_idx) * ACCESS_SIZE],  i);
    }
    else {
      // consumer threads
      int remote_cluster_id = (threadIdx.x - STAGE_NUM) / STAGE_NUM;

      set_barrier_transaction_bytes(sm_mbarrier_full[threadIdx.x - STAGE_NUM], ACCESS_SIZE * sizeof(float));
      arrive_remote(sm_mbarrier_empty[stage_id], remote_cluster_id, 1);

      wait_barrier(sm_mbarrier_full[threadIdx.x - STAGE_NUM], full_phase);
    }
    
    ++empty_phase;
    ++full_phase;

    dummy_sum += sm_buffer[threadIdx.x];
  }
  // dummy output
  output[threadIdx.x] = dummy_sum;
}


template<int STAGE_NUM, int CLUSTER_DIM, int ACCESS_SIZE>
void tma_test_main() {

  int grid_dim = 112;
  int cluster_num = grid_dim / CLUSTER_DIM;


  int64_t buffer_size = cluster_num * STAGE_NUM * ACCESS_SIZE * 512;
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
	  return;
  }

  cudaMemcpy(input_tma_desc_device, &input_tma_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);


  dim3 block_dim(STAGE_NUM*(CLUSTER_DIM+1));

  size_t sm_size = sizeof(float) * STAGE_NUM * CLUSTER_DIM * ACCESS_SIZE + sizeof(uint64_t) * STAGE_NUM * 2 * CLUSTER_DIM + 4096;


  std::cout << "tma mcast test: STAGE_NUM=" << STAGE_NUM << " CLUSTER_DIM=" << CLUSTER_DIM
	 << " buffer size: " << buffer_size << " sm_size: " << sm_size << std::endl;
  cudaFuncSetAttribute(test_tma_1<STAGE_NUM, ACCESS_SIZE>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sm_size);
  cudaFuncSetAttribute(
          test_tma_1<STAGE_NUM, ACCESS_SIZE>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

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

        cudaLaunchKernelEx(&config, test_tma_1<STAGE_NUM, ACCESS_SIZE>, data_input_device, data_output_device, input_tma_desc_device, buffer_size);
    }
  //test_tma_1<<<grid_dim, block_dim, sm_size>>>(data_input_device, data_output_device, input_tma_desc_device, buffer_size);
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(data_input_device);
  cudaFree(data_output_device);
  cudaFree(input_tma_desc_device);
}

int main(int argc, char** argv) {
  tma_test_main<64, 2, 256>();
  tma_test_main<32, 4, 256>();
  tma_test_main<128, 2, 128>();
  tma_test_main<64, 4, 128>();

}

