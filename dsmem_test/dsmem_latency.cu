#include <fstream>
#include <iostream>

#include "hopper_util/util.h"

#include <cooperative_groups.h>

using namespace cooperative_groups;

#define CLUSTER_DIM 16
#define TEST_COUNT 1024


__global__ void test_dsmem_latency(int* pointer_chase, int* dummy_output, int* latency_output) {
  dim3 cta_id_in_cluster = block_id_in_cluster();

  extern __shared__ char sm_base[];
  int* chase_sm = (int*)sm_base;
  int* temp_idx = chase_sm + TEST_COUNT;
  int* latency_sm = temp_idx + TEST_COUNT;

  int dummy_out = threadIdx.x;
  for (int i = 0; i < TEST_COUNT; i+=blockDim.x) {
    chase_sm[i] = pointer_chase[i];
  }

  __syncthreads();

  cluster_sync();

  if (cta_id_in_cluster.x == 0) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(chase_sm);
    // test latency of each sm
    for (int cid = 0; cid < CLUSTER_DIM; ++cid) {
      int pos = 0;
      int* curr_ptr = chase_sm;
      for (int i = 0; i < TEST_COUNT; ++i) {
        int start = clock();
  
        uint32_t smem_addr = cast_smem_ptr_to_uint(curr_ptr);

        asm volatile("{\t\n"
                 ".reg .b32 remAddr32;\n\t"
		 "mapa.shared::cluster.u32  remAddr32, %1, %2;\n\t"
                 "ld.shared::cluster.u32 %0, [remAddr32];\n\t"
                 "}"
                 : "=r"(pos)
                 : "r"(smem_addr), "r"(cid)
                 : "memory");
	curr_ptr += pos;
	temp_idx[i] = pos;
        int end = clock();
	latency_sm[i] = end - start;
      }
      __syncthreads();

      for (int i = 0; i < TEST_COUNT; ++i) {
        dummy_out += temp_idx[i];
	latency_output[cid * TEST_COUNT + i] = latency_sm[i];
      }
      __syncthreads();
    }
  }

  cluster_sync();

  dummy_output[cta_id_in_cluster.x] = dummy_out;
}

int main(int argc, char** argv) {
  int64_t pointer_chase_size = TEST_COUNT*sizeof(int);
  int* pointer_chase_device = nullptr;
  cudaMalloc(&pointer_chase_device, TEST_COUNT*sizeof(int));
  int* pointer_chase_host = new int[TEST_COUNT];
  for (int i = 0;i < TEST_COUNT; ++i) {
    pointer_chase_host[i] = 1;
  }

  gpuErrchk(cudaMemcpy(pointer_chase_device, pointer_chase_host,
                       TEST_COUNT * sizeof(int),
                       cudaMemcpyHostToDevice));

  int* dummy_output_device = nullptr;
  cudaMalloc(&dummy_output_device, TEST_COUNT*sizeof(int));

  int* latency_output_dev = nullptr;
  cudaMalloc(&latency_output_dev, CLUSTER_DIM * TEST_COUNT * sizeof(int));


  int grid_dim = CLUSTER_DIM;
  dim3 block_dim(1);

  size_t sm_size = TEST_COUNT * 3 * sizeof(int);


  std::cout << "sm_size: " << sm_size << std::endl;
  cudaFuncSetAttribute(test_dsmem_latency,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sm_size);
  cudaFuncSetAttribute(
          test_dsmem_latency, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

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

        cudaLaunchKernelEx(&config, test_dsmem_latency, pointer_chase_device, dummy_output_device, latency_output_dev);
    }
  gpuErrchk(cudaDeviceSynchronize());

  int* latency_result = new int[CLUSTER_DIM * TEST_COUNT];

  gpuErrchk(cudaMemcpy(latency_result, latency_output_dev,
                       CLUSTER_DIM * TEST_COUNT * sizeof(int),
                       cudaMemcpyDeviceToHost));

  std::ofstream ofs("dsmem_latency_16.csv");

  ofs << "offset(byte)";

  for (int cid = 0; cid < CLUSTER_DIM; ++cid) {
    ofs << ",sm_" << cid;
  }
  ofs <<"\n";

  for (int i = 0; i < TEST_COUNT; ++i) {
    ofs << i * sizeof(int);
    for (int cid = 0; cid < CLUSTER_DIM; ++cid) {
      ofs << "," << latency_result[cid * TEST_COUNT + i];
    }
    ofs << "\n";
  }
  
  for (int cid = 0; cid < CLUSTER_DIM; ++cid) {
    double sum = 0;
    for (int i = 0; i < TEST_COUNT; ++i) {
      sum += latency_result[cid * TEST_COUNT + i];
    }
    std::cout << "sm_0 to sm_" << cid << " avg rd latency: " << sum / TEST_COUNT << std::endl;
  }
}

