#include <fstream>
#include <iostream>

#include "hopper_util/util.h"

#include <cooperative_groups.h>

using namespace cooperative_groups;

#define CLUSTER_DIM 16

#define REPEAT 128

__global__ void test_dsmem_bandwidth(int* load_data, int* dummy_output, int* latency_output, int test_sm_count) {
  dim3 cta_id_in_cluster = block_id_in_cluster();

  extern __shared__ char sm_base[];
  int* data_sm = (int*)sm_base;

  int dummy_out = threadIdx.x;
  for (int i = threadIdx.x; i < test_sm_count; i += blockDim.x) {
    data_sm[i] = load_data[i];
  }

  __syncthreads();

  cluster_sync();

  int dummy_x[8] = {0};

  if (cta_id_in_cluster.x == 0) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(data_sm);
    // test latency of each sm
    for (int cid = 0; cid < CLUSTER_DIM; ++cid) {
      int pos = 0;
      int start = clock();
      for (int l = 0; l < REPEAT; ++l) {
        int offset = (l * 32 + threadIdx.x) % blockDim.x;
      #pragma unroll
      for (int i = 0; i < 64; ++i) {

        asm volatile("{\t\n"
                 ".reg .b32 remAddr32;\n\t"
		 "mapa.shared::cluster.u32  remAddr32, %1, %2;\n\t"
                 "ld.shared::cluster.u32 %0, [remAddr32];\n\t"
                 "}"
                 : "=r"(pos)
                 : "r"(smem_addr + (i * blockDim.x + offset) * 4), "r"(cid)
                 : "memory");
	dummy_x[i%8] += pos;
      }
      }
      __syncthreads();
      int end = clock();

      if (threadIdx.x == 0) {
        latency_output[cid] = end - start;
      }
      __syncthreads();
    }
  }

  cluster_sync();

  for (int i = 0; i < 8; ++i) {
    dummy_out += dummy_x[i];
  }

  dummy_output[cta_id_in_cluster.x] = dummy_out;
}

int main(int argc, char** argv) {
  int test_sm_byte = 128 * 1024; // 128 kb
  int* load_data_buffer = nullptr;
  cudaMalloc(&load_data_buffer, test_sm_byte);

  int* dummy_output_device = nullptr;
  cudaMalloc(&dummy_output_device, test_sm_byte);

  int* latency_output_dev = nullptr;
  cudaMalloc(&latency_output_dev, CLUSTER_DIM * sizeof(int));


  int grid_dim = CLUSTER_DIM;
  dim3 block_dim(512);

  size_t sm_size = test_sm_byte;

  std::cout << "sm_size: " << sm_size << std::endl;
  cudaFuncSetAttribute(test_dsmem_bandwidth,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sm_size);
  cudaFuncSetAttribute(
          test_dsmem_bandwidth, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

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

        cudaLaunchKernelEx(&config, test_dsmem_bandwidth, load_data_buffer, dummy_output_device, latency_output_dev, test_sm_byte/sizeof(int));
    }
  gpuErrchk(cudaDeviceSynchronize());

  int* latency_result = new int[CLUSTER_DIM];

  gpuErrchk(cudaMemcpy(latency_result, latency_output_dev,
                       CLUSTER_DIM * sizeof(int),
                       cudaMemcpyDeviceToHost));

  std::ofstream ofs("dsmem_bandwidth_16.csv");

  ofs << " ";

  for (int cid = 0; cid < CLUSTER_DIM; ++cid) {
    ofs << ",sm_" << cid;
  }
  ofs <<"\n";

  ofs << "byte_per_cycle";

  for (int cid = 0; cid < CLUSTER_DIM; ++cid) {
    double bw = REPEAT * test_sm_byte / static_cast<double>(latency_result[cid]);
    std::cout << "sm_0 read sm_" << cid
	   << " testsize:" << REPEAT * test_sm_byte << " test cycle: "
	   << latency_result[cid]  << " " << bw << " byte per cycle" << std::endl;
    ofs << "," << bw;
  }
  ofs << "\n";
}

