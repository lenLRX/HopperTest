#include <fstream>
#include <iostream>

#include "hopper_util/util.h"

#include <cooperative_groups.h>

using namespace cooperative_groups;

#define REPEAT (128 * 8)

__global__ void test_dsmem_bandwidth(int* load_data, int* dummy_output, int* latency_output, int cluster_dim, int test_sm_count) {
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

  uint32_t smem_addr = cast_smem_ptr_to_uint(data_sm);
  for (int cid = 0; cid < cluster_dim; ++cid) {
    if (cid == cta_id_in_cluster.x) {
      continue;
    }
    int pos = 0;
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

  }
  

  cluster_sync();

  for (int i = 0; i < 8; ++i) {
    dummy_out += dummy_x[i];
  }

  dummy_output[0] = dummy_out;
}

void test_dsmem_bandwidth_cluster(int cluster_dim) {
  int test_sm_byte = 128 * 1024; // 128 kb
  int* load_data_buffer = nullptr;
  cudaMalloc(&load_data_buffer, test_sm_byte);

  int* dummy_output_device = nullptr;
  cudaMalloc(&dummy_output_device, test_sm_byte);

  int* latency_output_dev = nullptr;
  cudaMalloc(&latency_output_dev, cluster_dim * sizeof(int));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm_count = prop.multiProcessorCount;

  int grid_dim = sm_count / cluster_dim * cluster_dim; // floor
  dim3 block_dim(512);

  size_t sm_size = test_sm_byte;

  std::cout << "sm_size: " << sm_size << std::endl;
  cudaFuncSetAttribute(test_dsmem_bandwidth,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sm_size);
  cudaFuncSetAttribute(
          test_dsmem_bandwidth, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  // Kernel invocation with runtime cluster size
    
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim = grid_dim;
        config.blockDim = block_dim;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = cluster_dim; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.dynamicSmemBytes = sm_size;

    float duration;
    cudaEvent_t start_event, stop_event;

    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    gpuErrchk(cudaEventRecord(start_event, 0));

    cudaLaunchKernelEx(&config, test_dsmem_bandwidth, load_data_buffer, dummy_output_device, latency_output_dev, cluster_dim, test_sm_byte/sizeof(int));

    gpuErrchk(cudaEventRecord(stop_event, 0));
    gpuErrchk(cudaEventSynchronize(stop_event));
    gpuErrchk(cudaEventElapsedTime(&duration, start_event, stop_event));
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);


  gpuErrchk(cudaDeviceSynchronize());

  double bw = static_cast<double>(test_sm_byte) * (cluster_dim - 1) * grid_dim * REPEAT / duration / 1000 / 1000; // GB/s
  printf("cluster %d dsmem all to all duration %6.2fms read bandwidth: %7.2fGB\n", cluster_dim, duration, bw);
}

int main(int argc, char** argv) {
  test_dsmem_bandwidth_cluster(2);
  test_dsmem_bandwidth_cluster(4);
  test_dsmem_bandwidth_cluster(8);
  test_dsmem_bandwidth_cluster(16);
}

