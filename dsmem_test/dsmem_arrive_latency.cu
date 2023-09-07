#include <fstream>
#include <iostream>

#include "hopper_util/util.h"

#include <cooperative_groups.h>

using namespace cooperative_groups;

#define CLUSTER_DIM 16
#define TEST_COUNT 32


__global__ void test_dsmem_arrive_latency(int* pointer_chase, int* dummy_output, int* latency_output) {
  dim3 cta_id_in_cluster = block_id_in_cluster();

  extern __shared__ char sm_base[];
  uint64_t* mb_arr = (uint64_t*)sm_base;
  uint64_t* sender_arr = mb_arr + TEST_COUNT;

  int dummy_out = threadIdx.x;

  if (threadIdx.x == 0) {
    for (int i = 0; i < TEST_COUNT; ++i) {
      initialize_barrier(mb_arr[i]);
      initialize_barrier(sender_arr[i]);
    }
  }

  if (cta_id_in_cluster.x > 0 && threadIdx.x > 0) {
    return;
  }

  __syncthreads();

      asm volatile(
      "{\n\t"
      "fence.mbarrier_init.release.cluster; \n"
      "}"
      ::);

  cluster_sync();

  if (cta_id_in_cluster.x == 0) {
    // test latency of each sm
    //printf("111 tid: %d\n", int(threadIdx.x));
    if (threadIdx.x == 1) {
      //printf("777\n");
      int latency_sum = 0;
      int phase = 0;
      for (int i = 0; i < TEST_COUNT; ++i) {
	int start = clock();
	//printf("666\n");
        //arrive_barrier(mb_arr[i]);
	//printf("555\n");
	//wait_barrier(sender_arr[i], phase);
	int end = clock();
        latency_sum += end - start;
      }
      latency_output[0] = latency_sum;
      //printf("222\n");

      phase = (phase + 1) % 2;

      for (int cid = 1; cid < CLUSTER_DIM; ++cid) {
	//printf("cluster %d start\n", cid);
	latency_sum = 0;
        for (int i = 0; i < TEST_COUNT; ++i) {
          int start = clock();

	  arrive_remote(mb_arr[i], cid, 1);
	  wait_barrier(sender_arr[i], phase);

          int end = clock();
	  latency_sum += end - start;
	}
        phase = (phase + 1) % 2;
	latency_output[cid] = latency_sum;
	//printf("cluster %d end\n", cid);
      }

    }
    else {
      printf("else %d\n", int(threadIdx.x));
      // thread 1
      int phase = 0;
      for (int i = 0; i < TEST_COUNT; ++i) {
	//printf("333\n");
	//arrive_barrier(sender_arr[i]);

        //wait_barrier(mb_arr[i], phase);
	//printf("444\n");
        //arrive_barrier(sender_arr[i]);
      }
    }
  }
  else {

    int phase = 0;
    for (int i = 0; i < TEST_COUNT; ++i) {
      wait_barrier(mb_arr[i], phase);
      //printf("cluster start %d\n", int(cta_id_in_cluster.x));
      arrive_remote(sender_arr[i], 0, 1);
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
  dim3 block_dim(2);

  size_t sm_size = TEST_COUNT * 2 * sizeof(uint64_t);


  std::cout << "sm_size: " << sm_size << std::endl;
  cudaFuncSetAttribute(test_dsmem_arrive_latency,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sm_size);
  cudaFuncSetAttribute(
          test_dsmem_arrive_latency, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

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

        cudaLaunchKernelEx(&config, test_dsmem_arrive_latency, pointer_chase_device, dummy_output_device, latency_output_dev);
    }
  gpuErrchk(cudaDeviceSynchronize());

  int* latency_result = new int[CLUSTER_DIM * TEST_COUNT];

  gpuErrchk(cudaMemcpy(latency_result, latency_output_dev,
                       CLUSTER_DIM * TEST_COUNT * sizeof(int),
                       cudaMemcpyDeviceToHost));

  std::ofstream ofs("dsmem_arrive_latency_16.csv");

  ofs << "latency(cycle)";

  for (int cid = 0; cid < CLUSTER_DIM; ++cid) {
    ofs << ",sm_" << cid;
  }
  ofs <<"\n";

  for (int cid = 0; cid < CLUSTER_DIM; ++cid) {
    double avg_latency = static_cast<double>(latency_result[cid]) / TEST_COUNT / 2;
    ofs << "," << avg_latency;
    std::cout << "sm_0 to sm_" << cid << " avg arrive latency: " << avg_latency << std::endl;
  }
}
