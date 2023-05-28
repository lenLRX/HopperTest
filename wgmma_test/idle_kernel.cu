__global__ void idle_kernel(int* output) {
  while(1) {
    __nanosleep(10000000);
  }
  output[threadIdx.x] = clock();
}

int main(int argc, char** argv) {
  int* dev_dummy_out;
  cudaMalloc(&dev_dummy_out, sizeof(int));
  idle_kernel<<<1, 1>>>(dev_dummy_out);
  cudaDeviceSynchronize();
}

