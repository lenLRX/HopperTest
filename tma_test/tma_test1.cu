__global__ void test_tma_1(float* input) {
}

int main(int argc, char** argv) {
  int64_t buffer_size = 1024*1024*1024;
  float* data_input_device = nullptr;
  cudaMalloc();
  float* data_output_device = nullptr;
  cuTensorMapEncodeTiled()
}
