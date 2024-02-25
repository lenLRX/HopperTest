#include <iostream>
#include <stdint.h>
#include <math.h>

#include <cuda_fp8.h>

__global__ void test_main(float* input, uint16_t *output) {
  //float a = NAN;
  //float b = INFINITY;
  //float o = 1000000;
  float a = input[0];
  float b = input[1];
  float o = input[2];

  uint16_t c;
  uint16_t d;
  uint16_t e;
  uint16_t f;
  uint16_t g;
  uint16_t h;

  c = __nv_cvt_float_to_fp8(a, __NV_NOSAT, __NV_E4M3);
  d = __nv_cvt_float_to_fp8(b, __NV_NOSAT, __NV_E4M3);
  e = __nv_cvt_float_to_fp8(o, __NV_NOSAT, __NV_E4M3);

  f = __nv_cvt_float_to_fp8(a, __NV_NOSAT, __NV_E5M2);
  g = __nv_cvt_float_to_fp8(b, __NV_NOSAT, __NV_E5M2);
  h = __nv_cvt_float_to_fp8(o, __NV_NOSAT, __NV_E5M2);



  output[0] = c;
  output[1] = d;
  output[2] = e;

  output[3] = f;
  output[4] = g;
  output[5] = h;


}

int main() {
    float* input_host = new float[3];
    float* input_device;
    cudaMalloc(&input_device, 3 * sizeof(float));
    cudaMemcpy(input_host, input_device, 3*sizeof(float),
                       cudaMemcpyHostToDevice);

    uint16_t* output_host = new uint16_t[6];
    uint16_t* output_device;
    cudaMalloc(&output_device, 6 * sizeof(uint16_t));

    test_main<<<1,1>>>(input_device, output_device);

    cudaDeviceSynchronize();
    cudaMemcpy(output_host, output_device, 6*sizeof(uint16_t),
                       cudaMemcpyDeviceToHost);
    std::cout << "e4m3 nan cast nosat: " << (output_host[0]&0xff) << "\n";
    std::cout << "e4m3 inf cast nosat: " << (output_host[1]&0xff) << "\n";
    std::cout << "e4m3 ovf cast nosat: " << (output_host[2]&0xff) << "\n";
    std::cout << "e5m2 nan cast nosat: " << (output_host[3]&0xff) << "\n";
    std::cout << "e5m2 inf cast nosat: " << (output_host[4]&0xff) << "\n";
    std::cout << "e5m2 ovf cast nosat: " << (output_host[5]&0xff) << "\n";



}
