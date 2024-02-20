#include <iostream>
#include <stdint.h>
#include <math.h>

__global__ void test_main(uint16_t *output) {
  uint8_t e4m3_nan = 255;
  uint8_t e5m2_nan = 255;
  uint8_t e5m2_neg_inf = 252;
  uint8_t e5m2_inf = 124;

  uint16_t c = (e4m3_nan << 8) | e4m3_nan;
  uint16_t d = (e5m2_nan << 8) | e5m2_nan;
  uint16_t e = (e5m2_neg_inf << 8) | e5m2_neg_inf;
  uint16_t a = (e5m2_inf << 8) | e5m2_inf;
  uint32_t f;
  uint32_t g;
  uint32_t h;
  uint32_t b;

  asm("cvt.rn.f16x2.e4m3x2  %0, %1;" : "=r"(f) : "h"(c));
  asm("cvt.rn.f16x2.e5m2x2  %0, %1;" : "=r"(g) : "h"(d));
  asm("cvt.rn.f16x2.e5m2x2  %0, %1;" : "=r"(h) : "h"(e));
  asm("cvt.rn.f16x2.e5m2x2  %0, %1;" : "=r"(b) : "h"(a));


  output[0] = (f & 0xffff);
  output[1] = (g & 0xffff);
  output[2] = (h & 0xffff);
  output[3] = (b & 0xffff);

}

int main() {
    uint16_t* output_host = new uint16_t[6];
    uint16_t* output_device;
    cudaMalloc(&output_device, 6 * sizeof(uint16_t));

    test_main<<<1,1>>>(output_device);

    cudaDeviceSynchronize();
    cudaMemcpy(output_host, output_device, 6*sizeof(uint16_t),
                       cudaMemcpyDeviceToHost);
    std::cout << "e4m3 nan to fp16: " << (output_host[0]) << "\n";
    std::cout << "e5m2 nan to fp16: " << (output_host[1]) << "\n";
    std::cout << "e5m2 -inf to fp16: " << (output_host[2]) << "\n";
    std::cout << "e5m2 inf to fp16: " << (output_host[3]) << "\n";



}
