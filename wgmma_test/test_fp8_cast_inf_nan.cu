#include <iostream>
#include <stdint.h>
#include <math.h>

__global__ void test_main(uint16_t *output) {
  float a = NAN;
  float b = INFINITY;
  float o = 1000000;

  uint16_t c;
  uint16_t d;
  uint16_t e;
  uint16_t f;
  uint16_t g;
  uint16_t h;

  asm("cvt.rn.satfinite.e4m3x2.f32  %0, %1, %2;" : "=h"(c) : "f"(a), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32  %0, %1, %2;" : "=h"(d) : "f"(b), "f"(b));
  asm("cvt.rn.satfinite.e4m3x2.f32  %0, %1, %2;" : "=h"(e) : "f"(o), "f"(o));




  asm("cvt.rn.satfinite.e5m2x2.f32  %0, %1, %2;" : "=h"(f) : "f"(a), "f"(a));
  asm("cvt.rn.satfinite.e5m2x2.f32  %0, %1, %2;" : "=h"(g) : "f"(b), "f"(b));
  asm("cvt.rn.satfinite.e5m2x2.f32  %0, %1, %2;" : "=h"(h) : "f"(o), "f"(o));

  output[0] = c;
  output[1] = d;
  output[2] = e;

  output[3] = f;
  output[4] = g;
  output[5] = h;


}

int main() {
    uint16_t* output_host = new uint16_t[6];
    uint16_t* output_device;
    cudaMalloc(&output_device, 6 * sizeof(uint16_t));

    test_main<<<1,1>>>(output_device);

    cudaDeviceSynchronize();
    cudaMemcpy(output_host, output_device, 6*sizeof(uint16_t),
                       cudaMemcpyDeviceToHost);
    std::cout << "e4m3 nan: " << (output_host[0]&0xff) << "\n";
    std::cout << "e4m3 inf: " << (output_host[1]&0xff) << "\n";
    std::cout << "e4m3 ovf: " << (output_host[2]&0xff) << "\n";
    std::cout << "e5m2 nan: " << (output_host[3]&0xff) << "\n";
    std::cout << "e5m2 inf: " << (output_host[4]&0xff) << "\n";
    std::cout << "e5m2 ovf: " << (output_host[5]&0xff) << "\n";



}
