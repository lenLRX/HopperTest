#include <iostream>
#include <stdint.h>
#include <math.h>

__global__ void test_main(uint16_t *output) {
  float a = NAN;
  float b = INFINITY;

  uint16_t c;
  uint16_t d;

  asm("cvt.rn.satfinite.e4m3x2.f32  %0, %1, %2;" : "=h"(c) : "f"(a), "f"(b));
  asm("cvt.rn.satfinite.e5m2x2.f32  %0, %1, %2;" : "=h"(c) : "f"(a), "f"(b));

  output[0] = c;
  output[1] = d;
}

int main() {
    uint16_t* output_host = new uint16_t[2];
    uint16_t* output_device;
    cudaMalloc(&output_device, 2 * sizeof(uint16_t));

    test_main<<<1,1>>>(output_device);

    cudaDeviceSynchronize();
    cudaMemcpy(output_host, output_device, 2*sizeof(uint16_t),
                       cudaMemcpyDeviceToHost);
    std::cout << "e4m3 nan: " << (output_host[0]&0xff) << " inf " << ((output_host[0]>>8)&0xff) << "\n";
    std::cout << "e5m2 nan: " << (output_host[1]&0xff) << " inf " << ((output_host[1]>>8)&0xff) << "\n";

}
