import torch
import nvtx

def test_gemm(m, k, n):
    test_name = "gemm(m={},k={},n={})".format(m,k,n)
    print(test_name)
    dev = torch.device("cuda")
    lhs = torch.zeros((m, k), device=dev).half()
    rhs = torch.zeros((k, n), device=dev).half()
    nvtx_rng = nvtx.start_range(message=test_name)
    lhs@rhs
    nvtx.end_range(nvtx_rng)

if __name__ == '__main__':
    test_gemm(8192, 6145, 1536)
    test_gemm(8193, 6144, 1536)
    test_gemm(8192, 6144, 1537)
    test_gemm(8193, 6145, 1537)
    test_gemm(128, 12288, 128)



