import torch
import nvtx

def test_gemm(m, k, n):
    test_name = "gemm(m={},k={},n={})".format(m,k,n)
    print(test_name)
    dev = torch.device("cuda")
    lhs = torch.rand((m, k), device=dev).half()
    rhs = torch.rand((k, n), device=dev).half()
    nvtx_rng = nvtx.start_range(message=test_name)
    lhs@rhs
    nvtx.end_range(nvtx_rng)

if __name__ == '__main__':
    # part1
    '''
    test_gemm(3072, 6144, 256)
    test_gemm(1536, 6144, 256)
    test_gemm(1536, 6144, 512)
    test_gemm(3072, 6144, 256)
    test_gemm(3072, 12288, 512)
    test_gemm(3072, 6144, 512)
    test_gemm(1536, 12288, 512)
    test_gemm(3072, 6144, 512)
    test_gemm(2048, 12288, 768)
    test_gemm(3072, 6144, 512)
    # part2
    test_gemm(1024, 6144, 768)
    test_gemm(2048, 12288, 768)
    test_gemm(1024, 12288, 1536)
    test_gemm(1536, 6144, 1024)
    test_gemm(768, 6144, 1536)
    '''
    # part3
    
    test_gemm(8192, 6144, 1536)
    test_gemm(6144, 6144, 2048)
    test_gemm(2048, 6144, 2304)
    test_gemm(3072, 6144, 2048)
    


