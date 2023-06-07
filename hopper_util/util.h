#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA) || defined(__clang__)
#  define CUTE_HOST_DEVICE __forceinline__ __host__ __device__
#  define CUTE_DEVICE      __forceinline__          __device__
#  define CUTE_HOST        __forceinline__ __host__
#else
#  define CUTE_HOST_DEVICE inline
#  define CUTE_DEVICE      inline
#  define CUTE_HOST        inline
#endif // CUTE_HOST_DEVICE, CUTE_DEVICE

#define CUTE_ARCH_TMA_SM90_ENABLED


#ifdef __CUDA_ARCH__
  #define CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED 1
#endif


// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && \
  ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))))
#  define CUTE_ARCH_CLUSTER_SM90_ENABLED
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
#  define CUTE_ARCH_ELECT_ONE_SM90_ENABLED
#endif

#define CUTLASS_DEVICE __device__
#define CUTE_RUNTIME_ASSERT(xxx)


CUTE_DEVICE void cluster_arrive_relaxed()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : : );
#else
  CUTE_RUNTIME_ASSERT("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_arrive()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.arrive.aligned;\n" : : );
#else
  CUTE_RUNTIME_ASSERT("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_wait()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.wait.aligned;\n" : : );
#else
  CUTE_RUNTIME_ASSERT("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_sync()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  cluster_arrive();
  cluster_wait();
#else
  CUTE_RUNTIME_ASSERT("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

// Returns the dim3 grid size in terms of number of clusters.
CUTE_DEVICE dim3 cluster_grid_dims()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %nclusterid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %nclusterid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %nclusterid.z;\n" : "=r"(z) : );
  return {x, y, z};
#elif defined(__CUDA_ARCH__)
  // MSVC requires protecting use of gridDim with __CUDA_ARCH__.
  return gridDim;
#elif defined(_MSC_VER)
  CUTE_RUNTIME_ASSERT("cluster_grid_dims() can only be called on device");
#else
  return {0, 0, 0};
#endif
}

// Returns the dim3 cluster rank in the grid.
CUTE_DEVICE dim3 cluster_id_in_grid()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %clusterid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %clusterid.z;\n" : "=r"(z) : );
  return {x, y, z};
#elif defined(__CUDA_ARCH__)
  // MSVC requires protecting use of blockIdx with __CUDA_ARCH__.
  return blockIdx;
#elif defined(_MSC_VER)
  CUTE_RUNTIME_ASSERT("cluster_id_in_grid() can only be called on device");
#else
  return {0, 0, 0};
#endif
}

// Returns the relative dim3 block rank local to the cluster.
CUTE_DEVICE dim3 block_id_in_cluster()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %cluster_ctaid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %cluster_ctaid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %cluster_ctaid.z;\n" : "=r"(z) : );
  return {x, y, z};
#else
  return {0,0,0};
#endif
}

// Returns the dim3 cluster shape.
CUTE_DEVICE dim3 cluster_shape()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %cluster_nctaid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %cluster_nctaid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %cluster_nctaid.z;\n" : "=r"(z) : );
  return {x, y, z};
#else
  return {1,1,1};
#endif
}

// Get 1D ctaid in a cluster.
CUTLASS_DEVICE uint32_t block_rank_in_cluster()
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t rank;
  asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
  return rank;
#else
  return 0;
#endif
}

// Set the destination block-ID in cluster for a given SMEM Address
CUTLASS_DEVICE uint32_t set_block_rank(uint32_t smemAddr, uint32_t rank)
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
              : "=r"(result)
              : "r"(smemAddr), "r"(rank));
  return result;
#else
  return smemAddr;
#endif
}

// Elect one thread in the warp. The elected thread gets its predicate set to true, all others obtain false.
CUTE_HOST_DEVICE uint32_t elect_one_sync()
{
#if defined(CUTE_ARCH_ELECT_ONE_SM90_ENABLED)
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %rx;\n"
    ".reg .pred %px;\n"
    "     elect.sync %rx|%px, %2;\n"
    "@%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return pred;
#elif defined(__CUDA_ARCH__)
  return (threadIdx.x % 32) == 0;
#else
  return true;
#endif
}

struct ElectOneLaneIdReturnType {
  uint32_t is_leader;
  uint32_t leader_lane_id;
};

CUTE_HOST_DEVICE
ElectOneLaneIdReturnType
elect_one_leader_sync()
{
#if defined(CUTE_ARCH_ELECT_ONE_SM90_ENABLED)
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %rx;\n"
    ".reg .pred %px;\n"
    "     elect.sync %rx|%px, %2;\n"
    "@%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return {pred, laneid};
#elif defined(__CUDA_ARCH__)
  return {(threadIdx.x % 32) == 0, 0};
#else
  return {true, 0};
#endif
}

// Store value to remote shared memory in the cluster
CUTE_DEVICE
void
store_shared_remote(uint32_t value, uint32_t smem_addr, uint32_t mbarrier_addr, uint32_t dst_cta_rank)
{
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t dsmem_addr = set_block_rank(smem_addr, dst_cta_rank);
  uint32_t remote_barrier_addr = set_block_rank(mbarrier_addr, dst_cta_rank);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];"
               : : "r"(dsmem_addr), "r"(value), "r"(remote_barrier_addr));
#endif
}





CUTE_DEVICE
uint32_t
cast_smem_ptr_to_uint(void const* const ptr)
{
#ifdef CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
#else
  return 0;
#endif
}



//////////////////////////////////////////////////////////////////////////////////////////////////////
/// Barriers are 64-bit of user-managed information used in broadly two types syncronization patterns
/// 1) arrive/wait on threads (usage: cp.async and warp-specialized kernels)
/// 2) transaction-based (usage: TMA transaction where a CTA issues one transaction)
//////////////////////////////////////////////////////////////////////////////////////////////////////

// Initialize barrier present in shared memory
CUTE_HOST_DEVICE
void
initialize_barrier(uint64_t& smem_barrier,                 // 64 bits user-manged barrier in smem
                   int thread_count = 1)                   // Thread count expected to arrive/wait on this barrier
{
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile ("mbarrier.init.shared.b64 [%0], %1;\n"
    :: "r"(smem_int_ptr),
       "r"(thread_count));
}

// Set the number of bytes transfered per transaction
CUTE_HOST_DEVICE
void
set_barrier_transaction_bytes(uint64_t& smem_barrier,      // 64 bits user-manged barrier in smem
                              uint32_t bytes)              // Number of bytes transfered by per TMA transaction
{
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile ("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
    :: "r"(smem_int_ptr),
       "r"(bytes));
}

// Barrier wait
CUTE_HOST_DEVICE
void
wait_barrier(uint64_t& smem_barrier,                       // 64 bits user-manged barrier in smem
             int phase_bit)                                // Current phase bit the barrier waiting to flip
{
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
    "{\n"
    ".reg .pred                P1;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
    "@P1                       bra.uni DONE;\n"
    "bra.uni                   LAB_WAIT;\n"
    "DONE:\n"
    "}\n"
    :: "r"(smem_int_ptr),
       "r"(phase_bit));

}

// Barrier arrive
CUTE_HOST_DEVICE
void
arrive_barrier(uint64_t& smem_barrier)                      // 64 bits user-manged barrier in smem
{
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
    "{\n"
    ".reg .b64 state; \n"
    "mbarrier.arrive.shared.b64   state, [%0];\n"
    "}\n"
    :: "r"(smem_int_ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TMA Descriptor and utilities
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace TMA {

enum class SmemSwizzleBits : uint8_t {
  DISABLE = 0,
  B32 = 1,
  B64 = 2,
  B128 = 3,
};

#if !defined(__CUDACC_RTC__)
#if (__CUDACC_VER_MAJOR__ >= 12)
#endif // (__CUDACC_VER_MAJOR__ >= 12)
#endif // !defined(__CUDACC_RTC__)
} // end namespace TMA

#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
using TmaDescriptor = CUtensorMap;
#else
using TmaDescriptor = struct { char bytes[128]; };
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
/// Initiates a TensorMap Prefetch
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
void
prefetch_tma_descriptor(TmaDescriptor const* desc_ptr)
{
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  // Prefetch TMA Descriptor using generic addressing (i.e. no specific state space: const or param)
  asm volatile (
    "prefetch.tensormap [%0];"
    :
    : "l"(gmem_int_desc)
    : "memory");
}




////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD : Initiates a TMA copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_1D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_2D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& crd0)
  {
    return SM90_TMA_LOAD_1D::copy(desc_ptr, smem_mbar, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    return SM90_TMA_LOAD_2D::copy(desc_ptr, smem_mbar, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
    return SM90_TMA_LOAD_3D::copy(desc_ptr, smem_mbar, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
    return SM90_TMA_LOAD_4D::copy(desc_ptr, smem_mbar, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    return SM90_TMA_LOAD_5D::copy(desc_ptr, smem_mbar, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD im2col: Initiates a TMA copy, in im2col mode, from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_IM2COL_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n,
       uint16_t const& offset_w)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2], {%6};"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_n),
        "h"(offset_w)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_IM2COL_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n,
       uint16_t const& offset_w,
       uint16_t const& offset_h)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], {%7, %8};"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n),
        "h"(offset_w), "h"(offset_h)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_IM2COL_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n,
       uint16_t const& offset_w,
       uint16_t const& offset_h,
       uint16_t const& offset_d)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], {%8, %9, %10};"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_d), "r"(coord_n),
        "h"(offset_w), "h"(offset_h), "h"(offset_d)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_IM2COL
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n,
       uint16_t const& offset_w)
  {
    return SM90_TMA_LOAD_IM2COL_3D::copy(desc_ptr, smem_mbar, smem_ptr,
                                         coord_c, coord_w, coord_n,
                                         offset_w);
  }

  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n,
       uint16_t const& offset_w,
       uint16_t const& offset_h)
  {
    return SM90_TMA_LOAD_IM2COL_4D::copy(desc_ptr, smem_mbar, smem_ptr,
					 coord_c, coord_w, coord_h, coord_n,
					 offset_w, offset_h);
  }

  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n,
       uint16_t const& offset_w,
       uint16_t const& offset_h,
       uint16_t const& offset_d)
  {
    return SM90_TMA_LOAD_IM2COL_5D::copy(desc_ptr, smem_mbar, smem_ptr,
					 coord_c, coord_w, coord_h, coord_d, coord_n,
					 offset_w, offset_h, offset_d);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD_MULTICAST: Initiates a TMA copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_MULTICAST_1D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar, uint16_t multicast_mask,
       void const* const smem_ptr,
       int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%4}], [%2], %3;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask),
        "r"(crd0)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_MULTICAST_2D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar, uint16_t multicast_mask,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%4, %5}], [%2], %3;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask),
        "r"(crd0), "r"(crd1)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_MULTICAST_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar, uint16_t multicast_mask,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%4, %5, %6}], [%2], %3;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask),
        "r"(crd0), "r"(crd1), "r"(crd2)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_MULTICAST_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar, uint16_t multicast_mask,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%4, %5, %6, %7}], [%2], %3;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask),
        "r"(crd0), "r"(crd1), "r"(crd2),  "r"(crd3)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_MULTICAST_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar, uint16_t multicast_mask,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%4, %5, %6, %7, %8}], [%2], %3;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_MULTICAST
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar, uint16_t multicast_mask,
       void const* const smem_ptr,
       int32_t const& crd0)
  {
    return SM90_TMA_LOAD_MULTICAST_1D::copy(desc_ptr, smem_mbar, multicast_mask, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar, uint16_t multicast_mask,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    return SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, smem_mbar, multicast_mask, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar, uint16_t multicast_mask,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
    return SM90_TMA_LOAD_MULTICAST_3D::copy(desc_ptr, smem_mbar, multicast_mask, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar, uint16_t multicast_mask,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
    return SM90_TMA_LOAD_MULTICAST_4D::copy(desc_ptr, smem_mbar, multicast_mask, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar, uint16_t multicast_mask,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    return SM90_TMA_LOAD_MULTICAST_5D::copy(desc_ptr, smem_mbar, multicast_mask, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD_MULTICAST im2col: Initiates a TMA copy, in im2col mode, from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_IM2COL_MULTICAST_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       uint16_t const& multicast_mask,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n,
       uint16_t const& offset_w)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4, %5}], [%2], {%6}, %7;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_n),
        "h"(offset_w),
	"h"(multicast_mask)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_IM2COL_MULTICAST_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       uint16_t const& multicast_mask,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n,
       uint16_t const& offset_w,
       uint16_t const& offset_h)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], {%7, %8}, %9;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n),
        "h"(offset_w), "h"(offset_h),
	"h"(multicast_mask)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_IM2COL_MULTICAST_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       uint16_t const& multicast_mask,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n,
       uint16_t const& offset_w,
       uint16_t const& offset_h,
       uint16_t const& offset_d)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], {%8, %9, %10}, %11;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_d), "r"(coord_n),
        "h"(offset_w), "h"(offset_h), "h"(offset_d),
	"h"(multicast_mask)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_IM2COL_MULTICAST
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       uint16_t const& multicast_mask,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n,
       uint16_t const& offset_w)
  {
    return SM90_TMA_LOAD_IM2COL_MULTICAST_3D::copy(desc_ptr, smem_mbar,
						   multicast_mask, smem_ptr,
						   coord_c, coord_w, coord_n,
						   offset_w);
  }

  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       uint16_t const& multicast_mask,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n,
       uint16_t const& offset_w,
       uint16_t const& offset_h)
  {
    return SM90_TMA_LOAD_IM2COL_MULTICAST_4D::copy(desc_ptr, smem_mbar,
						   multicast_mask, smem_ptr,
						   coord_c, coord_w, coord_h, coord_n,
						   offset_w, offset_h);
  }

  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr, uint64_t& smem_mbar,
       uint16_t const& multicast_mask,
       void const* const smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n,
       uint16_t const& offset_w,
       uint16_t const& offset_h,
       uint16_t const& offset_d)
  {
    return SM90_TMA_LOAD_IM2COL_MULTICAST_5D::copy(desc_ptr, smem_mbar,
                                                   multicast_mask, smem_ptr,
                                                   coord_c, coord_w, coord_h, coord_d, coord_n,
						   offset_w, offset_h, offset_d);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_STORE : Initiates a TMA copy from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_STORE_1D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%2}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_2D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1), "r"(crd2)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile (
      "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0)
  {
    return SM90_TMA_STORE_1D::copy(desc_ptr, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    return SM90_TMA_STORE_2D::copy(desc_ptr, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
    return SM90_TMA_STORE_3D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
    return SM90_TMA_STORE_4D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    return SM90_TMA_STORE_5D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
};

// Indicate arrival of warp issuing TMA_STORE
CUTE_HOST_DEVICE static void
tma_store_arrive() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    asm volatile("cp.async.bulk.commit_group;");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

// Wait on prior N (Count) TMA_STORE instructions to complete
template <int Count>
CUTE_HOST_DEVICE static void
tma_store_wait() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    asm volatile(
      "cp.async.bulk.wait_group.read %0;"
      :
      : "n"(Count)
      : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// BULK_COPY : Copy a bulk of memory between shared memory and global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_BULK_COPY_G2S
{
  CUTE_HOST_DEVICE static void
  copy(void const* const gmem_ptr, uint64_t& smem_mbar,
       void const* const smem_ptr, int32_t load_bytes)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :
                     : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(load_bytes), "r"(smem_int_mbar)
                     : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use BULK_COPY without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_BULK_COPY_S2G
{
  CUTE_HOST_DEVICE static void
  copy(void const* const smem_ptr,
       void const* const gmem_ptr, int32_t store_bytes)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                     :
                     : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
                     : "memory");
#else
    CUTE_RUNTIME_ASSERT("Trying to use BULK_COPY without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_BULK_COPY_AUTO {};

////////////////////////////////////////////////////////////////////////////////////////////////////


