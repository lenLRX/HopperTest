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


#if !defined(CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED) && CUTE_CVTA_GENERIC_TO_SHARED_SUPPORTED && defined(__CUDA_ARCH__)
  #define CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED 1
#endif


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


