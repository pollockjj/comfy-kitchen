// SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t shared_address(const void* ptr) {
    uint32_t out;
    asm("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }"
        : "=r"(out) : "l"(ptr));
    return out;
}

__device__ __forceinline__ void load_matrix_x4(uint32_t (&dst)[4], uint32_t address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(address));
}

__device__ __forceinline__ void mma_bf16_m16n8k16(
    float (&acc)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
#endif
}

// This is the small-M tiling used by llama.cpp's mmf path, adapted to retain
// native BF16 activations and outputs. One CTA computes a 16x128 output tile;
// the M=1 and M=3 decode rows share every weight load across four warps.
__global__ void bf16_small_batch_linear_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
    constexpr int block_m = 16;
    constexpr int block_n = 128;
    constexpr int block_k = 64;
    constexpr int warps = 4;
    constexpr int warp_n = block_n / warps;
    constexpr int n_mma = warp_n / 8;
    constexpr int k_mma = block_k / 16;
    constexpr int shared_stride_k = block_k + 8;

    __shared__ alignas(16) __nv_bfloat16 x_shared[block_m * shared_stride_k];
    __shared__ alignas(16) __nv_bfloat16 w_shared[block_n * shared_stride_k];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n_base = blockIdx.x * block_n;
    const int warp_n_base = warp * warp_n;

    float acc[n_mma][4] = {};

    for (int k_base = 0; k_base < k; k_base += block_k) {
        const int global_n = n_base + tid;
        if (global_n < n) {
            const __nv_bfloat16* src = weight + static_cast<int64_t>(global_n) * k + k_base;
            #pragma unroll
            for (int offset = 0; offset < block_k; offset += 8) {
                *reinterpret_cast<uint4*>(&w_shared[tid * shared_stride_k + offset]) =
                    *reinterpret_cast<const uint4*>(src + offset);
            }
        } else {
            #pragma unroll
            for (int offset = 0; offset < block_k; offset += 8) {
                *reinterpret_cast<uint4*>(&w_shared[tid * shared_stride_k + offset]) = make_uint4(0, 0, 0, 0);
            }
        }

        const int x_row = tid / (block_k / 8);
        const int x_col = (tid % (block_k / 8)) * 8;
        uint4* x_dst = reinterpret_cast<uint4*>(&x_shared[x_row * shared_stride_k + x_col]);
        if (x_row < m) {
            *x_dst = *reinterpret_cast<const uint4*>(x + static_cast<int64_t>(x_row) * k + k_base + x_col);
        } else {
            *x_dst = make_uint4(0, 0, 0, 0);
        }

        __syncthreads();

        #pragma unroll
        for (int kt = 0; kt < k_mma; ++kt) {
            const int k_offset = kt * 16;
            uint32_t a[4];
            const __nv_bfloat16* a_ptr =
                &x_shared[(lane % 16) * shared_stride_k + (lane / 16) * 8 + k_offset];
            load_matrix_x4(a, shared_address(a_ptr));

            #pragma unroll
            for (int pair = 0; pair < n_mma / 2; ++pair) {
                const int n_offset = warp_n_base + pair * 16;
                uint32_t b4[4];
                const __nv_bfloat16* b_ptr =
                    &w_shared[(n_offset + lane % 16) * shared_stride_k + (lane / 16) * 8 + k_offset];
                load_matrix_x4(b4, shared_address(b_ptr));
                const uint32_t b0[2] = {b4[0], b4[2]};
                const uint32_t b1[2] = {b4[1], b4[3]};
                mma_bf16_m16n8k16(acc[pair * 2], a, b0);
                mma_bf16_m16n8k16(acc[pair * 2 + 1], a, b1);
            }
        }

        __syncthreads();
    }

    const int row0 = lane / 4;
    const int col0 = (lane % 4) * 2;
    #pragma unroll
    for (int nt = 0; nt < n_mma; ++nt) {
        const int global_n0 = n_base + warp_n_base + nt * 8 + col0;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = row0 + half * 8;
            if (row < m) {
                if (global_n0 < n) {
                    out[static_cast<int64_t>(row) * n + global_n0] =
                        __float2bfloat16_rn(acc[nt][half * 2]);
                }
                if (global_n0 + 1 < n) {
                    out[static_cast<int64_t>(row) * n + global_n0 + 1] =
                        __float2bfloat16_rn(acc[nt][half * 2 + 1]);
                }
            }
        }
    }
}

}  // namespace

extern "C" void launch_bf16_small_batch_linear_kernel(
    const void* x,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((n + 127) / 128);
    bf16_small_batch_linear_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(out),
        m,
        n,
        k);
}
