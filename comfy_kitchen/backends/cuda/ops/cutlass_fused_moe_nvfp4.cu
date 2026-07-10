/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * The NVFP4 block quantization and CUTLASS scale-factor layout follow the
 * NVIDIA implementation already used by Comfy Kitchen's quantize_nvfp4 op.
 */

#include "float_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

extern "C" bool launch_cutlass_grouped_gemm_nvfp4_variable(
    const void* a_ptr,
    const void* block_scale_a_ptr,
    const void* b_ptr,
    const void* block_scale_b_ptr,
    void* d_ptr,
    const float* alpha_ptr,
    const int32_t* m_indptr_ptr,
    int64_t num_groups,
    int64_t scale_group_m,
    int64_t n,
    int64_t k,
    int out_dtype_code,
    void* workspace_ptr,
    int64_t workspace_size,
    cudaStream_t stream);

namespace comfy {
namespace fused_moe_nvfp4 {

#if CUDA_VERSION >= 12080

constexpr int kNvfp4BlockSize = 16;
constexpr int kValuesPerThread = 4;
constexpr int kThreadsPerQuantGroup = kNvfp4BlockSize / kValuesPerThread;
constexpr int kQuantThreads = 256;
constexpr int kScaleRowAlignment = 128;

class WorkspaceArena {
public:
    WorkspaceArena(void* ptr, size_t size)
        : base_(static_cast<uint8_t*>(ptr)), size_(size), offset_(0) {}

    template <class T>
    T* allocate(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) {
            return nullptr;
        }
        return static_cast<T*>(allocate_bytes(count * sizeof(T)));
    }

    void* allocate_bytes(size_t bytes) {
        constexpr size_t alignment = 16;
        if (offset_ > SIZE_MAX - (alignment - 1)) {
            return nullptr;
        }
        const size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
        if (aligned > size_ || bytes > size_ - aligned) {
            return nullptr;
        }
        void* result = base_ + aligned;
        offset_ = aligned + bytes;
        return result;
    }

    void* tail() {
        return allocate_bytes(0);
    }

    size_t remaining() const {
        return offset_ <= size_ ? size_ - offset_ : 0;
    }

private:
    uint8_t* base_;
    size_t size_;
    size_t offset_;
};

__global__ void count_and_rank_routes(
    const int32_t* expert_ids,
    int32_t* counts,
    int32_t* route_rank,
    int routes,
    int num_experts) {
    for (int route = blockIdx.x * blockDim.x + threadIdx.x;
         route < routes;
         route += blockDim.x * gridDim.x) {
        const int expert = expert_ids[route];
        route_rank[route] =
            expert >= 0 && expert < num_experts ? atomicAdd(counts + expert, 1) : -1;
    }
}

__global__ void prefix_and_place_routes(
    const int32_t* expert_ids,
    int32_t* counts,
    int32_t* indptr,
    const int32_t* route_rank,
    int32_t* route_dest,
    int routes,
    int num_experts,
    int scale_group_m) {
    // One block is intentional: E=128 for DiffusionGemma, and this keeps the
    // prefix and placement stage on device without a host synchronization.
    if (threadIdx.x == 0) {
        int running = 0;
        for (int expert = 0; expert < num_experts; ++expert) {
            const int count = counts[expert] < scale_group_m ? counts[expert] : scale_group_m;
            counts[expert] = count;
            indptr[expert] = running;
            running += count;
        }
        indptr[num_experts] = running;
    }
    __syncthreads();

    for (int route = threadIdx.x; route < routes; route += blockDim.x) {
        const int expert = expert_ids[route];
        const int rank = route_rank[route];
        route_dest[route] =
            expert >= 0 && expert < num_experts && rank >= 0 && rank < scale_group_m
                ? indptr[expert] + rank
                : -1;
    }
}

__device__ __forceinline__ void quantize_four_low_first(
    float v0,
    float v1,
    float v2,
    float v3,
    float global_decode_scale,
    __nv_fp4x2_e2m1* qrow,
    size_t q_group_index,
    __nv_fp8_e4m3* scale_base,
    int scale_row,
    int scale_col,
    int scale_cols) {
    float block_absmax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
#pragma unroll
    for (int offset = kThreadsPerQuantGroup / 2; offset >= 1; offset /= 2) {
        block_absmax =
            fmaxf(block_absmax, __shfl_down_sync(0xffffffffu, block_absmax, offset,
                                                kThreadsPerQuantGroup));
    }
    block_absmax =
        __shfl_sync(0xffffffffu, block_absmax, 0, kThreadsPerQuantGroup);

    // Explicit zero-block handling avoids an infinite encode scale and keeps
    // both the packed values and E4M3 scale byte canonical zero.
    if (block_absmax == 0.0f) {
        if ((threadIdx.x & (kThreadsPerQuantGroup - 1)) == 0) {
            const size_t scale_offset =
                scale_factor_swizzled_offset(scale_row, scale_col, scale_cols);
            scale_base[scale_offset] = static_cast<__nv_fp8_e4m3>(0.0f);
        }
        store_fp4x4<__nv_fp4x2_e2m1, false>(qrow, q_group_index, 0.0f, 0.0f, 0.0f,
                                            0.0f);
        return;
    }

    float block_decode_scale = block_absmax * FP4LimitsTrait<__nv_fp4x2_storage_t>::max_inverse;
    block_decode_scale /= global_decode_scale;
    block_decode_scale = fminf(block_decode_scale, FP8LimitsTrait<__nv_fp8_e4m3>::max);
    const __nv_fp8_e4m3 scale_fp8 = static_cast<__nv_fp8_e4m3>(block_decode_scale);
    const float rounded_block_decode_scale = static_cast<float>(scale_fp8);

    if ((threadIdx.x & (kThreadsPerQuantGroup - 1)) == 0) {
        const size_t scale_offset =
            scale_factor_swizzled_offset(scale_row, scale_col, scale_cols);
        scale_base[scale_offset] = scale_fp8;
    }

    const float encode_scale =
        fminf(1.0f / (rounded_block_decode_scale * global_decode_scale), FLT_MAX);
    store_fp4x4<__nv_fp4x2_e2m1, false>(qrow, q_group_index, v0 * encode_scale,
                                        v1 * encode_scale, v2 * encode_scale,
                                        v3 * encode_scale);
}

__global__ void gather_and_quantize_input(
    const __nv_bfloat16* input,
    const int32_t* expert_ids,
    const int32_t* route_rank,
    const int32_t* route_dest,
    const float* input_decode_scale,
    __nv_fp4x2_e2m1* qdata,
    __nv_fp8_e4m3* block_scales,
    int hidden_size,
    int top_k,
    int scale_group_m,
    int scale_cols) {
    const int route = blockIdx.x;
    const int dest = route_dest[route];
    if (dest < 0) {
        return;
    }

    const int expert = expert_ids[route];
    const int rank = route_rank[route];
    const int token = route / top_k;
    const __nv_bfloat16* input_row = input + static_cast<size_t>(token) * hidden_size;
    __nv_fp4x2_e2m1* qrow = qdata + static_cast<size_t>(dest) * (hidden_size / 2);
    __nv_fp8_e4m3* scale_base =
        block_scales + static_cast<size_t>(expert) * scale_group_m * scale_cols;

    constexpr int groups_per_block = kQuantThreads / kThreadsPerQuantGroup;
    const int group_in_block = threadIdx.x / kThreadsPerQuantGroup;
    const int lane_in_group = threadIdx.x & (kThreadsPerQuantGroup - 1);
    const float global_decode_scale = input_decode_scale[0];
    for (int block_col = group_in_block; block_col < scale_cols;
         block_col += groups_per_block) {
        const int col = block_col * kNvfp4BlockSize + lane_in_group * kValuesPerThread;
        const float v0 = __bfloat162float(input_row[col]);
        const float v1 = __bfloat162float(input_row[col + 1]);
        const float v2 = __bfloat162float(input_row[col + 2]);
        const float v3 = __bfloat162float(input_row[col + 3]);
        const size_t q_group_index = static_cast<size_t>(block_col) * 4 + lane_in_group;
        quantize_four_low_first(v0, v1, v2, v3, global_decode_scale, qrow,
                                q_group_index, scale_base, rank, block_col, scale_cols);
    }
}

__device__ __forceinline__ __nv_bfloat16 gelu_tanh_bf16(__nv_bfloat16 value) {
    const float x = __bfloat162float(value);
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    constexpr float kCubicCoefficient = 0.044715f;
    const float gelu =
        0.5f * x * (1.0f + tanhf(kSqrtTwoOverPi * (x + kCubicCoefficient * x * x * x)));
    return __float2bfloat16_rn(gelu);
}

__global__ void activate_and_quantize_intermediate(
    const __nv_bfloat16* gate_up,
    const int32_t* expert_ids,
    const int32_t* route_rank,
    const int32_t* route_dest,
    const float* intermediate_decode_scale,
    __nv_fp4x2_e2m1* qdata,
    __nv_fp8_e4m3* block_scales,
    int intermediate_size,
    int scale_group_m,
    int scale_cols) {
    const int route = blockIdx.x;
    const int dest = route_dest[route];
    if (dest < 0) {
        return;
    }

    const int expert = expert_ids[route];
    const int rank = route_rank[route];
    const __nv_bfloat16* gate_up_row =
        gate_up + static_cast<size_t>(dest) * (2 * intermediate_size);
    __nv_fp4x2_e2m1* qrow =
        qdata + static_cast<size_t>(dest) * (intermediate_size / 2);
    __nv_fp8_e4m3* scale_base =
        block_scales + static_cast<size_t>(expert) * scale_group_m * scale_cols;

    constexpr int groups_per_block = kQuantThreads / kThreadsPerQuantGroup;
    const int group_in_block = threadIdx.x / kThreadsPerQuantGroup;
    const int lane_in_group = threadIdx.x & (kThreadsPerQuantGroup - 1);
    const float global_decode_scale = intermediate_decode_scale[0];
    for (int block_col = group_in_block; block_col < scale_cols;
         block_col += groups_per_block) {
        const int col = block_col * kNvfp4BlockSize + lane_in_group * kValuesPerThread;
        float values[kValuesPerThread];
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            // The stored fc1 order is [up, gate]. Match Comfy's BF16 execution
            // boundary: round GELU to BF16, then round the multiply to BF16,
            // before deriving the NVFP4 block scale.
            const __nv_bfloat16 up = gate_up_row[col + i];
            const __nv_bfloat16 gate = gate_up_row[intermediate_size + col + i];
            const __nv_bfloat16 gelu = gelu_tanh_bf16(gate);
            const __nv_bfloat16 product =
                __float2bfloat16_rn(__bfloat162float(gelu) * __bfloat162float(up));
            values[i] = __bfloat162float(product);
        }
        const size_t q_group_index = static_cast<size_t>(block_col) * 4 + lane_in_group;
        quantize_four_low_first(values[0], values[1], values[2], values[3],
                                global_decode_scale, qrow, q_group_index, scale_base,
                                rank, block_col, scale_cols);
    }
}

__global__ void weighted_route_reduction(
    const __nv_bfloat16* routed_output,
    const int32_t* route_dest,
    const float* router_weights,
    __nv_bfloat16* output,
    int num_tokens,
    int hidden_size,
    int top_k) {
    const int64_t elements = static_cast<int64_t>(num_tokens) * hidden_size;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int token = static_cast<int>(linear / hidden_size);
        const int col = static_cast<int>(linear - static_cast<int64_t>(token) * hidden_size);
        float sum = 0.0f;
        bool invalid_route = false;
#pragma unroll
        for (int position = 0; position < 8; ++position) {
            if (position >= top_k) {
                break;
            }
            const int route = token * top_k + position;
            const int dest = route_dest[route];
            if (dest < 0) {
                invalid_route = true;
                continue;
            }
            const float value = __bfloat162float(
                routed_output[static_cast<size_t>(dest) * hidden_size + col]);
            sum = fmaf(router_weights[route], value, sum);
        }
        output[linear] = __float2bfloat16_rn(invalid_route ? CUDART_NAN_F : sum);
    }
}

__host__ __forceinline__ bool is_aligned(const void* ptr, uintptr_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

#endif  // CUDA_VERSION >= 12080

}  // namespace fused_moe_nvfp4
}  // namespace comfy

extern "C" bool launch_cutlass_fused_moe_nvfp4(
    const void* input_bf16,
    const int32_t* expert_ids,
    const float* router_weights,
    const void* fc1_qdata,
    const void* fc1_block_scales,
    const void* fc2_qdata,
    const void* fc2_block_scales,
    const float* input_decode_scale,
    const float* intermediate_decode_scale,
    const float* alpha1,
    const float* alpha2,
    void* output_bf16,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_experts,
    int64_t top_k,
    void* workspace_ptr,
    int64_t workspace_size,
    cudaStream_t stream) {
#if CUDA_VERSION >= 12080
    using namespace comfy::fused_moe_nvfp4;

    if (input_bf16 == nullptr || expert_ids == nullptr || router_weights == nullptr ||
        fc1_qdata == nullptr || fc1_block_scales == nullptr || fc2_qdata == nullptr ||
        fc2_block_scales == nullptr || input_decode_scale == nullptr ||
        intermediate_decode_scale == nullptr || alpha1 == nullptr || alpha2 == nullptr ||
        output_bf16 == nullptr || workspace_ptr == nullptr || stream == nullptr) {
        return false;
    }

    // Revision 0 deliberately exposes only the two shapes exercised by the
    // DiffusionGemma text-only and visual-token workflows.
    if ((num_tokens != 256 && num_tokens != 340) || hidden_size != 2816 ||
        intermediate_size != 704 || num_experts != 128 || top_k != 8 ||
        workspace_size <= 0 || hidden_size % 64 != 0 || intermediate_size % 64 != 0) {
        return false;
    }
    if (!is_aligned(input_bf16, 2) || !is_aligned(expert_ids, 4) ||
        !is_aligned(router_weights, 4) || !is_aligned(fc1_qdata, 16) ||
        !is_aligned(fc1_block_scales, 16) || !is_aligned(fc2_qdata, 16) ||
        !is_aligned(fc2_block_scales, 16) || !is_aligned(input_decode_scale, 4) ||
        !is_aligned(intermediate_decode_scale, 4) || !is_aligned(alpha1, 4) ||
        !is_aligned(alpha2, 4) || !is_aligned(output_bf16, 2) ||
        !is_aligned(workspace_ptr, 16)) {
        return false;
    }

    const int n = static_cast<int>(num_tokens);
    const int h = static_cast<int>(hidden_size);
    const int i = static_cast<int>(intermediate_size);
    const int e = static_cast<int>(num_experts);
    const int k = static_cast<int>(top_k);
    const int routes = n * k;
    const int scale_group_m =
        ((n + kScaleRowAlignment - 1) / kScaleRowAlignment) * kScaleRowAlignment;
    const int input_scale_cols = h / kNvfp4BlockSize;
    const int intermediate_scale_cols = i / kNvfp4BlockSize;

    WorkspaceArena arena(workspace_ptr, static_cast<size_t>(workspace_size));
    // Persistent workspace layout. Every allocation is complete before the
    // tail is handed to either grouped GEMM, so CUTLASS cannot overlap any
    // routing, quantized activation, scale, or BF16 intermediate buffer.
    int32_t* counts = arena.allocate<int32_t>(e);
    int32_t* indptr = arena.allocate<int32_t>(e + 1);
    int32_t* route_rank = arena.allocate<int32_t>(routes);
    int32_t* route_dest = arena.allocate<int32_t>(routes);
    uint8_t* qx = arena.allocate<uint8_t>(static_cast<size_t>(routes) * (h / 2));
    uint8_t* input_block_scales = arena.allocate<uint8_t>(
        static_cast<size_t>(e) * scale_group_m * input_scale_cols);
    __nv_bfloat16* gate_up = arena.allocate<__nv_bfloat16>(
        static_cast<size_t>(routes) * (2 * i));
    uint8_t* qi = arena.allocate<uint8_t>(static_cast<size_t>(routes) * (i / 2));
    uint8_t* intermediate_block_scales = arena.allocate<uint8_t>(
        static_cast<size_t>(e) * scale_group_m * intermediate_scale_cols);
    __nv_bfloat16* routed_down =
        arena.allocate<__nv_bfloat16>(static_cast<size_t>(routes) * h);
    void* gemm_workspace = arena.tail();
    const size_t gemm_workspace_size = arena.remaining();
    if (counts == nullptr || indptr == nullptr || route_rank == nullptr ||
        route_dest == nullptr || qx == nullptr || input_block_scales == nullptr ||
        gate_up == nullptr || qi == nullptr || intermediate_block_scales == nullptr ||
        routed_down == nullptr || gemm_workspace == nullptr || gemm_workspace_size == 0 ||
        gemm_workspace_size > static_cast<size_t>(INT64_MAX)) {
        return false;
    }

    if (cudaMemsetAsync(counts, 0, static_cast<size_t>(e) * sizeof(int32_t), stream) !=
        cudaSuccess) {
        return false;
    }
    const int route_blocks = (routes + 255) / 256;
    count_and_rank_routes<<<route_blocks, 256, 0, stream>>>(
        expert_ids, counts, route_rank, routes, e);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }
    prefix_and_place_routes<<<1, 256, 0, stream>>>(
        expert_ids, counts, indptr, route_rank, route_dest, routes, e, scale_group_m);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }

    gather_and_quantize_input<<<routes, kQuantThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input_bf16), expert_ids, route_rank, route_dest,
        input_decode_scale, reinterpret_cast<__nv_fp4x2_e2m1*>(qx),
        reinterpret_cast<__nv_fp8_e4m3*>(input_block_scales), h, k, scale_group_m,
        input_scale_cols);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }

    try {
        if (!launch_cutlass_grouped_gemm_nvfp4_variable(
                qx, input_block_scales, fc1_qdata, fc1_block_scales, gate_up, alpha1,
                indptr, e, scale_group_m, 2 * i, h, 2, gemm_workspace,
                static_cast<int64_t>(gemm_workspace_size), stream)) {
            return false;
        }
    } catch (...) {
        return false;
    }

    activate_and_quantize_intermediate<<<routes, kQuantThreads, 0, stream>>>(
        gate_up, expert_ids, route_rank, route_dest, intermediate_decode_scale,
        reinterpret_cast<__nv_fp4x2_e2m1*>(qi),
        reinterpret_cast<__nv_fp8_e4m3*>(intermediate_block_scales), i, scale_group_m,
        intermediate_scale_cols);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }

    try {
        if (!launch_cutlass_grouped_gemm_nvfp4_variable(
                qi, intermediate_block_scales, fc2_qdata, fc2_block_scales, routed_down,
                alpha2, indptr, e, scale_group_m, h, i, 2, gemm_workspace,
                static_cast<int64_t>(gemm_workspace_size), stream)) {
            return false;
        }
    } catch (...) {
        return false;
    }

    const int64_t output_elements = num_tokens * hidden_size;
    const int reduction_blocks = static_cast<int>((output_elements + 255) / 256);
    weighted_route_reduction<<<reduction_blocks, 256, 0, stream>>>(
        routed_down, route_dest, router_weights, static_cast<__nv_bfloat16*>(output_bf16),
        n, h, k);
    return cudaPeekAtLastError() == cudaSuccess;
#else
    (void)input_bf16;
    (void)expert_ids;
    (void)router_weights;
    (void)fc1_qdata;
    (void)fc1_block_scales;
    (void)fc2_qdata;
    (void)fc2_block_scales;
    (void)input_decode_scale;
    (void)intermediate_decode_scale;
    (void)alpha1;
    (void)alpha2;
    (void)output_bf16;
    (void)num_tokens;
    (void)hidden_size;
    (void)intermediate_size;
    (void)num_experts;
    (void)top_k;
    (void)workspace_ptr;
    (void)workspace_size;
    (void)stream;
    return false;
#endif
}
