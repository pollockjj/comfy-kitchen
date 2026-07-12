/*
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
 */

#include "utils.cuh"
#include "float_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

extern "C" bool launch_cutlass_grouped_gemm_mxfp8_variable(
    const void* activation_ptr,
    const void* activation_scale_ptr,
    const void* weight_ptr,
    const void* weight_scale_ptr,
    void* output_ptr,
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
namespace fused_moe_mxfp8 {

thread_local int last_error_stage = 0;

#if CUDA_VERSION >= 12080

constexpr int kBlockSize = 32;
constexpr int kValuesPerThread = 8;
constexpr int kThreadsPerQuantGroup = kBlockSize / kValuesPerThread;
constexpr int kQuantThreads = 256;
constexpr int kScaleRowAlignment = 128;
constexpr int kTopK = 8;

__global__ void prepare_scaled_routes(
    const int64_t* source_expert_ids,
    const float* normalized_router_weights,
    const __nv_bfloat16* expert_scale,
    int32_t* expert_ids,
    float* router_weights,
    int routes,
    int num_experts) {
    for (int route = blockIdx.x * blockDim.x + threadIdx.x;
         route < routes;
         route += blockDim.x * gridDim.x) {
        const int64_t expert = source_expert_ids[route];
        if (expert >= 0 && expert < num_experts) {
            expert_ids[route] = static_cast<int32_t>(expert);
            router_weights[route] = __fmul_rn(
                normalized_router_weights[route],
                __bfloat162float(expert_scale[expert]));
        } else {
            expert_ids[route] = -1;
            router_weights[route] = __int_as_float(0x7fc00000);
        }
    }
}

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

    void* tail(size_t alignment) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
            offset_ > SIZE_MAX - (alignment - 1)) {
            return nullptr;
        }
        const size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
        if (aligned > size_) {
            return nullptr;
        }
        offset_ = aligned;
        return base_ + aligned;
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

template <class T>
__device__ __forceinline__ float lowp_to_float(T value);

template <>
__device__ __forceinline__ float lowp_to_float(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float lowp_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <class T>
__device__ __forceinline__ T float_to_lowp(float value);

template <>
__device__ __forceinline__ __half float_to_lowp(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_lowp(float value) {
    return __float2bfloat16_rn(value);
}

struct PackedMxfp8x8 {
    uint64_t values;
    uint8_t scale;
};

template <class InputType>
__device__ __forceinline__ PackedMxfp8x8 quantize_eight(const InputType* values) {
    float block_absmax = 0.0f;
#pragma unroll
    for (int index = 0; index < kValuesPerThread; ++index) {
        block_absmax = fmaxf(block_absmax, fabsf(lowp_to_float(values[index])));
    }
#pragma unroll
    for (int offset = kThreadsPerQuantGroup / 2; offset >= 1; offset /= 2) {
        block_absmax = fmaxf(
            block_absmax,
            __shfl_xor_sync(0xffffffffu, block_absmax, offset, kThreadsPerQuantGroup));
    }

    constexpr float kFp8Max = 448.0f;
    const __nv_fp8_storage_t e8m0 = __nv_cvt_float_to_e8m0(
        block_absmax / kFp8Max, __NV_SATFINITE, cudaRoundPosInf);
    const uint32_t encode_bits = static_cast<uint32_t>(254 - e8m0) << 23;
    const float encode_scale = __uint_as_float(encode_bits);

    union {
        uint64_t u64;
        __nv_fp8_e4m3 fp8[kValuesPerThread];
    } packed;
#pragma unroll
    for (int index = 0; index < kValuesPerThread; ++index) {
        float scaled = lowp_to_float(values[index]) * encode_scale;
        scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
        packed.fp8[index] = static_cast<__nv_fp8_e4m3>(scaled);
    }
    return {packed.u64, e8m0};
}

template <class InputType>
__global__ void quantize_and_route_input_ranked(
    const InputType* input,
    const int32_t* expert_ids,
    const int32_t* route_rank,
    const int32_t* route_dest,
    uint8_t* qdata,
    uint8_t* block_scales,
    int hidden_size,
    int top_k,
    int scale_group_m,
    int block_cols,
    int scale_storage_cols) {
    const int token = blockIdx.x;
    const InputType* input_row = input + static_cast<size_t>(token) * hidden_size;
    constexpr int groups_per_block = kQuantThreads / kThreadsPerQuantGroup;
    const int group_in_block = threadIdx.x / kThreadsPerQuantGroup;
    const int lane_in_group = threadIdx.x & (kThreadsPerQuantGroup - 1);

    for (int block_col = group_in_block; block_col < block_cols;
         block_col += groups_per_block) {
        const int col = block_col * kBlockSize + lane_in_group * kValuesPerThread;
        const PackedMxfp8x8 packed = quantize_eight(input_row + col);
#pragma unroll
        for (int position = 0; position < kTopK; ++position) {
            if (position >= top_k) {
                break;
            }
            const int route = token * top_k + position;
            const int dest = route_dest[route];
            if (dest < 0) {
                continue;
            }
            uint8_t* qrow = qdata + static_cast<size_t>(dest) * hidden_size;
            *reinterpret_cast<uint64_t*>(qrow + col) = packed.values;
            if (lane_in_group == 0) {
                const int expert = expert_ids[route];
                const int rank = route_rank[route];
                uint8_t* scale_base =
                    block_scales + static_cast<size_t>(expert) * scale_group_m *
                        scale_storage_cols;
                const size_t scale_offset =
                    scale_factor_swizzled_offset(rank, block_col, block_cols);
                scale_base[scale_offset] = packed.scale;
            }
        }
    }
}

template <class OutputType>
__device__ __forceinline__ OutputType geglu_product(OutputType gate, OutputType up) {
    const float x = lowp_to_float(gate);
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    constexpr float kCubicCoefficient = 0.044715f;
    const float gelu =
        0.5f * x * (1.0f + tanhf(kSqrtTwoOverPi * (x + kCubicCoefficient * x * x * x)));
    const OutputType rounded_gelu = float_to_lowp<OutputType>(gelu);
    return float_to_lowp<OutputType>(
        lowp_to_float(rounded_gelu) * lowp_to_float(up));
}

template <class OutputType>
__global__ void activate_and_quantize_intermediate(
    const OutputType* gate_up,
    const int32_t* expert_ids,
    const int32_t* route_rank,
    const int32_t* route_dest,
    uint8_t* qdata,
    uint8_t* block_scales,
    int intermediate_size,
    int scale_group_m,
    int block_cols,
    int scale_storage_cols) {
    const int route = blockIdx.x;
    const int dest = route_dest[route];
    if (dest < 0) {
        return;
    }

    const int expert = expert_ids[route];
    const int rank = route_rank[route];
    const OutputType* gate_up_row =
        gate_up + static_cast<size_t>(dest) * (2 * intermediate_size);
    uint8_t* qrow = qdata + static_cast<size_t>(dest) * intermediate_size;
    uint8_t* scale_base =
        block_scales + static_cast<size_t>(expert) * scale_group_m * scale_storage_cols;
    constexpr int groups_per_block = kQuantThreads / kThreadsPerQuantGroup;
    const int group_in_block = threadIdx.x / kThreadsPerQuantGroup;
    const int lane_in_group = threadIdx.x & (kThreadsPerQuantGroup - 1);

    for (int block_col = group_in_block; block_col < block_cols;
         block_col += groups_per_block) {
        const int col = block_col * kBlockSize + lane_in_group * kValuesPerThread;
        OutputType values[kValuesPerThread];
#pragma unroll
        for (int index = 0; index < kValuesPerThread; ++index) {
            const OutputType gate = gate_up_row[col + index];
            const OutputType up = gate_up_row[intermediate_size + col + index];
            values[index] = geglu_product(gate, up);
        }
        const PackedMxfp8x8 packed = quantize_eight(values);
        *reinterpret_cast<uint64_t*>(qrow + col) = packed.values;
        if (lane_in_group == 0) {
            const size_t scale_offset =
                scale_factor_swizzled_offset(rank, block_col, block_cols);
            scale_base[scale_offset] = packed.scale;
        }
    }
}

template <class OutputType>
__global__ void weighted_route_reduction(
    const OutputType* routed_output,
    const int32_t* route_dest,
    const float* router_weights,
    OutputType* output,
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
        for (int position = 0; position < kTopK; ++position) {
            if (position >= top_k) {
                break;
            }
            const int route = token * top_k + position;
            const int dest = route_dest[route];
            if (dest < 0) {
                invalid_route = true;
                continue;
            }
            const float value = lowp_to_float(
                routed_output[static_cast<size_t>(dest) * hidden_size + col]);
            sum = fmaf(router_weights[route], value, sum);
        }
        output[linear] = float_to_lowp<OutputType>(
            invalid_route ? __int_as_float(0x7fc00000) : sum);
    }
}

__host__ __forceinline__ bool is_aligned(const void* ptr, uintptr_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

template <class OutputType>
bool run_fused_moe_mxfp8(
    const void* input,
    const int32_t* expert_ids,
    const float* router_weights,
    const int64_t* source_expert_ids,
    const __nv_bfloat16* expert_scale,
    const void* fc1_qdata,
    const void* fc1_block_scales,
    const void* fc2_qdata,
    const void* fc2_block_scales,
    void* output,
    int n,
    int h,
    int i,
    int e,
    int top_k,
    void* workspace_ptr,
    size_t workspace_size,
    int out_dtype_code,
    cudaStream_t stream) {
    const int routes = n * top_k;
    const int scale_group_m =
        ((n + kScaleRowAlignment - 1) / kScaleRowAlignment) * kScaleRowAlignment;
    const int input_block_cols = h / kBlockSize;
    const int input_scale_cols = ((input_block_cols + 3) / 4) * 4;
    const int intermediate_block_cols = i / kBlockSize;
    const int intermediate_scale_cols = ((intermediate_block_cols + 3) / 4) * 4;

    last_error_stage = 2;
    WorkspaceArena arena(workspace_ptr, workspace_size);
    int32_t* prepared_expert_ids = nullptr;
    float* prepared_router_weights = nullptr;
    if (source_expert_ids != nullptr) {
        prepared_expert_ids = arena.allocate<int32_t>(routes);
        prepared_router_weights = arena.allocate<float>(routes);
    }
    int32_t* counts = arena.allocate<int32_t>(e);
    int32_t* indptr = arena.allocate<int32_t>(e + 1);
    int32_t* route_rank = arena.allocate<int32_t>(routes);
    int32_t* route_dest = arena.allocate<int32_t>(routes);
    uint8_t* qx = arena.allocate<uint8_t>(static_cast<size_t>(routes) * h);
    uint8_t* input_block_scales = arena.allocate<uint8_t>(
        static_cast<size_t>(e) * scale_group_m * input_scale_cols);
    OutputType* gate_up =
        arena.allocate<OutputType>(static_cast<size_t>(routes) * (2 * i));
    uint8_t* qi = arena.allocate<uint8_t>(static_cast<size_t>(routes) * i);
    uint8_t* intermediate_block_scales = arena.allocate<uint8_t>(
        static_cast<size_t>(e) * scale_group_m * intermediate_scale_cols);
    OutputType* routed_down =
        arena.allocate<OutputType>(static_cast<size_t>(routes) * h);
    void* gemm_workspace = arena.tail(256);
    const size_t gemm_workspace_size = arena.remaining();
    if ((source_expert_ids != nullptr &&
         (prepared_expert_ids == nullptr || prepared_router_weights == nullptr)) ||
        counts == nullptr || indptr == nullptr || route_rank == nullptr ||
        route_dest == nullptr || qx == nullptr || input_block_scales == nullptr ||
        gate_up == nullptr || qi == nullptr || intermediate_block_scales == nullptr ||
        routed_down == nullptr || gemm_workspace == nullptr || gemm_workspace_size == 0 ||
        gemm_workspace_size > static_cast<size_t>(INT64_MAX)) {
        return false;
    }

    if (source_expert_ids != nullptr) {
        const int route_blocks = (routes + 255) / 256;
        prepare_scaled_routes<<<route_blocks, 256, 0, stream>>>(
            source_expert_ids, router_weights, expert_scale, prepared_expert_ids,
            prepared_router_weights, routes, e);
        if (cudaPeekAtLastError() != cudaSuccess) {
            return false;
        }
        expert_ids = prepared_expert_ids;
        router_weights = prepared_router_weights;
    }

    last_error_stage = 3;
    const size_t input_scale_bytes =
        static_cast<size_t>(e) * scale_group_m * input_scale_cols;
    const size_t intermediate_scale_bytes =
        static_cast<size_t>(e) * scale_group_m * intermediate_scale_cols;
    if (cudaMemsetAsync(input_block_scales, 0, input_scale_bytes, stream) != cudaSuccess ||
        cudaMemsetAsync(
            intermediate_block_scales, 0, intermediate_scale_bytes, stream) != cudaSuccess) {
        return false;
    }
    if (cudaMemsetAsync(counts, 0, static_cast<size_t>(e) * sizeof(int32_t), stream) !=
        cudaSuccess) {
        return false;
    }
    const int route_blocks = (routes + 255) / 256;
    last_error_stage = 4;
    count_and_rank_routes<<<route_blocks, 256, 0, stream>>>(
        expert_ids, counts, route_rank, routes, e);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }
    last_error_stage = 5;
    prefix_and_place_routes<<<1, 256, 0, stream>>>(
        expert_ids, counts, indptr, route_rank, route_dest, routes, e, scale_group_m);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }

    last_error_stage = 6;
    quantize_and_route_input_ranked<OutputType><<<n, kQuantThreads, 0, stream>>>(
        static_cast<const OutputType*>(input), expert_ids, route_rank, route_dest, qx,
        input_block_scales, h, top_k, scale_group_m, input_block_cols, input_scale_cols);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }

    last_error_stage = 7;
    try {
        if (!launch_cutlass_grouped_gemm_mxfp8_variable(
                qx, input_block_scales, fc1_qdata, fc1_block_scales, gate_up, indptr, e,
                scale_group_m, 2 * i, h, out_dtype_code, gemm_workspace,
                static_cast<int64_t>(gemm_workspace_size), stream)) {
            return false;
        }
    } catch (...) {
        return false;
    }

    last_error_stage = 8;
    activate_and_quantize_intermediate<OutputType><<<routes, kQuantThreads, 0, stream>>>(
        gate_up, expert_ids, route_rank, route_dest, qi, intermediate_block_scales, i,
        scale_group_m, intermediate_block_cols, intermediate_scale_cols);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }

    last_error_stage = 9;
    try {
        if (!launch_cutlass_grouped_gemm_mxfp8_variable(
                qi, intermediate_block_scales, fc2_qdata, fc2_block_scales, routed_down,
                indptr, e, scale_group_m, h, i, out_dtype_code, gemm_workspace,
                static_cast<int64_t>(gemm_workspace_size), stream)) {
            return false;
        }
    } catch (...) {
        return false;
    }

    const int64_t output_elements = static_cast<int64_t>(n) * h;
    const int reduction_blocks = static_cast<int>((output_elements + 255) / 256);
    last_error_stage = 10;
    weighted_route_reduction<OutputType><<<reduction_blocks, 256, 0, stream>>>(
        routed_down, route_dest, router_weights, static_cast<OutputType*>(output), n, h,
        top_k);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }
    last_error_stage = 0;
    return true;
}

#endif  // CUDA_VERSION >= 12080

}  // namespace fused_moe_mxfp8
}  // namespace comfy

extern "C" bool launch_cutlass_fused_moe_mxfp8(
    const void* input,
    const int32_t* expert_ids,
    const float* router_weights,
    const void* fc1_qdata,
    const void* fc1_block_scales,
    const void* fc2_qdata,
    const void* fc2_block_scales,
    void* output,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_experts,
    int64_t top_k,
    int input_dtype_code,
    void* workspace_ptr,
    int64_t workspace_size,
    cudaStream_t stream) {
#if CUDA_VERSION >= 12080
    using namespace comfy::fused_moe_mxfp8;
    last_error_stage = 1;
    if (input == nullptr || expert_ids == nullptr || router_weights == nullptr ||
        fc1_qdata == nullptr || fc1_block_scales == nullptr || fc2_qdata == nullptr ||
        fc2_block_scales == nullptr || output == nullptr || workspace_ptr == nullptr) {
        return false;
    }
    if ((num_tokens != 256 && num_tokens != 340) || hidden_size != 2816 ||
        intermediate_size != 704 || num_experts != 128 || top_k != kTopK ||
        workspace_size <= 0 || input_dtype_code < 1 || input_dtype_code > 2 ||
        hidden_size % kBlockSize != 0 || intermediate_size % kBlockSize != 0) {
        return false;
    }
    if (!is_aligned(input, 2) || !is_aligned(expert_ids, 4) ||
        !is_aligned(router_weights, 4) || !is_aligned(fc1_qdata, 16) ||
        !is_aligned(fc1_block_scales, 16) || !is_aligned(fc2_qdata, 16) ||
        !is_aligned(fc2_block_scales, 16) || !is_aligned(output, 2) ||
        !is_aligned(workspace_ptr, 16)) {
        return false;
    }

    if (input_dtype_code == 1) {
        return run_fused_moe_mxfp8<__half>(
            input, expert_ids, router_weights, nullptr, nullptr, fc1_qdata, fc1_block_scales, fc2_qdata,
            fc2_block_scales, output, static_cast<int>(num_tokens),
            static_cast<int>(hidden_size), static_cast<int>(intermediate_size),
            static_cast<int>(num_experts), static_cast<int>(top_k), workspace_ptr,
            static_cast<size_t>(workspace_size), input_dtype_code, stream);
    }
    return run_fused_moe_mxfp8<__nv_bfloat16>(
        input, expert_ids, router_weights, nullptr, nullptr, fc1_qdata, fc1_block_scales, fc2_qdata,
        fc2_block_scales, output, static_cast<int>(num_tokens),
        static_cast<int>(hidden_size), static_cast<int>(intermediate_size),
        static_cast<int>(num_experts), static_cast<int>(top_k), workspace_ptr,
        static_cast<size_t>(workspace_size), input_dtype_code, stream);
#else
    (void)input;
    (void)expert_ids;
    (void)router_weights;
    (void)fc1_qdata;
    (void)fc1_block_scales;
    (void)fc2_qdata;
    (void)fc2_block_scales;
    (void)output;
    (void)num_tokens;
    (void)hidden_size;
    (void)intermediate_size;
    (void)num_experts;
    (void)top_k;
    (void)input_dtype_code;
    (void)workspace_ptr;
    (void)workspace_size;
    (void)stream;
    return false;
#endif
}

extern "C" bool launch_cutlass_fused_moe_mxfp8_scaled(
    const void* input,
    const int64_t* expert_ids,
    const float* normalized_router_weights,
    const void* expert_scale_bf16,
    const void* fc1_qdata,
    const void* fc1_block_scales,
    const void* fc2_qdata,
    const void* fc2_block_scales,
    void* output,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_experts,
    int64_t top_k,
    int input_dtype_code,
    void* workspace_ptr,
    int64_t workspace_size,
    cudaStream_t stream) {
#if CUDA_VERSION >= 12080
    using namespace comfy::fused_moe_mxfp8;
    last_error_stage = 1;
    if (input == nullptr || expert_ids == nullptr || normalized_router_weights == nullptr ||
        expert_scale_bf16 == nullptr || fc1_qdata == nullptr ||
        fc1_block_scales == nullptr || fc2_qdata == nullptr ||
        fc2_block_scales == nullptr || output == nullptr || workspace_ptr == nullptr) {
        return false;
    }
    if ((num_tokens != 256 && num_tokens != 340) || hidden_size != 2816 ||
        intermediate_size != 704 || num_experts != 128 || top_k != kTopK ||
        workspace_size <= 0 || input_dtype_code < 1 || input_dtype_code > 2 ||
        hidden_size % kBlockSize != 0 || intermediate_size % kBlockSize != 0) {
        return false;
    }
    if (!is_aligned(input, 2) || !is_aligned(expert_ids, 8) ||
        !is_aligned(normalized_router_weights, 4) || !is_aligned(expert_scale_bf16, 2) ||
        !is_aligned(fc1_qdata, 16) || !is_aligned(fc1_block_scales, 16) ||
        !is_aligned(fc2_qdata, 16) || !is_aligned(fc2_block_scales, 16) ||
        !is_aligned(output, 2) || !is_aligned(workspace_ptr, 16)) {
        return false;
    }

    const auto* expert_scale = static_cast<const __nv_bfloat16*>(expert_scale_bf16);
    if (input_dtype_code == 1) {
        return run_fused_moe_mxfp8<__half>(
            input, nullptr, normalized_router_weights, expert_ids, expert_scale, fc1_qdata,
            fc1_block_scales, fc2_qdata, fc2_block_scales, output,
            static_cast<int>(num_tokens), static_cast<int>(hidden_size),
            static_cast<int>(intermediate_size), static_cast<int>(num_experts),
            static_cast<int>(top_k), workspace_ptr, static_cast<size_t>(workspace_size),
            input_dtype_code, stream);
    }
    return run_fused_moe_mxfp8<__nv_bfloat16>(
        input, nullptr, normalized_router_weights, expert_ids, expert_scale, fc1_qdata,
        fc1_block_scales, fc2_qdata, fc2_block_scales, output,
        static_cast<int>(num_tokens), static_cast<int>(hidden_size),
        static_cast<int>(intermediate_size), static_cast<int>(num_experts),
        static_cast<int>(top_k), workspace_ptr, static_cast<size_t>(workspace_size),
        input_dtype_code, stream);
#else
    (void)input;
    (void)expert_ids;
    (void)normalized_router_weights;
    (void)expert_scale_bf16;
    (void)fc1_qdata;
    (void)fc1_block_scales;
    (void)fc2_qdata;
    (void)fc2_block_scales;
    (void)output;
    (void)num_tokens;
    (void)hidden_size;
    (void)intermediate_size;
    (void)num_experts;
    (void)top_k;
    (void)input_dtype_code;
    (void)workspace_ptr;
    (void)workspace_size;
    (void)stream;
    return false;
#endif
}

extern "C" int cutlass_fused_moe_mxfp8_last_error_stage() {
    return comfy::fused_moe_mxfp8::last_error_stage;
}
