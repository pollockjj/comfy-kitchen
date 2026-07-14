/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>

void launch_quantize_int8_rowwise_convrot64_kernel(
    const void* input,
    void* output,
    void* scales,
    int64_t num_rows,
    int64_t num_cols,
    int group_size,
    int input_dtype_code,
    bool stochastic,
    uint64_t seed,
    cudaStream_t stream);

extern "C" size_t cutlass_grouped_int8_dequant_packed_workspace_size(
    int64_t groups,
    int64_t rows);

extern "C" bool launch_cutlass_grouped_int8_dequant_packed(
    const void* activations,
    const void* weights,
    const void* activation_scales,
    const void* weight_scales,
    const int32_t* expert_indptr,
    void* accumulator,
    void* output,
    int64_t groups,
    int64_t rows,
    int64_t n,
    int64_t k,
    void* workspace,
    size_t workspace_size,
    int out_dtype_code,
    cudaStream_t stream);

namespace comfy {
namespace fused_moe_int8_convrot {

thread_local int last_error_stage = 0;

constexpr int kRouteThreads = 256;
constexpr int kTopKMax = 8;

class WorkspaceArena {
public:
    WorkspaceArena(void* ptr, size_t size)
        : base_(static_cast<uint8_t*>(ptr)), size_(size) {}

    template <class T>
    T* allocate(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) {
            return nullptr;
        }
        return static_cast<T*>(allocate_bytes(count * sizeof(T)));
    }

    void* tail(size_t alignment) {
        const size_t aligned = align(offset_, alignment);
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
    static size_t align(size_t offset, size_t alignment) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
            offset > SIZE_MAX - (alignment - 1)) {
            return SIZE_MAX;
        }
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    void* allocate_bytes(size_t bytes) {
        const size_t aligned = align(offset_, 16);
        if (aligned > size_ || bytes > size_ - aligned) {
            return nullptr;
        }
        void* result = base_ + aligned;
        offset_ = aligned + bytes;
        return result;
    }

    uint8_t* base_;
    size_t size_;
    size_t offset_{0};
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
    const int32_t* counts,
    int32_t* indptr,
    const int32_t* route_rank,
    int32_t* route_dest,
    int routes,
    int num_experts) {
    if (threadIdx.x == 0) {
        int running = 0;
        for (int expert = 0; expert < num_experts; ++expert) {
            indptr[expert] = running;
            running += counts[expert];
        }
        indptr[num_experts] = running;
    }
    __syncthreads();

    for (int route = threadIdx.x; route < routes; route += blockDim.x) {
        const int expert = expert_ids[route];
        const int rank = route_rank[route];
        route_dest[route] = expert >= 0 && expert < num_experts && rank >= 0
            ? indptr[expert] + rank
            : -1;
    }
}

__global__ void replicate_quantized_routes(
    const int8_t* token_qdata,
    const float* token_scales,
    const int32_t* route_dest,
    int8_t* routed_qdata,
    float* routed_scales,
    int num_tokens,
    int hidden_size,
    int top_k) {
    const int token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }
    if (threadIdx.x < top_k) {
        const int route = token * top_k + threadIdx.x;
        const int dest = route_dest[route];
        if (dest >= 0) {
            routed_scales[dest] = token_scales[token];
        }
    }
    for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
        const int8_t value = token_qdata[static_cast<size_t>(token) * hidden_size + col];
#pragma unroll
        for (int position = 0; position < kTopKMax; ++position) {
            if (position >= top_k) {
                break;
            }
            const int dest = route_dest[token * top_k + position];
            if (dest >= 0) {
                routed_qdata[static_cast<size_t>(dest) * hidden_size + col] = value;
            }
        }
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

template <class T>
__device__ __forceinline__ T geglu_product(T gate, T up) {
    const float x = lowp_to_float(gate);
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    constexpr float kCubicCoefficient = 0.044715f;
    const float gelu =
        0.5f * x * (1.0f + tanhf(kSqrtTwoOverPi * (x + kCubicCoefficient * x * x * x)));
    const T rounded_gelu = float_to_lowp<T>(gelu);
    return float_to_lowp<T>(lowp_to_float(rounded_gelu) * lowp_to_float(up));
}

template <class T>
__global__ void activate_intermediate(
    const T* gate_up,
    T* intermediate,
    int routes,
    int intermediate_size) {
    const int route = blockIdx.x;
    if (route >= routes) {
        return;
    }
    const T* source = gate_up + static_cast<size_t>(route) * 2 * intermediate_size;
    T* destination = intermediate + static_cast<size_t>(route) * intermediate_size;
    for (int col = threadIdx.x; col < intermediate_size; col += blockDim.x) {
        destination[col] = geglu_product(source[col], source[intermediate_size + col]);
    }
}

template <class T>
__global__ void weighted_route_reduction(
    const T* routed_output,
    const int32_t* route_dest,
    const float* router_weights,
    T* output,
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
        bool invalid = false;
#pragma unroll
        for (int position = 0; position < kTopKMax; ++position) {
            if (position >= top_k) {
                break;
            }
            const int route = token * top_k + position;
            const int dest = route_dest[route];
            if (dest < 0) {
                invalid = true;
                continue;
            }
            const float value = lowp_to_float(
                routed_output[static_cast<size_t>(dest) * hidden_size + col]);
            sum = fmaf(router_weights[route], value, sum);
        }
        output[linear] = float_to_lowp<T>(invalid ? __int_as_float(0x7fc00000) : sum);
    }
}

template <class OutputType>
bool run_fused_moe_int8_convrot(
    const void* input,
    const int32_t* expert_ids,
    const float* router_weights,
    const int8_t* fc1_qdata,
    const float* fc1_scales,
    const int8_t* fc2_qdata,
    const float* fc2_scales,
    void* output,
    int n,
    int h,
    int i,
    int e,
    int top_k,
    int fc1_group_size,
    int fc2_group_size,
    int dtype_code,
    void* workspace_ptr,
    size_t workspace_size,
    cudaStream_t stream) {
    const int routes = n * top_k;

    last_error_stage = 2;
    WorkspaceArena persistent(workspace_ptr, workspace_size);
    int32_t* counts = persistent.allocate<int32_t>(e);
    int32_t* indptr = persistent.allocate<int32_t>(e + 1);
    int32_t* route_rank = persistent.allocate<int32_t>(routes);
    int32_t* route_dest = persistent.allocate<int32_t>(routes);
    void* scratch = persistent.tail(256);
    const size_t scratch_size = persistent.remaining();
    if (counts == nullptr || indptr == nullptr || route_rank == nullptr ||
        route_dest == nullptr || scratch == nullptr || scratch_size == 0) {
        return false;
    }

    if (cudaMemsetAsync(counts, 0, static_cast<size_t>(e) * sizeof(int32_t), stream) !=
        cudaSuccess) {
        return false;
    }
    const int route_blocks = (routes + kRouteThreads - 1) / kRouteThreads;
    last_error_stage = 3;
    count_and_rank_routes<<<route_blocks, kRouteThreads, 0, stream>>>(
        expert_ids, counts, route_rank, routes, e);
    prefix_and_place_routes<<<1, kRouteThreads, 0, stream>>>(
        expert_ids, counts, indptr, route_rank, route_dest, routes, e);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }

    last_error_stage = 4;
    WorkspaceArena stage1(scratch, scratch_size);
    int8_t* token_qdata = stage1.allocate<int8_t>(static_cast<size_t>(n) * h);
    float* token_scales = stage1.allocate<float>(n);
    int8_t* routed_qdata = stage1.allocate<int8_t>(static_cast<size_t>(routes) * h);
    float* routed_scales = stage1.allocate<float>(routes);
    int32_t* fc1_accumulator =
        stage1.allocate<int32_t>(static_cast<size_t>(routes) * 2 * i);
    OutputType* gate_up =
        stage1.allocate<OutputType>(static_cast<size_t>(routes) * 2 * i);
    void* fc1_workspace = stage1.tail(256);
    const size_t fc1_workspace_size = stage1.remaining();
    const size_t packed_workspace_size =
        cutlass_grouped_int8_dequant_packed_workspace_size(e, routes);
    if (token_qdata == nullptr || token_scales == nullptr || routed_qdata == nullptr ||
        routed_scales == nullptr || fc1_accumulator == nullptr || gate_up == nullptr ||
        fc1_workspace == nullptr || fc1_workspace_size < packed_workspace_size) {
        return false;
    }

    try {
        launch_quantize_int8_rowwise_convrot64_kernel(
            input, token_qdata, token_scales, n, h, fc1_group_size, dtype_code,
            false, 0, stream);
    } catch (...) {
        return false;
    }
    replicate_quantized_routes<<<n, kRouteThreads, 0, stream>>>(
        token_qdata, token_scales, route_dest, routed_qdata, routed_scales, n, h, top_k);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }

    last_error_stage = 5;
    if (!launch_cutlass_grouped_int8_dequant_packed(
            routed_qdata, fc1_qdata, routed_scales, fc1_scales, indptr,
            fc1_accumulator, gate_up, e, routes, 2 * i, h, fc1_workspace,
            fc1_workspace_size, dtype_code, stream)) {
        return false;
    }

    last_error_stage = 6;
    WorkspaceArena activation(scratch, scratch_size);
    OutputType* intermediate =
        activation.allocate<OutputType>(static_cast<size_t>(routes) * i);
    int8_t* intermediate_qdata =
        activation.allocate<int8_t>(static_cast<size_t>(routes) * i);
    float* intermediate_scales = activation.allocate<float>(routes);
    if (intermediate == nullptr || intermediate_qdata == nullptr ||
        intermediate_scales == nullptr) {
        return false;
    }
    activate_intermediate<OutputType><<<routes, kRouteThreads, 0, stream>>>(
        gate_up, intermediate, routes, i);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }
    try {
        launch_quantize_int8_rowwise_convrot64_kernel(
            intermediate, intermediate_qdata, intermediate_scales, routes, i,
            fc2_group_size, dtype_code, false, 0, stream);
    } catch (...) {
        return false;
    }

    last_error_stage = 7;
    WorkspaceArena stage2(scratch, scratch_size);
    if (stage2.allocate<OutputType>(static_cast<size_t>(routes) * i) != intermediate ||
        stage2.allocate<int8_t>(static_cast<size_t>(routes) * i) != intermediate_qdata ||
        stage2.allocate<float>(routes) != intermediate_scales) {
        return false;
    }
    int32_t* fc2_accumulator =
        stage2.allocate<int32_t>(static_cast<size_t>(routes) * h);
    OutputType* routed_down =
        stage2.allocate<OutputType>(static_cast<size_t>(routes) * h);
    void* fc2_workspace = stage2.tail(256);
    const size_t fc2_workspace_size = stage2.remaining();
    if (fc2_accumulator == nullptr || routed_down == nullptr || fc2_workspace == nullptr ||
        fc2_workspace_size < packed_workspace_size) {
        return false;
    }
    if (!launch_cutlass_grouped_int8_dequant_packed(
            intermediate_qdata, fc2_qdata, intermediate_scales, fc2_scales, indptr,
            fc2_accumulator, routed_down, e, routes, h, i, fc2_workspace,
            fc2_workspace_size, dtype_code, stream)) {
        return false;
    }

    const int64_t output_elements = static_cast<int64_t>(n) * h;
    const int reduction_blocks = static_cast<int>(
        (output_elements + kRouteThreads - 1) / kRouteThreads);
    last_error_stage = 8;
    weighted_route_reduction<OutputType><<<reduction_blocks, kRouteThreads, 0, stream>>>(
        routed_down, route_dest, router_weights, static_cast<OutputType*>(output), n, h,
        top_k);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }
    last_error_stage = 0;
    return true;
}

__host__ __forceinline__ bool is_aligned(const void* ptr, uintptr_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

}  // namespace fused_moe_int8_convrot
}  // namespace comfy

extern "C" bool launch_cutlass_fused_moe_int8_convrot(
    const void* input,
    const int32_t* expert_ids,
    const float* router_weights,
    const void* fc1_qdata,
    const float* fc1_scales,
    const void* fc2_qdata,
    const float* fc2_scales,
    void* output,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_experts,
    int64_t top_k,
    int fc1_group_size,
    int fc2_group_size,
    int dtype_code,
    void* workspace_ptr,
    int64_t workspace_size,
    cudaStream_t stream) {
    using namespace comfy::fused_moe_int8_convrot;
    last_error_stage = 1;
    if (input == nullptr || expert_ids == nullptr || router_weights == nullptr ||
        fc1_qdata == nullptr || fc1_scales == nullptr || fc2_qdata == nullptr ||
        fc2_scales == nullptr || output == nullptr || workspace_ptr == nullptr) {
        return false;
    }
    if (num_tokens <= 0 || hidden_size <= 0 || intermediate_size <= 0 ||
        num_experts <= 0 || top_k <= 0 || top_k > kTopKMax ||
        num_tokens > INT_MAX || hidden_size > INT_MAX || intermediate_size > INT_MAX ||
        num_experts > INT_MAX || num_tokens > INT_MAX / top_k ||
        (fc1_group_size != 64 && fc1_group_size != 256) ||
        (fc2_group_size != 64 && fc2_group_size != 256) ||
        hidden_size % fc1_group_size != 0 ||
        intermediate_size % fc2_group_size != 0 ||
        (dtype_code != 1 && dtype_code != 2) || workspace_size <= 0) {
        return false;
    }
    if (!is_aligned(input, 2) || !is_aligned(expert_ids, 4) ||
        !is_aligned(router_weights, 4) || !is_aligned(fc1_qdata, 16) ||
        !is_aligned(fc1_scales, 4) || !is_aligned(fc2_qdata, 16) ||
        !is_aligned(fc2_scales, 4) || !is_aligned(output, 2) ||
        !is_aligned(workspace_ptr, 16)) {
        return false;
    }

    if (dtype_code == 1) {
        return run_fused_moe_int8_convrot<__half>(
            input, expert_ids, router_weights, static_cast<const int8_t*>(fc1_qdata),
            fc1_scales, static_cast<const int8_t*>(fc2_qdata), fc2_scales, output,
            static_cast<int>(num_tokens), static_cast<int>(hidden_size),
            static_cast<int>(intermediate_size), static_cast<int>(num_experts),
            static_cast<int>(top_k), fc1_group_size, fc2_group_size, dtype_code,
            workspace_ptr, static_cast<size_t>(workspace_size), stream);
    }
    return run_fused_moe_int8_convrot<__nv_bfloat16>(
        input, expert_ids, router_weights, static_cast<const int8_t*>(fc1_qdata),
        fc1_scales, static_cast<const int8_t*>(fc2_qdata), fc2_scales, output,
        static_cast<int>(num_tokens), static_cast<int>(hidden_size),
        static_cast<int>(intermediate_size), static_cast<int>(num_experts),
        static_cast<int>(top_k), fc1_group_size, fc2_group_size, dtype_code,
        workspace_ptr, static_cast<size_t>(workspace_size), stream);
}

extern "C" int cutlass_fused_moe_int8_convrot_last_error_stage() {
    return comfy::fused_moe_int8_convrot::last_error_stage;
}
