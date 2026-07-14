/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace comfy {
namespace {

constexpr int kThreads = 512;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int64_t kMaxBlocks = 65535;

struct MaxPair {
    float value;
    int64_t index;
};

__device__ __forceinline__ MaxPair better_pair(MaxPair a, MaxPair b) {
    return (b.value > a.value || (b.value == a.value && b.index < a.index)) ? b : a;
}

__device__ __forceinline__ MaxPair warp_reduce_max_pair(MaxPair value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        MaxPair other{
            __shfl_down_sync(0xffffffffu, value.value, offset),
            __shfl_down_sync(0xffffffffu, value.index, offset),
        };
        value = better_pair(value, other);
    }
    return value;
}

__device__ __forceinline__ MaxPair block_reduce_max_pair(
    MaxPair value,
    float* warp_values,
    int64_t* warp_indices)
{
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_max_pair(value);
    if (lane == 0) {
        warp_values[warp] = value.value;
        warp_indices[warp] = value.index;
    }
    __syncthreads();

    if (warp == 0) {
        MaxPair block_value = lane < kWarps
            ? MaxPair{warp_values[lane], warp_indices[lane]}
            : MaxPair{-FLT_MAX, INT64_MAX};
        block_value = warp_reduce_max_pair(block_value);
        if (lane == 0) {
            warp_values[0] = block_value.value;
            warp_indices[0] = block_value.index;
        }
    }
    __syncthreads();
    return MaxPair{warp_values[0], warp_indices[0]};
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float* warp_values) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_sum(value);
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    if (warp == 0) {
        float block_value = lane < kWarps ? warp_values[lane] : 0.0f;
        block_value = warp_reduce_sum(block_value);
        if (lane == 0) warp_values[0] = block_value;
    }
    __syncthreads();
    return warp_values[0];
}

__global__ void softcap_scale_bf16_kernel(
    const __nv_bfloat16* __restrict__ raw_logits,
    float* __restrict__ output,
    int64_t numel,
    float cap,
    float inverse_temperature)
{
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < numel;
         index += stride) {
        const float value = __bfloat162float(raw_logits[index]);
        output[index] = tanhf(value * (1.0f / cap)) * cap * inverse_temperature;
    }
}

__global__ void softcap_categorical_stats_sample_bf16_kernel(
    const __nv_bfloat16* __restrict__ raw_logits,
    const float* __restrict__ exponential_noise,
    float* __restrict__ processed_logits,
    __nv_bfloat16* __restrict__ self_conditioning_logits,
    float* __restrict__ entropy,
    int64_t* __restrict__ argmax,
    int64_t* __restrict__ sample,
    int32_t* __restrict__ invalid,
    int64_t vocab_size,
    float cap,
    float inverse_temperature)
{
    const int64_t row = blockIdx.x;
    const int64_t row_offset = row * vocab_size;
    const __nv_bfloat16* row_raw = raw_logits + row_offset;
    const float* row_noise = exponential_noise + row_offset;
    float* row_processed = processed_logits + row_offset;
    __nv_bfloat16* row_self_conditioning = self_conditioning_logits + row_offset;
    __shared__ float warp_values[kWarps];
    __shared__ int64_t warp_indices[kWarps];
    __shared__ float log_normalizer;
    __shared__ float normalized_max;
    __shared__ float inverse_exponential_sum;

    MaxPair local_max{-FLT_MAX, INT64_MAX};
    for (int64_t col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        const float raw = __bfloat162float(row_raw[col]);
        const float processed = tanhf(raw * (1.0f / cap)) * cap * inverse_temperature;
        row_processed[col] = processed;
        row_self_conditioning[col] = __float2bfloat16_rn(processed);
        local_max = better_pair(local_max, MaxPair{processed, col});
        if (!isfinite(processed)) atomicExch(invalid, 1);
    }
    const MaxPair maximum = block_reduce_max_pair(local_max, warp_values, warp_indices);

    float exponential_sum = 0.0f;
    for (int64_t col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        exponential_sum += __expf(row_processed[col] - maximum.value);
    }
    exponential_sum = block_reduce_sum(exponential_sum, warp_values);
    if (threadIdx.x == 0) {
        log_normalizer = __logf(exponential_sum) + maximum.value;
        normalized_max = maximum.value - log_normalizer;
        inverse_exponential_sum = __fdividef(1.0f, exponential_sum);
        argmax[row] = maximum.index;
    }
    __syncthreads();

    float entropy_sum = 0.0f;
    MaxPair local_sample{-FLT_MAX, INT64_MAX};
    for (int64_t col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        const float normalized = row_processed[col] - log_normalizer;
        const float probability = __expf(normalized - normalized_max) * inverse_exponential_sum;
        const float noise = row_noise[col];
        if (!isfinite(noise) || noise <= 0.0f) {
            atomicExch(invalid, 1);
        } else {
            local_sample = better_pair(local_sample, MaxPair{__fdiv_rn(probability, noise), col});
        }
        entropy_sum += fmaxf(normalized, -FLT_MAX) * probability;
    }
    entropy_sum = block_reduce_sum(entropy_sum, warp_values);
    const MaxPair sampled = block_reduce_max_pair(local_sample, warp_values, warp_indices);
    if (threadIdx.x == 0) {
        entropy[row] = -entropy_sum;
        sample[row] = sampled.index;
    }
}

}  // namespace
}  // namespace comfy

extern "C" void launch_softcap_scale_kernel(
    const void* raw_logits,
    float* output,
    int64_t numel,
    float cap,
    float inverse_temperature,
    cudaStream_t stream)
{
    const int64_t required_blocks = (numel + comfy::kThreads - 1) / comfy::kThreads;
    const unsigned blocks = static_cast<unsigned>(
        required_blocks < comfy::kMaxBlocks ? required_blocks : comfy::kMaxBlocks);
    comfy::softcap_scale_bf16_kernel<<<blocks, comfy::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw_logits),
        output,
        numel,
        cap,
        inverse_temperature);
}

extern "C" void launch_softcap_categorical_stats_sample_kernel(
    const void* raw_logits,
    const float* exponential_noise,
    float* processed_logits,
    void* self_conditioning_logits,
    float* entropy,
    int64_t* argmax,
    int64_t* sample,
    int32_t* invalid,
    int64_t rows,
    int64_t vocab_size,
    float cap,
    float inverse_temperature,
    cudaStream_t stream)
{
    comfy::softcap_categorical_stats_sample_bf16_kernel<<<
        static_cast<unsigned>(rows), comfy::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw_logits),
        exponential_noise,
        processed_logits,
        static_cast<__nv_bfloat16*>(self_conditioning_logits),
        entropy,
        argmax,
        sample,
        invalid,
        vocab_size,
        cap,
        inverse_temperature);
}
