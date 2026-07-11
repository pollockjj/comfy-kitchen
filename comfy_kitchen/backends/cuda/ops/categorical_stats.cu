/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace comfy {
namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;

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
            : MaxPair{-CUDART_INF_F, INT64_MAX};
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

__global__ void categorical_logsumexp_argmax_kernel(
    const float* __restrict__ logits,
    float* __restrict__ row_stats,
    int64_t* __restrict__ argmax,
    int64_t vocab_size)
{
    const int64_t row = blockIdx.x;
    const float* row_logits = logits + row * vocab_size;
    __shared__ float warp_values[kWarps];
    __shared__ int64_t warp_indices[kWarps];

    MaxPair local{-CUDART_INF_F, INT64_MAX};
    for (int64_t col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        local = better_pair(local, MaxPair{row_logits[col], col});
    }
    const MaxPair maximum = block_reduce_max_pair(local, warp_values, warp_indices);

    float exponential_sum = 0.0f;
    for (int64_t col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        exponential_sum += expf(row_logits[col] - maximum.value);
    }
    exponential_sum = block_reduce_sum(exponential_sum, warp_values);

    if (threadIdx.x == 0) {
        const float log_normalizer = logf(exponential_sum) + maximum.value;
        row_stats[row * 2] = log_normalizer;
        row_stats[row * 2 + 1] = maximum.value - log_normalizer;
        argmax[row] = maximum.index;
    }
}

__global__ void categorical_probs_entropy_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ row_stats,
    float* __restrict__ probs,
    float* __restrict__ entropy,
    int64_t vocab_size)
{
    const int64_t row = blockIdx.x;
    const float* row_logits = logits + row * vocab_size;
    float* row_probs = probs + row * vocab_size;
    const float log_normalizer = row_stats[row * 2];
    const float normalized_max = row_stats[row * 2 + 1];
    __shared__ float warp_values[kWarps];

    float softmax_sum = 0.0f;
    for (int64_t col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        const float normalized = row_logits[col] - log_normalizer;
        softmax_sum += expf(normalized - normalized_max);
    }
    softmax_sum = block_reduce_sum(softmax_sum, warp_values);

    float entropy_sum = 0.0f;
    for (int64_t col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        const float normalized = row_logits[col] - log_normalizer;
        const float probability = expf(normalized - normalized_max) / softmax_sum;
        row_probs[col] = probability;
        entropy_sum += fmaxf(normalized, -FLT_MAX) * probability;
    }
    entropy_sum = block_reduce_sum(entropy_sum, warp_values);
    if (threadIdx.x == 0) entropy[row] = -entropy_sum;
}

}  // namespace
}  // namespace comfy

extern "C" void launch_categorical_stats_kernel(
    const float* logits,
    float* probs,
    float* entropy,
    int64_t* argmax,
    float* row_stats,
    int64_t rows,
    int64_t vocab_size,
    cudaStream_t stream)
{
    comfy::categorical_logsumexp_argmax_kernel<<<static_cast<unsigned>(rows), comfy::kThreads, 0, stream>>>(
        logits, row_stats, argmax, vocab_size);
    comfy::categorical_probs_entropy_kernel<<<static_cast<unsigned>(rows), comfy::kThreads, 0, stream>>>(
        logits, row_stats, probs, entropy, vocab_size);
}
