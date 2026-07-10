/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cub/block/block_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace comfy {
namespace gemma4_routing {

constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kThreads = 32;
constexpr int kItemsPerThread = kExperts / kThreads;
constexpr float kLog2E = 1.4426950408889634f;

__device__ __forceinline__ uint16_t descending_bf16_key(uint16_t bits) {
    const uint16_t ascending =
        bits ^ ((bits & 0x8000u) != 0 ? 0xffffu : 0x8000u);
    return static_cast<uint16_t>(~ascending);
}

__global__ void gemma4_fused_routing_kernel(
    const __nv_bfloat16* logits,
    const __nv_bfloat16* per_expert_scale,
    float* topk_weights,
    int32_t* topk_ids) {
    using BlockSort = cub::BlockRadixSort<uint32_t, kThreads, kItemsPerThread>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    __shared__ int32_t sorted_ids[kTopK];
    __shared__ float sorted_logits[kTopK];

    const int token = blockIdx.x;
    uint32_t packed[kItemsPerThread];
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        const int expert = threadIdx.x + item * kThreads;
        const uint16_t raw = reinterpret_cast<const uint16_t*>(logits)[
            token * kExperts + expert];
        packed[item] =
            (static_cast<uint32_t>(descending_bf16_key(raw)) << 16) |
            static_cast<uint16_t>(expert);
    }
    BlockSort(sort_storage).Sort(packed);

    if (threadIdx.x < 2) {
#pragma unroll
        for (int item = 0; item < kItemsPerThread; ++item) {
            const int rank = threadIdx.x * kItemsPerThread + item;
            const int32_t expert = static_cast<int32_t>(packed[item] & 0xffffu);
            sorted_ids[rank] = expert;
            sorted_logits[rank] = __bfloat162float(logits[token * kExperts + expert]);
        }
    }
    __syncwarp();

    float weight = threadIdx.x < kTopK
        ? exp2f((sorted_logits[threadIdx.x] - sorted_logits[0]) * kLog2E)
        : 0.0f;
    float denominator = weight;
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        denominator += __shfl_down_sync(0xffffffffu, denominator, offset);
    }
    denominator = __shfl_sync(0xffffffffu, denominator, 0);

    if (threadIdx.x < kTopK) {
        const int32_t expert = sorted_ids[threadIdx.x];
        const int offset = token * kTopK + threadIdx.x;
        const float expert_scale = __bfloat162float(per_expert_scale[expert]);
        topk_ids[offset] = expert;
        topk_weights[offset] = weight / denominator * expert_scale;
    }
}

}  // namespace gemma4_routing
}  // namespace comfy

extern "C" bool launch_gemma4_fused_routing(
    const void* logits_bf16,
    const void* per_expert_scale_bf16,
    float* topk_weights,
    int32_t* topk_ids,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t top_k,
    cudaStream_t stream) {
    using namespace comfy::gemma4_routing;
    if (logits_bf16 == nullptr || per_expert_scale_bf16 == nullptr ||
        topk_weights == nullptr || topk_ids == nullptr || num_tokens <= 0 ||
        num_experts != kExperts || top_k != kTopK) {
        return false;
    }
    gemma4_fused_routing_kernel<<<num_tokens, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits_bf16),
        static_cast<const __nv_bfloat16*>(per_expert_scale_bf16),
        topk_weights,
        topk_ids);
    return cudaPeekAtLastError() == cudaSuccess;
}
