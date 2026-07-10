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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace comfy {
namespace gemma4_routing {

constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kThreads = 32;

__global__ void gemma4_fused_routing_kernel(
    float* topk_weights,
    const int64_t* selected_ids,
    const __nv_bfloat16* per_expert_scale,
    int32_t* topk_ids) {
    const int token = blockIdx.x;
    const int offset = token * kTopK + threadIdx.x;
    float weight = threadIdx.x < kTopK ? topk_weights[offset] : 0.0f;
    float denominator = weight;
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        denominator += __shfl_down_sync(0xffffffffu, denominator, offset);
    }
    denominator = __shfl_sync(0xffffffffu, denominator, 0);

    if (threadIdx.x < kTopK) {
        const int32_t expert = static_cast<int32_t>(selected_ids[offset]);
        const float expert_scale = __bfloat162float(per_expert_scale[expert]);
        topk_ids[offset] = expert;
        topk_weights[offset] = weight / denominator * expert_scale;
    }
}

}  // namespace gemma4_routing
}  // namespace comfy

extern "C" bool launch_gemma4_fused_routing(
    float* topk_weights,
    const int64_t* selected_ids,
    const void* per_expert_scale_bf16,
    int32_t* topk_ids,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t top_k,
    cudaStream_t stream) {
    using namespace comfy::gemma4_routing;
    if (topk_weights == nullptr || selected_ids == nullptr ||
        per_expert_scale_bf16 == nullptr ||
        topk_ids == nullptr || num_tokens <= 0 ||
        num_experts != kExperts || top_k != kTopK) {
        return false;
    }
    gemma4_fused_routing_kernel<<<num_tokens, kThreads, 0, stream>>>(
        topk_weights,
        selected_ids,
        static_cast<const __nv_bfloat16*>(per_expert_scale_bf16),
        topk_ids);
    return cudaPeekAtLastError() == cudaSuccess;
}
