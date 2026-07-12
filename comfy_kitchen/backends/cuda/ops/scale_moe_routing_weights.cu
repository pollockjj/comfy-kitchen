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
namespace moe_routing {

constexpr int kThreads = 256;

__global__ void scale_moe_routing_weights_kernel(
    const float* normalized_weights,
    const int64_t* expert_ids,
    const __nv_bfloat16* expert_scale,
    float* output,
    int64_t num_routes,
    int64_t num_experts) {
    for (int64_t route = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         route < num_routes;
         route += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t expert = expert_ids[route];
        output[route] = expert >= 0 && expert < num_experts
            ? __fmul_rn(normalized_weights[route], __bfloat162float(expert_scale[expert]))
            : __int_as_float(0x7fc00000);
    }
}

}  // namespace moe_routing
}  // namespace comfy

extern "C" void launch_scale_moe_routing_weights_kernel(
    const float* normalized_weights,
    const int64_t* expert_ids,
    const void* expert_scale_bf16,
    float* output,
    int64_t num_routes,
    int64_t num_experts,
    cudaStream_t stream) {
    const int64_t blocks = (num_routes + comfy::moe_routing::kThreads - 1) /
        comfy::moe_routing::kThreads;
    comfy::moe_routing::scale_moe_routing_weights_kernel<<<
        static_cast<unsigned int>(blocks), comfy::moe_routing::kThreads, 0, stream>>>(
        normalized_weights,
        expert_ids,
        static_cast<const __nv_bfloat16*>(expert_scale_bf16),
        output,
        num_routes,
        num_experts);
}
