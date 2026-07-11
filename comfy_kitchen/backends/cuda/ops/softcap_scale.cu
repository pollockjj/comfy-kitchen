/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace comfy {
namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 65535;

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
