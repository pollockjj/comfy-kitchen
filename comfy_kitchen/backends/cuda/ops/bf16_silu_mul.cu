// SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

__global__ void bf16_silu_mul_kernel(
    const nv_bfloat16* __restrict__ gate,
    const nv_bfloat16* __restrict__ up,
    nv_bfloat16* __restrict__ out,
    int64_t numel)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }

    const float gate_f = __bfloat162float(gate[index]);
    const float silu_f = gate_f / (1.0f + expf(-gate_f));
    const nv_bfloat16 silu_bf16 = __float2bfloat16_rn(silu_f);
    const float product = __bfloat162float(silu_bf16) * __bfloat162float(up[index]);
    out[index] = __float2bfloat16_rn(product);
}

}  // namespace

extern "C" void launch_bf16_silu_mul_kernel(
    const void* gate,
    const void* up,
    void* out,
    int64_t numel,
    cudaStream_t stream)
{
    constexpr int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    bf16_silu_mul_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const nv_bfloat16*>(gate),
        static_cast<const nv_bfloat16*>(up),
        static_cast<nv_bfloat16*>(out),
        numel);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("bf16_silu_mul kernel launch failed: ") + cudaGetErrorString(error));
    }
}
