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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifdef COMFY_HAVE_CUTLASS
#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"
#endif

namespace comfy {
namespace dg_mxfp8_probe {

#if defined(COMFY_HAVE_CUTLASS) && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

// TRTLLM-Gen's Apache-licensed launcher and generated cubin manifest identify
// m128n8k32 as the narrow MXFP8 MMA tile.  The generated mainloop itself is not
// distributed as source.  This one-warp probe exercises the corresponding
// public CUTLASS/CuTe SM120 m16n8k32 block-scale instruction directly, without
// importing FlashInfer or a TRTLLM runtime dependency.
__global__ void m16n8k32_probe(
    const uint8_t* weights,
    const uint8_t* weight_scales,
    const uint8_t* activations,
    const uint8_t* activation_scales,
    __nv_bfloat16* output) {
    using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
        cutlass::float_e4m3_t,
        cutlass::float_e4m3_t,
        float,
        cutlass::float_ue8m0_t,
        32>;

    const int lane = threadIdx.x;
    uint32_t a[4] = {};
    uint32_t b[2] = {};

#pragma unroll
    for (int value = 0; value < 16; ++value) {
        const int linear =
            (lane & 3) * 64 + (lane >> 2) + (value & 3) * 16 +
            ((value >> 2) & 1) * 8 + (value >> 3) * 256;
        const uint32_t byte = weights[linear];
        a[value >> 2] |= byte << ((value & 3) * 8);
    }
#pragma unroll
    for (int value = 0; value < 8; ++value) {
        const int linear =
            (lane & 3) * 32 + (lane >> 2) + (value & 3) * 8 +
            (value >> 2) * 128;
        const uint32_t byte = activations[linear];
        b[value >> 2] |= byte << ((value & 3) * 8);
    }

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const uint8_t scale_a = weight_scales[(lane & 1) * 8 + (lane >> 2)];
    const uint8_t scale_b = activation_scales[lane >> 2];
    Mma::fma(
        d0, d1, d2, d3,
        a[0], a[1], a[2], a[3],
        b[0], b[1],
        d0, d1, d2, d3,
        scale_a, scale_b);

    const float values[4] = {d0, d1, d2, d3};
#pragma unroll
    for (int value = 0; value < 4; ++value) {
        const int linear =
            (lane & 3) * 32 + (lane >> 2) + (value & 1) * 16 +
            (value >> 1) * 8;
        const int row = linear >> 3;
        const int col = linear & 7;
        output[col * 16 + row] = __float2bfloat16_rn(values[value]);
    }
}

#endif

}  // namespace dg_mxfp8_probe
}  // namespace comfy

extern "C" bool launch_dg_mxfp8_mma_probe(
    const void* weights,
    const uint8_t* weight_scales,
    const void* activations,
    const uint8_t* activation_scales,
    void* output,
    cudaStream_t stream) {
#if defined(COMFY_HAVE_CUTLASS) && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int device = 0;
    cudaDeviceProp props{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&props, device) != cudaSuccess ||
        props.major != 12 || props.minor != 0) {
        return false;
    }
    comfy::dg_mxfp8_probe::m16n8k32_probe<<<1, 32, 0, stream>>>(
        static_cast<const uint8_t*>(weights), weight_scales,
        static_cast<const uint8_t*>(activations), activation_scales,
        static_cast<__nv_bfloat16*>(output));
    return cudaPeekAtLastError() == cudaSuccess;
#else
    (void)weights;
    (void)weight_scales;
    (void)activations;
    (void)activation_scales;
    (void)output;
    (void)stream;
    return false;
#endif
}
