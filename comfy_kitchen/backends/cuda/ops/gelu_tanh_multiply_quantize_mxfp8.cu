/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utils.cuh"
#include "float_utils.cuh"
#include "dtype_dispatch.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace comfy {

constexpr unsigned int kGeluQuantValsPerThread = 8;
constexpr unsigned int kGeluQuantBlockSize = 32;
constexpr unsigned int kGeluQuantThreadsPerGroup =
    kGeluQuantBlockSize / kGeluQuantValsPerThread;

namespace {

template <typename IType>
__device__ __forceinline__ IType round_to_lowp(float value);

template <>
__device__ __forceinline__ __half round_to_lowp<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 round_to_lowp<__nv_bfloat16>(
    float value) {
    return __float2bfloat16_rn(value);
}

template <typename IType>
__device__ __forceinline__ IType gelu_tanh_multiply(IType gate, IType up) {
    const float x = static_cast<float>(gate);
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    constexpr float kCubicCoefficient = 0.044715f;
    const float x_cube = x * x * x;
    const float inner =
        kSqrtTwoOverPi * (x + kCubicCoefficient * x_cube);
    const float gelu = 0.5f * x * (1.0f + tanhf(inner));
    const IType rounded_gelu = round_to_lowp<IType>(gelu);
    return round_to_lowp<IType>(
        static_cast<float>(rounded_gelu) * static_cast<float>(up));
}

template <typename IType, bool Misaligned>
__global__ void gelu_tanh_multiply_quantize_mxfp8_kernel(
    const IType* __restrict__ gate,
    const IType* __restrict__ up,
    __nv_fp8_e4m3* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    const size_t num_cols,
    const size_t num_rows,
    const size_t orig_rows,
    const size_t orig_cols) {

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t global_elem_idx =
        static_cast<size_t>(idx) * kGeluQuantValsPerThread;
    if (global_elem_idx >= num_rows * num_cols) return;

    IType vals[kGeluQuantValsPerThread];
    if constexpr (Misaligned) {
#pragma unroll
        for (int i = 0; i < kGeluQuantValsPerThread; i++) {
            const size_t elem_idx = global_elem_idx + i;
            const size_t row = elem_idx / num_cols;
            const size_t col = elem_idx % num_cols;
            if (row < orig_rows && col < orig_cols) {
                const size_t input_idx = row * orig_cols + col;
                vals[i] = gelu_tanh_multiply(gate[input_idx], up[input_idx]);
            } else {
                vals[i] = static_cast<IType>(0.0f);
            }
        }
    } else {
        IType gate_vals[kGeluQuantValsPerThread];
        IType up_vals[kGeluQuantValsPerThread];
        *reinterpret_cast<float4*>(gate_vals) =
            *reinterpret_cast<const float4*>(gate + global_elem_idx);
        *reinterpret_cast<float4*>(up_vals) =
            *reinterpret_cast<const float4*>(up + global_elem_idx);
#pragma unroll
        for (int i = 0; i < kGeluQuantValsPerThread; i++) {
            vals[i] = gelu_tanh_multiply(gate_vals[i], up_vals[i]);
        }
    }

    IType absmax = __habs(vals[0]);
#pragma unroll
    for (int i = 1; i < kGeluQuantValsPerThread; i++) {
        absmax = __hmax(absmax, __habs(vals[i]));
    }
    constexpr unsigned int mask = 0xffffffff;
#pragma unroll
    for (int offset = kGeluQuantThreadsPerGroup / 2; offset >= 1; offset /= 2) {
        const IType other = __shfl_xor_sync(mask, absmax, offset);
        absmax = __hmax(absmax, other);
    }

    constexpr float fp8_max = 448.0f;
    // Match the existing quantize_mxfp8 kernel, which is compiled with
    // --use_fast_math and therefore uses the CUDA fast divide intrinsic.
    const float ratio = __fdividef(static_cast<float>(absmax), fp8_max);
    const __nv_fp8_storage_t e8m0_val =
        __nv_cvt_float_to_e8m0(ratio, __NV_SATFINITE, cudaRoundPosInf);
    if ((threadIdx.x % kGeluQuantThreadsPerGroup) == 0) {
        const size_t block_linear_idx = global_elem_idx / kGeluQuantBlockSize;
        const size_t num_blocks_per_row = num_cols / kGeluQuantBlockSize;
        const size_t row_idx = block_linear_idx / num_blocks_per_row;
        const size_t col_idx = block_linear_idx % num_blocks_per_row;
        const size_t scale_offset =
            scale_factor_swizzled_offset(row_idx, col_idx, num_blocks_per_row);
        block_scales[scale_offset] = e8m0_val;
    }

    const uint32_t encode_bits = static_cast<uint32_t>(254 - e8m0_val) << 23;
    const float encode_scale = __uint_as_float(encode_bits);
    __nv_fp8_e4m3 vals_output[kGeluQuantValsPerThread];
#pragma unroll
    for (int i = 0; i < kGeluQuantValsPerThread; i++) {
        float val_scaled = static_cast<float>(vals[i]) * encode_scale;
        val_scaled = fminf(fmaxf(val_scaled, -fp8_max), fp8_max);
        vals_output[i] = static_cast<__nv_fp8_e4m3>(val_scaled);
    }
    *reinterpret_cast<float2*>(output + global_elem_idx) =
        *reinterpret_cast<float2*>(vals_output);
}

}  // namespace
}  // namespace comfy

extern "C" {

void launch_gelu_tanh_multiply_quantize_mxfp8_kernel(
    const void* gate,
    const void* up,
    void* output,
    void* block_scales,
    int64_t num_rows,
    int64_t num_cols,
    int64_t orig_rows,
    int64_t orig_cols,
    int input_dtype_code,
    cudaStream_t stream) {

    if (num_rows == 0 || num_cols == 0) return;
    if (num_rows % comfy::kGeluQuantBlockSize != 0 ||
        num_cols % comfy::kGeluQuantBlockSize != 0) {
        throw std::runtime_error(
            "num_rows and num_cols must be divisible by 32 for MXFP8 block quantization");
    }

    const bool misaligned = (orig_rows != num_rows) || (orig_cols != num_cols);
    const int64_t numel = num_rows * num_cols;
    constexpr int threads_per_block = 128;
    const int64_t total_threads_needed = numel / comfy::kGeluQuantValsPerThread;
    const int blocks = static_cast<int>(
        (total_threads_needed + threads_per_block - 1) / threads_per_block);

    DISPATCH_HALF_DTYPE(input_dtype_code, InputType, [&] {
        if (misaligned) {
            comfy::gelu_tanh_multiply_quantize_mxfp8_kernel<InputType, true>
                <<<blocks, threads_per_block, 0, stream>>>(
                    static_cast<const InputType*>(gate),
                    static_cast<const InputType*>(up),
                    static_cast<__nv_fp8_e4m3*>(output),
                    static_cast<uint8_t*>(block_scales),
                    num_cols, num_rows, orig_rows, orig_cols);
        } else {
            comfy::gelu_tanh_multiply_quantize_mxfp8_kernel<InputType, false>
                <<<blocks, threads_per_block, 0, stream>>>(
                    static_cast<const InputType*>(gate),
                    static_cast<const InputType*>(up),
                    static_cast<__nv_fp8_e4m3*>(output),
                    static_cast<uint8_t*>(block_scales),
                    num_cols, num_rows, orig_rows, orig_cols);
        }
    });

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

}  // extern "C"
