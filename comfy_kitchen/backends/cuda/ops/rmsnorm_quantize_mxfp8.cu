/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "float_utils.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace comfy {
namespace {

#if CUDART_VERSION >= 12080

constexpr int kHiddenSize = 2816;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kMxBlockSize = 32;
constexpr int kValuesPerThread = 8;
constexpr int kThreadsPerMxBlock = kMxBlockSize / kValuesPerThread;
constexpr int kMxBlockCols = kHiddenSize / kMxBlockSize;

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float* warp_sums) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    value = warp_reduce_sum(value);
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();

    if (warp == 0) {
        float block_value = lane < kWarps ? warp_sums[lane] : 0.0f;
        block_value = warp_reduce_sum(block_value);
        if (lane == 0) warp_sums[0] = block_value;
    }
    __syncthreads();
    return warp_sums[0];
}

__global__ void rmsnorm_quantize_mxfp8_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_fp8_e4m3* __restrict__ qdata,
    uint8_t* __restrict__ block_scales,
    float eps)
{
    const int row = blockIdx.x;
    const __nv_bfloat16* input_row = input + static_cast<size_t>(row) * kHiddenSize;
    __nv_fp8_e4m3* qdata_row = qdata + static_cast<size_t>(row) * kHiddenSize;
    __shared__ float warp_sums[kWarps];
    __shared__ float inverse_rms;

    float square_sum = 0.0f;
    for (int col = threadIdx.x; col < kHiddenSize; col += blockDim.x) {
        const float value = __bfloat162float(input_row[col]);
        square_sum = fmaf(value, value, square_sum);
    }
    square_sum = block_reduce_sum(square_sum, warp_sums);
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum * (1.0f / static_cast<float>(kHiddenSize)) + eps);
    }
    __syncthreads();

    constexpr int groups_per_block = kThreads / kThreadsPerMxBlock;
    const int group = threadIdx.x / kThreadsPerMxBlock;
    const int lane = threadIdx.x & (kThreadsPerMxBlock - 1);
    for (int block_col = group; block_col < kMxBlockCols; block_col += groups_per_block) {
        const int col = block_col * kMxBlockSize + lane * kValuesPerThread;
        __nv_bfloat16 normalized[kValuesPerThread];
        float block_absmax = 0.0f;
#pragma unroll
        for (int index = 0; index < kValuesPerThread; ++index) {
            const int element = col + index;
            const float value =
                __bfloat162float(input_row[element]) * inverse_rms *
                __bfloat162float(weight[element]);
            normalized[index] = __float2bfloat16_rn(value);
            block_absmax = fmaxf(block_absmax, fabsf(__bfloat162float(normalized[index])));
        }
#pragma unroll
        for (int offset = kThreadsPerMxBlock / 2; offset >= 1; offset /= 2) {
            block_absmax = fmaxf(
                block_absmax,
                __shfl_xor_sync(0xffffffffu, block_absmax, offset, kThreadsPerMxBlock));
        }

        constexpr float kFp8Max = 448.0f;
        const __nv_fp8_storage_t e8m0 = __nv_cvt_float_to_e8m0(
            block_absmax / kFp8Max, __NV_SATFINITE, cudaRoundPosInf);
        if (lane == 0) {
            block_scales[scale_factor_swizzled_offset(row, block_col, kMxBlockCols)] = e8m0;
        }
        const uint32_t encode_bits = static_cast<uint32_t>(254 - e8m0) << 23;
        const float encode_scale = __uint_as_float(encode_bits);

        union {
            uint64_t u64;
            __nv_fp8_e4m3 fp8[kValuesPerThread];
        } packed;
#pragma unroll
        for (int index = 0; index < kValuesPerThread; ++index) {
            float scaled = __bfloat162float(normalized[index]) * encode_scale;
            scaled = fminf(fmaxf(scaled, -kFp8Max), kFp8Max);
            packed.fp8[index] = static_cast<__nv_fp8_e4m3>(scaled);
        }
        *reinterpret_cast<uint64_t*>(qdata_row + col) = packed.u64;
    }
}

#endif  // CUDART_VERSION >= 12080

}  // namespace
}  // namespace comfy

extern "C" void launch_rmsnorm_quantize_mxfp8_kernel(
    const void* input,
    const void* weight,
    void* qdata,
    void* block_scales,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    cudaStream_t stream)
{
#if CUDART_VERSION >= 12080
    if (input == nullptr || weight == nullptr || qdata == nullptr || block_scales == nullptr) {
        throw std::runtime_error("rmsnorm_quantize_mxfp8 received a null pointer");
    }
    if ((rows != 256 && rows != 340) || hidden_size != comfy::kHiddenSize) {
        throw std::runtime_error("rmsnorm_quantize_mxfp8 requires [256|340, 2816]");
    }
    if (!(eps > 0.0f)) {
        throw std::runtime_error("rmsnorm_quantize_mxfp8 requires positive eps");
    }
    comfy::rmsnorm_quantize_mxfp8_bf16_kernel<<<
        static_cast<unsigned>(rows), comfy::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_fp8_e4m3*>(qdata),
        static_cast<uint8_t*>(block_scales),
        eps);
    const cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("rmsnorm_quantize_mxfp8 launch failed: ") + cudaGetErrorString(error));
    }
#else
    (void)input;
    (void)weight;
    (void)qdata;
    (void)block_scales;
    (void)rows;
    (void)hidden_size;
    (void)eps;
    (void)stream;
    throw std::runtime_error("rmsnorm_quantize_mxfp8 requires CUDA 12.8 or newer");
#endif
}
