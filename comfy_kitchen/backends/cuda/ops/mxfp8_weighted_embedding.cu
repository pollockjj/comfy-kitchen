/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include "float_utils.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace comfy {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 64;
constexpr int kSharedStrideA = kTileK + 8;
constexpr int kSharedStrideB = kTileN + 8;
constexpr int kWarpM = 32;
constexpr int kWarpN = 32;
constexpr int kWmma = 16;
constexpr int kThreads = 512;
constexpr int kAccumulatorRows = kWarpM / kWmma;
constexpr int kAccumulatorCols = kWarpN / kWmma;
constexpr int kFp8ValuesPerVector = 16;
constexpr int kFp8VectorsPerRow = kTileN / kFp8ValuesPerVector;

using Accumulator = wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float>;
using FragmentA = wmma::fragment<
    wmma::matrix_a, kWmma, kWmma, kWmma, __nv_bfloat16, wmma::row_major>;
using FragmentB = wmma::fragment<
    wmma::matrix_b, kWmma, kWmma, kWmma, __nv_bfloat16, wmma::row_major>;

__global__ void mxfp8_weighted_embedding_kernel(
    const __nv_fp8_e4m3* __restrict__ qweight,
    const uint8_t* __restrict__ block_scales,
    const __nv_bfloat16* __restrict__ weights,
    float* __restrict__ partials,
    int64_t m,
    int64_t k,
    int64_t n,
    int split_k)
{
    extern __shared__ __align__(32) unsigned char shared_bytes[];
    auto* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_bytes);
    auto* shared_b = shared_a + kTileM * kSharedStrideA;
    auto* shared_scales = reinterpret_cast<uint32_t*>(
        shared_b + kTileK * kSharedStrideB);

    const int tile_n = static_cast<int>(blockIdx.x);
    const int tile_m = static_cast<int>(blockIdx.y);
    const int partition = static_cast<int>(blockIdx.z);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int64_t m_base = static_cast<int64_t>(tile_m) * kTileM;
    const int64_t n_base = static_cast<int64_t>(tile_n) * kTileN;

    const int64_t partition_size =
        ((k + split_k - 1) / split_k + kTileK - 1) / kTileK * kTileK;
    const int64_t k_begin = static_cast<int64_t>(partition) * partition_size;
    const int64_t k_end = min(k, k_begin + partition_size);

    Accumulator accumulators[kAccumulatorRows][kAccumulatorCols];
#pragma unroll
    for (int row = 0; row < kAccumulatorRows; ++row) {
#pragma unroll
        for (int col = 0; col < kAccumulatorCols; ++col) {
            wmma::fill_fragment(accumulators[row][col], 0.0f);
        }
    }

    const uint32_t scale_cols = static_cast<uint32_t>(n / 32);
    for (int64_t k_base = k_begin; k_base < k_end; k_base += kTileK) {
        for (int index = static_cast<int>(threadIdx.x);
             index < kTileM * kTileK;
             index += kThreads) {
            const int local_m = index / kTileK;
            const int local_k = index % kTileK;
            shared_a[local_m * kSharedStrideA + local_k] =
                weights[(m_base + local_m) * k + k_base + local_k];
        }

        if (threadIdx.x < kTileK) {
            const int64_t global_k = k_base + threadIdx.x;
            const size_t scale_offset = scale_factor_swizzled_offset(
                static_cast<size_t>(global_k),
                static_cast<size_t>(n_base / 32),
                scale_cols);
            shared_scales[threadIdx.x] =
                *reinterpret_cast<const uint32_t*>(block_scales + scale_offset);
        }
        __syncthreads();

        for (int vector_index = static_cast<int>(threadIdx.x);
             vector_index < kTileK * kFp8VectorsPerRow;
             vector_index += kThreads) {
            const int local_k = vector_index / kFp8VectorsPerRow;
            const int local_n =
                (vector_index % kFp8VectorsPerRow) * kFp8ValuesPerVector;
            const int64_t global_k = k_base + local_k;
            const int64_t global_n = n_base + local_n;
            const uint4 packed_qweight = *reinterpret_cast<const uint4*>(
                qweight + global_k * n + global_n);
            const uint32_t packed_scales = shared_scales[local_k];
            const uint8_t exponent = static_cast<uint8_t>(
                packed_scales >> ((local_n / 32) * 8));
            const uint32_t scale_bits =
                exponent == 0 ? 0 : static_cast<uint32_t>(exponent) << 23;
            const float scale = __uint_as_float(scale_bits);

            uint4 packed_bf16[2];
            const auto* fp8_bytes = reinterpret_cast<const uint8_t*>(&packed_qweight);
            auto* bf16_values = reinterpret_cast<__nv_bfloat16*>(packed_bf16);
#pragma unroll
            for (int value_index = 0;
                 value_index < kFp8ValuesPerVector;
                 ++value_index) {
                __nv_fp8_e4m3 fp8_value;
                fp8_value.__x = fp8_bytes[value_index];
                const float value = static_cast<float>(fp8_value) * scale;
                bf16_values[value_index] = __float2bfloat16_rn(value);
            }
            auto* shared_vectors = reinterpret_cast<uint4*>(
                shared_b + local_k * kSharedStrideB + local_n);
            shared_vectors[0] = packed_bf16[0];
            shared_vectors[1] = packed_bf16[1];
        }
        __syncthreads();

#pragma unroll
        for (int local_k = 0; local_k < kTileK; local_k += kWmma) {
            FragmentA fragments_a[kAccumulatorRows];
            FragmentB fragments_b[kAccumulatorCols];
#pragma unroll
            for (int row = 0; row < kAccumulatorRows; ++row) {
                const int a_row = warp_m * kWarpM + row * kWmma;
                wmma::load_matrix_sync(
                    fragments_a[row],
                    shared_a + a_row * kSharedStrideA + local_k,
                    kSharedStrideA);
            }
#pragma unroll
            for (int col = 0; col < kAccumulatorCols; ++col) {
                const int b_col = warp_n * kWarpN + col * kWmma;
                wmma::load_matrix_sync(
                    fragments_b[col],
                    shared_b + local_k * kSharedStrideB + b_col,
                    kSharedStrideB);
            }
#pragma unroll
            for (int row = 0; row < kAccumulatorRows; ++row) {
#pragma unroll
                for (int col = 0; col < kAccumulatorCols; ++col) {
                    wmma::mma_sync(
                        accumulators[row][col], fragments_a[row], fragments_b[col],
                        accumulators[row][col]);
                }
            }
        }
        __syncthreads();
    }

    float* partition_output =
        partials + static_cast<int64_t>(partition) * m * n;
#pragma unroll
    for (int row = 0; row < kAccumulatorRows; ++row) {
#pragma unroll
        for (int col = 0; col < kAccumulatorCols; ++col) {
            const int64_t output_row = m_base + warp_m * kWarpM + row * kWmma;
            const int64_t output_col = n_base + warp_n * kWarpN + col * kWmma;
            wmma::store_matrix_sync(
                partition_output + output_row * n + output_col,
                accumulators[row][col], n, wmma::mem_row_major);
        }
    }
}

__global__ void reduce_split_k_kernel(
    const float* __restrict__ partials,
    float* __restrict__ output,
    int64_t elements,
    int split_k)
{
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float sum = partials[index];
        for (int partition = 1; partition < split_k; ++partition) {
            sum = __fadd_rn(partials[static_cast<int64_t>(partition) * elements + index], sum);
        }
        output[index] = sum;
    }
}

}  // namespace
}  // namespace comfy

extern "C" void launch_mxfp8_weighted_embedding_kernel(
    const void* qweight,
    const void* block_scales,
    const void* weights,
    float* partials,
    float* output,
    int64_t m,
    int64_t k,
    int64_t n,
    int split_k,
    cudaStream_t stream)
{
    if (m <= 0 || k <= 0 || n <= 0 || m % 128 || k % 64 || n % 128 || split_k <= 0) {
        throw std::runtime_error("invalid mxfp8_weighted_embedding launch shape");
    }
    const dim3 grid(
        static_cast<unsigned>(n / comfy::kTileN),
        static_cast<unsigned>(m / comfy::kTileM),
        static_cast<unsigned>(split_k));
    constexpr size_t shared_bytes =
        (comfy::kTileM * comfy::kSharedStrideA
         + comfy::kTileK * comfy::kSharedStrideB)
        * sizeof(__nv_bfloat16) + comfy::kTileK * sizeof(uint32_t);
    comfy::mxfp8_weighted_embedding_kernel<<<grid, comfy::kThreads, shared_bytes, stream>>>(
        static_cast<const __nv_fp8_e4m3*>(qweight),
        static_cast<const uint8_t*>(block_scales),
        static_cast<const __nv_bfloat16*>(weights),
        partials,
        m,
        k,
        n,
        split_k);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("mxfp8_weighted_embedding kernel launch failed: ")
            + cudaGetErrorString(error));
    }

    if (split_k > 1) {
        const int64_t elements = m * n;
        constexpr int threads = 256;
        const int blocks = static_cast<int>(std::min<int64_t>((elements + threads - 1) / threads, 4096));
        comfy::reduce_split_k_kernel<<<blocks, threads, 0, stream>>>(
            partials, output, elements, split_k);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("mxfp8_weighted_embedding reduction launch failed: ")
                + cudaGetErrorString(error));
        }
    }
}
