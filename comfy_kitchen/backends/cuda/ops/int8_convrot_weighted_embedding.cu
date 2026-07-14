/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include "svdquant_utils.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace comfy {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int kReductionChunk = 65504;
constexpr int kTileM = 128;
constexpr int kTileN = 256;
constexpr int kTileK = 32;
constexpr int kSharedStrideA = kTileK + 8;
constexpr int kSharedStrideB = kTileN + 8;
constexpr int kWarpM = 64;
constexpr int kWarpN = 64;
constexpr int kWmma = 16;
constexpr int kThreads = 256;
constexpr int kAccumulatorRows = kWarpM / kWmma;
constexpr int kAccumulatorCols = kWarpN / kWmma;
constexpr int kBf16ValuesPerVector = sizeof(uint4) / sizeof(__nv_bfloat16);
constexpr int kAVectorsPerRow = kTileK / kBf16ValuesPerVector;

using Accumulator = wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float>;
using FragmentA = wmma::fragment<
    wmma::matrix_a, kWmma, kWmma, kWmma, __nv_bfloat16, wmma::row_major>;
using FragmentB = wmma::fragment<
    wmma::matrix_b, kWmma, kWmma, kWmma, __nv_bfloat16, wmma::row_major>;

__device__ __forceinline__ float h4_row_dot(
    int row, float x0, float x1, float x2, float x3)
{
    switch (row) {
        case 0: return x0 + x1 + x2 - x3;
        case 1: return x0 + x1 - x2 + x3;
        case 2: return x0 - x1 + x2 + x3;
        default: return -x0 + x1 + x2 + x3;
    }
}

__device__ __forceinline__ void load_convrot_row(
    const int8_t* __restrict__ qrow,
    float scale,
    __nv_bfloat16* __restrict__ destination)
{
    constexpr unsigned mask = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int half = lane >> 4;
    const int d2 = lane & 3;
    const int d3 = (lane >> 2) & 3;
    float values[8];
    float next[8];

#pragma unroll
    for (int d1_low = 0; d1_low < 2; ++d1_low) {
#pragma unroll
        for (int d0 = 0; d0 < 4; ++d0) {
            const int index = d0 + 4 * (d1_low + 2 * half) + 16 * d2 + 64 * d3;
            values[d1_low * 4 + d0] = static_cast<float>(qrow[index]) * scale;
        }
    }

#pragma unroll
    for (int d1_low = 0; d1_low < 2; ++d1_low) {
        const int offset = d1_low * 4;
        const float x0 = values[offset];
        const float x1 = values[offset + 1];
        const float x2 = values[offset + 2];
        const float x3 = values[offset + 3];
        next[offset] = 0.5f * h4_row_dot(0, x0, x1, x2, x3);
        next[offset + 1] = 0.5f * h4_row_dot(1, x0, x1, x2, x3);
        next[offset + 2] = 0.5f * h4_row_dot(2, x0, x1, x2, x3);
        next[offset + 3] = 0.5f * h4_row_dot(3, x0, x1, x2, x3);
    }

#pragma unroll
    for (int d0 = 0; d0 < 4; ++d0) {
        const float local0 = next[d0];
        const float local1 = next[4 + d0];
        const float remote0 = __shfl_xor_sync(mask, local0, 16);
        const float remote1 = __shfl_xor_sync(mask, local1, 16);
        const float x0 = half == 0 ? local0 : remote0;
        const float x1 = half == 0 ? local1 : remote1;
        const float x2 = half == 0 ? remote0 : local0;
        const float x3 = half == 0 ? remote1 : local1;
        values[d0] = 0.5f * h4_row_dot(half == 0 ? 0 : 2, x0, x1, x2, x3);
        values[4 + d0] = 0.5f * h4_row_dot(half == 0 ? 1 : 3, x0, x1, x2, x3);
    }

#pragma unroll
    for (int value = 0; value < 8; ++value) {
        const int lane_base = lane & ~3;
        const float x0 = __shfl_sync(mask, values[value], lane_base);
        const float x1 = __shfl_sync(mask, values[value], lane_base + 1);
        const float x2 = __shfl_sync(mask, values[value], lane_base + 2);
        const float x3 = __shfl_sync(mask, values[value], lane_base + 3);
        next[value] = 0.5f * h4_row_dot(d2, x0, x1, x2, x3);
    }

#pragma unroll
    for (int value = 0; value < 8; ++value) {
        const int lane_base = lane & ~12;
        const float x0 = __shfl_sync(mask, next[value], lane_base);
        const float x1 = __shfl_sync(mask, next[value], lane_base + 4);
        const float x2 = __shfl_sync(mask, next[value], lane_base + 8);
        const float x3 = __shfl_sync(mask, next[value], lane_base + 12);
        values[value] = 0.5f * h4_row_dot(d3, x0, x1, x2, x3);
    }

#pragma unroll
    for (int d1_low = 0; d1_low < 2; ++d1_low) {
#pragma unroll
        for (int d0 = 0; d0 < 4; ++d0) {
            const int value = d1_low * 4 + d0;
            const int index = d0 + 4 * (d1_low + 2 * half) + 16 * d2 + 64 * d3;
            destination[index] = __float2bfloat16_rn(values[value]);
        }
    }
}

__global__ void int8_convrot_weighted_embedding_kernel(
    const int8_t* __restrict__ qweight,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ weights,
    float* __restrict__ partials,
    int64_t m,
    int64_t k,
    int64_t n)
{
    extern __shared__ __align__(32) unsigned char shared_bytes[];
    auto* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_bytes);
    auto* shared_b = shared_a + kTileM * kSharedStrideA;

    const int tile_n = static_cast<int>(blockIdx.x);
    const int tile_m = static_cast<int>(blockIdx.y);
    const int partition = static_cast<int>(blockIdx.z);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int warp_m = warp & 1;
    const int warp_n = warp >> 1;
    const int64_t m_base = static_cast<int64_t>(tile_m) * kTileM;
    const int64_t n_base = static_cast<int64_t>(tile_n) * kTileN;
    const int64_t k_begin = static_cast<int64_t>(partition) * kReductionChunk;
    const int64_t k_end = min(k, k_begin + static_cast<int64_t>(kReductionChunk));

    Accumulator accumulators[kAccumulatorRows][kAccumulatorCols];
#pragma unroll
    for (int row = 0; row < kAccumulatorRows; ++row) {
#pragma unroll
        for (int col = 0; col < kAccumulatorCols; ++col) {
            wmma::fill_fragment(accumulators[row][col], 0.0f);
        }
    }

    for (int64_t k_base = k_begin; k_base < k_end; k_base += kTileK) {
        for (int vector_index = static_cast<int>(threadIdx.x);
             vector_index < kTileM * kAVectorsPerRow;
             vector_index += kThreads) {
            const int local_m = vector_index / kAVectorsPerRow;
            const int local_k =
                (vector_index % kAVectorsPerRow) * kBf16ValuesPerVector;
            const auto* source = reinterpret_cast<const uint4*>(
                weights + (m_base + local_m) * k + k_base + local_k);
            auto* destination = reinterpret_cast<uint4*>(
                shared_a + local_m * kSharedStrideA + local_k);
            svdquant::cp_async_16b(destination, source);
        }
        svdquant::cp_async_commit_group();

        for (int local_k = warp; local_k < kTileK; local_k += kThreads / 32) {
            const int64_t global_k = k_base + local_k;
            load_convrot_row(
                qweight + global_k * n + n_base,
                scales[global_k],
                shared_b + local_k * kSharedStrideB);
        }

        svdquant::cp_async_wait_group<0>();
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

    float* partition_output = partials + static_cast<int64_t>(partition) * m * n;
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

__global__ void reduce_int8_convrot_partitions_kernel(
    const float* __restrict__ partials,
    float* __restrict__ output,
    int64_t elements,
    int partitions)
{
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float sum = partials[index];
        for (int partition = 1; partition < partitions; ++partition) {
            sum = __fadd_rn(sum, partials[static_cast<int64_t>(partition) * elements + index]);
        }
        output[index] = sum;
    }
}

}  // namespace
}  // namespace comfy

extern "C" void launch_int8_convrot_weighted_embedding_kernel(
    const void* qweight,
    const void* scales,
    const void* weights,
    float* partials,
    float* output,
    int64_t m,
    int64_t k,
    int64_t n,
    int partitions,
    cudaStream_t stream)
{
    const int expected_partitions = static_cast<int>((k + comfy::kReductionChunk - 1) / comfy::kReductionChunk);
    if (m <= 0 || k <= 0 || n <= 0 || m % comfy::kTileM || k % comfy::kTileK
        || n % comfy::kTileN || partitions != expected_partitions) {
        throw std::runtime_error("invalid int8_convrot_weighted_embedding launch shape");
    }

    const dim3 grid(
        static_cast<unsigned>(n / comfy::kTileN),
        static_cast<unsigned>(m / comfy::kTileM),
        static_cast<unsigned>(partitions));
    constexpr size_t shared_bytes =
        (comfy::kTileM * comfy::kSharedStrideA
         + comfy::kTileK * comfy::kSharedStrideB) * sizeof(__nv_bfloat16);
    comfy::int8_convrot_weighted_embedding_kernel
        <<<grid, comfy::kThreads, shared_bytes, stream>>>(
            static_cast<const int8_t*>(qweight),
            static_cast<const float*>(scales),
            static_cast<const __nv_bfloat16*>(weights),
            partials, m, k, n);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("int8_convrot_weighted_embedding kernel launch failed: ")
            + cudaGetErrorString(error));
    }

    if (partitions > 1) {
        const int64_t elements = m * n;
        constexpr int threads = 256;
        const int blocks = static_cast<int>(
            std::min<int64_t>((elements + threads - 1) / threads, 4096));
        comfy::reduce_int8_convrot_partitions_kernel<<<blocks, threads, 0, stream>>>(
            partials, output, elements, partitions);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("int8_convrot_weighted_embedding reduction launch failed: ")
                + cudaGetErrorString(error));
        }
    }
}
