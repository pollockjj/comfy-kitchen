/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dtype_dispatch.cuh"
#include "float_utils.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace comfy {
namespace {

constexpr int kThreads = 256;

template <typename IndexType, typename OutputType>
__global__ void mxfp8_embedding_kernel(
    const __nv_fp8_e4m3* __restrict__ qweight,
    const uint8_t* __restrict__ block_scales,
    const IndexType* __restrict__ indices,
    OutputType* __restrict__ output,
    int32_t* __restrict__ invalid,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t num_indices)
{
    const int64_t selected_row = static_cast<int64_t>(blockIdx.x);
    if (selected_row >= num_indices) {
        return;
    }

    const int64_t source_row = static_cast<int64_t>(indices[selected_row]);
    if (source_row < 0 || source_row >= num_embeddings) {
        if (threadIdx.x == 0) {
            atomicExch(invalid, 1);
        }
        for (int64_t col = threadIdx.x; col < embedding_dim; col += blockDim.x) {
            output[selected_row * embedding_dim + col] = static_cast<OutputType>(0.0f);
        }
        return;
    }

    const uint32_t scale_cols = static_cast<uint32_t>(embedding_dim / 32);
    const int64_t source_base = source_row * embedding_dim;
    const int64_t output_base = selected_row * embedding_dim;
    for (int64_t col = threadIdx.x; col < embedding_dim; col += blockDim.x) {
        const size_t scale_offset = scale_factor_swizzled_offset(
            static_cast<size_t>(source_row),
            static_cast<size_t>(col / 32),
            scale_cols);
        const uint8_t exponent = block_scales[scale_offset];
        const float scale = exponent == 0
            ? 0.0f
            : __uint_as_float(static_cast<uint32_t>(exponent) << 23);
        const float value = static_cast<float>(qweight[source_base + col]) * scale;
        output[output_base + col] = static_cast<OutputType>(value);
    }
}

template <typename IndexType>
void dispatch_output(
    const void* qweight,
    const void* block_scales,
    const void* indices,
    void* output,
    int32_t* invalid,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t num_indices,
    int output_dtype_code,
    cudaStream_t stream)
{
    DISPATCH_FP_DTYPE(output_dtype_code, OutputType, [&] {
        mxfp8_embedding_kernel<IndexType, OutputType>
            <<<static_cast<unsigned>(num_indices), kThreads, 0, stream>>>(
                static_cast<const __nv_fp8_e4m3*>(qweight),
                static_cast<const uint8_t*>(block_scales),
                static_cast<const IndexType*>(indices),
                static_cast<OutputType*>(output),
                invalid,
                num_embeddings,
                embedding_dim,
                num_indices);
    });
}

}  // namespace
}  // namespace comfy

extern "C" void launch_mxfp8_embedding_kernel(
    const void* qweight,
    const void* block_scales,
    const void* indices,
    void* output,
    int32_t* invalid,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t num_indices,
    int index_bits,
    int output_dtype_code,
    cudaStream_t stream)
{
    if (num_embeddings <= 0 || embedding_dim <= 0 || num_indices <= 0) {
        throw std::runtime_error("mxfp8_embedding requires non-empty weights and indices");
    }
    if (embedding_dim % 32 != 0) {
        throw std::runtime_error("mxfp8_embedding requires embedding_dim divisible by 32");
    }

    cudaError_t error = cudaMemsetAsync(invalid, 0, sizeof(int32_t), stream);
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("mxfp8_embedding invalid-flag initialization failed: ")
            + cudaGetErrorString(error));
    }

    if (index_bits == 32) {
        comfy::dispatch_output<int32_t>(
            qweight, block_scales, indices, output, invalid,
            num_embeddings, embedding_dim, num_indices, output_dtype_code, stream);
    } else if (index_bits == 64) {
        comfy::dispatch_output<int64_t>(
            qweight, block_scales, indices, output, invalid,
            num_embeddings, embedding_dim, num_indices, output_dtype_code, stream);
    } else {
        throw std::runtime_error("mxfp8_embedding indices must be int32 or int64");
    }

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("mxfp8_embedding kernel launch failed: ")
            + cudaGetErrorString(error));
    }
}
