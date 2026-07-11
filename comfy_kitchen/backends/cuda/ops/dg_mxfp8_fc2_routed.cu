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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "float_utils.cuh"

#include <cstddef>
#include <cstdint>

#ifdef COMFY_HAVE_CUTLASS
#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"
#endif

namespace comfy {
namespace dg_mxfp8_fc2 {

#if defined(COMFY_HAVE_CUTLASS) && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

constexpr int kNumExperts = 128;
constexpr int kRoutesPerTask = 8;
constexpr int kOutputSize = 2816;
constexpr int kReductionSize = 704;
constexpr int kOutputTile = 128;
constexpr int kOutputTiles = kOutputSize / kOutputTile;
constexpr int kMmaRows = 16;
constexpr int kBlockCols = kReductionSize / 32;
constexpr int kScaleStorageCols = ((kBlockCols + 3) / 4) * 4;
constexpr int kPersistentCtas = 256;

struct alignas(16) Fc2Task {
    int32_t expert;
    int32_t route_start;
    int32_t rank_start;
    int32_t valid_rows;
};

static_assert(sizeof(Fc2Task) == 16);

__global__ void build_fc2_tasks(
    const int32_t* indptr,
    Fc2Task* tasks,
    int32_t* task_count,
    int32_t* next_task) {
    __shared__ int32_t chunk_offsets[kNumExperts + 1];
    const int expert = threadIdx.x;
    if (expert < kNumExperts) {
        const int routes = indptr[expert + 1] - indptr[expert];
        chunk_offsets[expert] = (routes + kRoutesPerTask - 1) / kRoutesPerTask;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t running = 0;
#pragma unroll
        for (int index = 0; index < kNumExperts; ++index) {
            const int32_t chunks = chunk_offsets[index];
            chunk_offsets[index] = running;
            running += chunks;
        }
        chunk_offsets[kNumExperts] = running;
        *task_count = running;
        *next_task = 0;
    }
    __syncthreads();

    if (expert < kNumExperts) {
        const int route_start = indptr[expert];
        const int route_count = indptr[expert + 1] - route_start;
        const int chunks = (route_count + kRoutesPerTask - 1) / kRoutesPerTask;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int rank_start = chunk * kRoutesPerTask;
            const int valid_rows = min(kRoutesPerTask, route_count - rank_start);
            tasks[chunk_offsets[expert] + chunk] = {
                expert,
                route_start + rank_start,
                rank_start,
                valid_rows};
        }
    }
}

template <class OutputType>
__device__ __forceinline__ OutputType convert_output(float value);

template <>
__device__ __forceinline__ __half convert_output(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 convert_output(float value) {
    return __float2bfloat16_rn(value);
}

template <class OutputType>
__global__ __launch_bounds__(256) void fc2_routed_persistent(
    const uint8_t* weights,
    const uint8_t* weight_scales,
    const uint8_t* activations,
    const uint8_t* activation_scales,
    OutputType* output,
    const Fc2Task* tasks,
    const int32_t* task_count,
    int32_t* next_task,
    int scale_group_m) {
    using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
        cutlass::float_e4m3_t,
        cutlass::float_e4m3_t,
        float,
        cutlass::float_ue8m0_t,
        32>;

    __shared__ int32_t shared_task_index;
    __shared__ Fc2Task shared_task;
    __shared__ uint8_t shared_activations[kRoutesPerTask * kReductionSize];
    __shared__ uint8_t shared_activation_scales[kRoutesPerTask * kBlockCols];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    while (true) {
        if (threadIdx.x == 0) {
            shared_task_index = atomicAdd(next_task, 1);
        }
        __syncthreads();
        if (shared_task_index >= *task_count) {
            return;
        }
        if (threadIdx.x == 0) {
            shared_task = tasks[shared_task_index];
        }
        __syncthreads();

        const int expert = shared_task.expert;
        const int route_start = shared_task.route_start;
        const int rank_start = shared_task.rank_start;
        const int valid_rows = shared_task.valid_rows;
        const uint8_t* expert_weights =
            weights + static_cast<size_t>(expert) * kOutputSize * kReductionSize;
        const uint8_t* expert_weight_scales =
            weight_scales + static_cast<size_t>(expert) * kOutputSize * kScaleStorageCols;
        const uint8_t* expert_activation_scales =
            activation_scales +
            static_cast<size_t>(expert) * scale_group_m * kScaleStorageCols;

        for (int linear = threadIdx.x;
             linear < kRoutesPerTask * kReductionSize;
             linear += blockDim.x) {
            const int row = linear / kReductionSize;
            const int col = linear - row * kReductionSize;
            shared_activations[linear] = row < valid_rows
                ? activations[
                      static_cast<size_t>(route_start + row) * kReductionSize + col]
                : 0;
        }
        for (int linear = threadIdx.x;
             linear < kRoutesPerTask * kBlockCols;
             linear += blockDim.x) {
            const int row = linear / kBlockCols;
            const int block_col = linear - row * kBlockCols;
            shared_activation_scales[linear] = row < valid_rows
                ? expert_activation_scales[scale_factor_swizzled_offset(
                      rank_start + row, block_col, kBlockCols)]
                : 0;
        }
        __syncthreads();

        for (int output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
            const int output_base = output_tile * kOutputTile + warp * kMmaRows;
            float d0 = 0.0f;
            float d1 = 0.0f;
            float d2 = 0.0f;
            float d3 = 0.0f;

#pragma unroll
            for (int block_col = 0; block_col < kBlockCols; ++block_col) {
                uint32_t a[4] = {};
                uint32_t b[2] = {};
#pragma unroll
                for (int value = 0; value < 16; ++value) {
                    const int linear =
                        (lane & 3) * 64 + (lane >> 2) + (value & 3) * 16 +
                        ((value >> 2) & 1) * 8 + (value >> 3) * 256;
                    const int row = output_base + (linear & 15);
                    const int col = block_col * 32 + (linear >> 4);
                    const uint32_t byte =
                        expert_weights[static_cast<size_t>(row) * kReductionSize + col];
                    a[value >> 2] |= byte << ((value & 3) * 8);
                }
#pragma unroll
                for (int value = 0; value < 8; ++value) {
                    const int linear =
                        (lane & 3) * 32 + (lane >> 2) + (value & 3) * 8 +
                        (value >> 2) * 128;
                    const int row = linear & 7;
                    const int col = block_col * 32 + (linear >> 3);
                    const uint32_t byte =
                        shared_activations[row * kReductionSize + col];
                    b[value >> 2] |= byte << ((value & 3) * 8);
                }

                const int scale_a_row = output_base + (lane & 1) * 8 + (lane >> 2);
                const int scale_b_row = lane >> 2;
                const uint8_t scale_a = expert_weight_scales[
                    scale_factor_swizzled_offset(scale_a_row, block_col, kBlockCols)];
                const uint8_t scale_b =
                    shared_activation_scales[scale_b_row * kBlockCols + block_col];
                Mma::fma(
                    d0, d1, d2, d3,
                    a[0], a[1], a[2], a[3],
                    b[0], b[1],
                    d0, d1, d2, d3,
                    scale_a, scale_b);
            }

            const float values[4] = {d0, d1, d2, d3};
#pragma unroll
            for (int value = 0; value < 4; ++value) {
                const int linear =
                    (lane & 3) * 32 + (lane >> 2) + (value & 1) * 16 +
                    (value >> 1) * 8;
                const int output_row = output_base + (linear & 15);
                const int route = linear >> 4;
                if (route < valid_rows) {
                    output[
                        static_cast<size_t>(route_start + route) * kOutputSize + output_row] =
                        convert_output<OutputType>(values[value]);
                }
            }
        }
        __syncthreads();
    }
}

#endif

}  // namespace dg_mxfp8_fc2
}  // namespace comfy

extern "C" bool launch_dg_mxfp8_fc2_routed(
    const void* activations,
    const void* activation_scales,
    const void* weights,
    const void* weight_scales,
    void* output,
    const int32_t* indptr,
    int64_t routes,
    int64_t scale_group_m,
    int64_t num_experts,
    int out_dtype_code,
    void* workspace,
    int64_t workspace_size,
    cudaStream_t stream) {
#if defined(COMFY_HAVE_CUTLASS) && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    using namespace comfy::dg_mxfp8_fc2;
    if (activations == nullptr || activation_scales == nullptr || weights == nullptr ||
        weight_scales == nullptr || output == nullptr || indptr == nullptr ||
        workspace == nullptr || routes <= 0 || scale_group_m <= 0 ||
        num_experts != kNumExperts || (out_dtype_code != 1 && out_dtype_code != 2)) {
        return false;
    }
    const int64_t max_chunks =
        (routes + 7 * num_experts + 7) / kRoutesPerTask;
    const int64_t max_tasks = max_chunks;
    if (max_tasks <= 0 || max_tasks > INT32_MAX) {
        return false;
    }
    const size_t task_bytes = static_cast<size_t>(max_tasks) * sizeof(Fc2Task);
    const size_t required_bytes = task_bytes + 2 * sizeof(int32_t);
    if (workspace_size < 0 || required_bytes > static_cast<size_t>(workspace_size)) {
        return false;
    }
    auto* tasks = static_cast<Fc2Task*>(workspace);
    auto* task_count = reinterpret_cast<int32_t*>(
        static_cast<uint8_t*>(workspace) + task_bytes);
    int32_t* next_task = task_count + 1;
    build_fc2_tasks<<<1, kNumExperts, 0, stream>>>(
        indptr, tasks, task_count, next_task);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }
    if (out_dtype_code == 1) {
        fc2_routed_persistent<__half><<<kPersistentCtas, 256, 0, stream>>>(
            static_cast<const uint8_t*>(weights),
            static_cast<const uint8_t*>(weight_scales),
            static_cast<const uint8_t*>(activations),
            static_cast<const uint8_t*>(activation_scales),
            static_cast<__half*>(output), tasks, task_count, next_task,
            static_cast<int>(scale_group_m));
    } else {
        fc2_routed_persistent<__nv_bfloat16><<<kPersistentCtas, 256, 0, stream>>>(
            static_cast<const uint8_t*>(weights),
            static_cast<const uint8_t*>(weight_scales),
            static_cast<const uint8_t*>(activations),
            static_cast<const uint8_t*>(activation_scales),
            static_cast<__nv_bfloat16*>(output), tasks, task_count, next_task,
            static_cast<int>(scale_group_m));
    }
    return cudaPeekAtLastError() == cudaSuccess;
#else
    (void)activations;
    (void)activation_scales;
    (void)weights;
    (void)weight_scales;
    (void)output;
    (void)indptr;
    (void)routes;
    (void)scale_group_m;
    (void)num_experts;
    (void)out_dtype_code;
    (void)workspace;
    (void)workspace_size;
    (void)stream;
    return false;
#endif
}
