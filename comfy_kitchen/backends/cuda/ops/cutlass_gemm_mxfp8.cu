/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#ifdef COMFY_HAVE_CUTLASS
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#endif

namespace comfy {
namespace {

#ifdef COMFY_HAVE_CUTLASS

using namespace cute;

template <int TileM, int TileN, int TileK, bool SwapAB>
struct DenseMxfp8Gemm {
    using ElementInput = cutlass::float_e4m3_t;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementC = void;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = std::conditional_t<
        SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    using Arch = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
    using ThreadBlockShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<_1, _1, _1>;

    static constexpr int AlignmentInput =
        128 / cutlass::sizeof_bits<ElementInput>::value;
    static constexpr int AlignmentOutput =
        128 / cutlass::sizeof_bits<ElementOutput>::value;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch,
        OperatorClass,
        ThreadBlockShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutD,
        AlignmentOutput,
        ElementOutput,
        LayoutD,
        AlignmentOutput,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        cutlass::epilogue::fusion::LinearCombination<
            ElementOutput, float, void, float>>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch,
        OperatorClass,
        cutlass::mx_float8_t<ElementInput>,
        LayoutA,
        AlignmentInput,
        cutlass::mx_float8_t<ElementInput>,
        LayoutB,
        AlignmentInput,
        ElementAccumulator,
        ThreadBlockShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class Gemm>
typename Gemm::Arguments make_dense_mxfp8_args(
    void* output,
    const void* a,
    const void* b,
    const void* scale_a,
    const void* scale_b,
    int m,
    int n,
    int k) {
    using ScaleConfig =
        typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementScale = cutlass::float_ue8m0_t;
    using ElementC = void;
    using ElementD = typename Gemm::ElementD;

    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = make_shape(m, n, k, 1);
    args.mainloop.ptr_A = static_cast<const ElementA*>(a);
    args.mainloop.ptr_B = static_cast<const ElementB*>(b);
    args.mainloop.ptr_SFA = static_cast<const ElementScale*>(scale_a);
    args.mainloop.ptr_SFB = static_cast<const ElementScale*>(scale_b);
    args.epilogue.ptr_C = static_cast<const ElementC*>(output);
    args.epilogue.ptr_D = static_cast<ElementD*>(output);
    args.epilogue.thread.alpha_ptr = nullptr;

    args.mainloop.dA = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {m, k, 1});
    args.mainloop.dB = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {n, k, 1});
    args.epilogue.dC = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {m, n, 1});
    args.epilogue.dD = args.epilogue.dC;
    args.mainloop.layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(args.problem_shape);
    args.mainloop.layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(args.problem_shape);

    if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>) {
        args.scheduler.max_swizzle_size = 1;
    }
    if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>) {
        using RasterOrder = decltype(args.scheduler.raster_order);
        args.scheduler.raster_order = RasterOrder::Heuristic;
    }
    args.hw_info.cluster_shape = dim3(1, 1, 1);
    return args;
}

template <int TileM, int TileN, int TileK, bool SwapAB>
bool run_dense_mxfp8(
    const void* activation,
    const void* activation_scale,
    const void* weight,
    const void* weight_scale,
    void* output,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    using Gemm = typename DenseMxfp8Gemm<TileM, TileN, TileK, SwapAB>::Gemm;
    auto args = [&]() {
        if constexpr (SwapAB) {
            return make_dense_mxfp8_args<Gemm>(
                output, weight, activation, weight_scale, activation_scale, n, m, k);
        }
        return make_dense_mxfp8_args<Gemm>(
            output, activation, weight, activation_scale, weight_scale, m, n, k);
    }();

    Gemm gemm;
    const size_t required_workspace = Gemm::get_workspace_size(args);
    if (required_workspace > workspace_size) {
        return false;
    }
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        return false;
    }
    if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
        return false;
    }
    return gemm.run(args, workspace, stream, nullptr, true) == cutlass::Status::kSuccess;
#else
    (void)activation;
    (void)activation_scale;
    (void)weight;
    (void)weight_scale;
    (void)output;
    (void)m;
    (void)n;
    (void)k;
    (void)workspace;
    (void)workspace_size;
    (void)stream;
    return false;
#endif
}

#endif

}  // namespace
}  // namespace comfy

extern "C" bool launch_cutlass_gemm_mxfp8(
    const void* activation,
    const void* activation_scale,
    const void* weight,
    const void* weight_scale,
    void* output,
    int64_t m,
    int64_t n,
    int64_t k,
    int tactic,
    void* workspace,
    int64_t workspace_size,
    cudaStream_t stream) {
#ifdef COMFY_HAVE_CUTLASS
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
        properties.major != 12 || properties.minor != 0) {
        return false;
    }
    if (m <= 0 || n <= 0 || k <= 0 || m % 32 || n % 32 || k % 32) {
        return false;
    }

    switch (tactic) {
    case 0:
        return comfy::run_dense_mxfp8<128, 32, 128, false>(
            activation, activation_scale, weight, weight_scale, output,
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            workspace, static_cast<size_t>(workspace_size), stream);
    case 1:
        return comfy::run_dense_mxfp8<128, 32, 128, true>(
            activation, activation_scale, weight, weight_scale, output,
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            workspace, static_cast<size_t>(workspace_size), stream);
    case 4:
        return comfy::run_dense_mxfp8<128, 128, 128, false>(
            activation, activation_scale, weight, weight_scale, output,
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            workspace, static_cast<size_t>(workspace_size), stream);
    case 5:
        return comfy::run_dense_mxfp8<128, 128, 128, true>(
            activation, activation_scale, weight, weight_scale, output,
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            workspace, static_cast<size_t>(workspace_size), stream);
    default:
        return false;
    }
#else
    (void)activation;
    (void)activation_scale;
    (void)weight;
    (void)weight_scale;
    (void)output;
    (void)m;
    (void)n;
    (void)k;
    (void)tactic;
    (void)workspace;
    (void)workspace_size;
    (void)stream;
    return false;
#endif
}
