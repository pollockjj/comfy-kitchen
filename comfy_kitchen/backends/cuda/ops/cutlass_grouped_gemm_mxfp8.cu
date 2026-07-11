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

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#ifdef COMFY_HAVE_CUTLASS
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#endif

namespace comfy {
namespace {

#ifdef COMFY_HAVE_CUTLASS

using namespace cute;

class WorkspaceArena {
public:
    WorkspaceArena(void* ptr, size_t size)
        : base_(static_cast<uint8_t*>(ptr)), size_(size), offset_(0) {}

    template <class T>
    T* allocate(size_t count) {
        return static_cast<T*>(allocate_bytes(sizeof(T) * count));
    }

    void* allocate_bytes(size_t bytes) {
        constexpr size_t alignment = 16;
        const size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
        if (aligned > size_ || bytes > size_ - aligned) {
            throw std::runtime_error("insufficient workspace for grouped MXFP8 GEMM");
        }
        void* result = base_ + aligned;
        offset_ = aligned + bytes;
        return result;
    }

private:
    uint8_t* base_;
    size_t size_;
    size_t offset_;
};

template <
    int ScaleGranularity,
    class ScaleConfig,
    class ElementA,
    class ElementB,
    class ElementSFA,
    class ElementSFB,
    class ElementD,
    class ProblemShape,
    class StrideA,
    class StrideB,
    class StrideD,
    class LayoutSFA,
    class LayoutSFB>
__global__ void prepare_grouped_mxfp8_args(
    ElementA* activations,
    ElementB* weights,
    ElementSFA* activation_scales,
    ElementSFB* weight_scales,
    ElementD* output,
    int group_m,
    int n,
    int k,
    int num_groups,
    ProblemShape* problem_sizes,
    const ElementA** a_ptr,
    const ElementB** b_ptr,
    const ElementSFA** scale_a_ptr,
    const ElementSFB** scale_b_ptr,
    ElementD** output_ptr,
    StrideA* stride_a,
    StrideB* stride_b,
    StrideD* stride_d,
    LayoutSFA* layout_scale_a,
    LayoutSFB* layout_scale_b) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= num_groups) {
        return;
    }

    constexpr size_t scale_mn_alignment = 128;
    constexpr size_t scale_k_alignment = static_cast<size_t>(ScaleGranularity) * 4;
    const size_t scale_n =
        (static_cast<size_t>(n) + scale_mn_alignment - 1) / scale_mn_alignment *
        scale_mn_alignment;
    const size_t swizzled_k =
        (static_cast<size_t>(k) + scale_k_alignment - 1) / scale_k_alignment *
        scale_k_alignment;
    const size_t scale_k = swizzled_k / static_cast<size_t>(ScaleGranularity);
    const size_t activation_row = static_cast<size_t>(group) * group_m;

    // Swap A/B so the large expert output dimension is CUTLASS M and the
    // small routed-token bucket is CUTLASS N.
    problem_sizes[group] = ProblemShape(n, group_m, k);
    stride_a[group] = cutlass::make_cute_packed_stride(StrideA{}, {n, k, 1});
    stride_b[group] = cutlass::make_cute_packed_stride(StrideB{}, {group_m, k, 1});
    stride_d[group] = cutlass::make_cute_packed_stride(StrideD{}, {n, group_m, 1});
    a_ptr[group] = weights + static_cast<size_t>(group) * n * k;
    b_ptr[group] = activations + activation_row * static_cast<size_t>(k);
    output_ptr[group] = output + activation_row * static_cast<size_t>(n);

    layout_scale_a[group] = ScaleConfig::tile_atom_to_shape_SFA(
        make_shape(static_cast<int>(scale_n), group_m, static_cast<int>(swizzled_k), 1));
    scale_a_ptr[group] =
        weight_scales + static_cast<size_t>(group) * scale_n * scale_k;
    layout_scale_b[group] = ScaleConfig::tile_atom_to_shape_SFB(
        make_shape(static_cast<int>(scale_n), group_m, static_cast<int>(swizzled_k), 1));
    scale_b_ptr[group] =
        activation_scales + static_cast<size_t>(group) * group_m * scale_k;
}

template <class ElementD>
bool run_grouped_mxfp8(
    const void* activations_raw,
    const void* activation_scales_raw,
    const void* weights_raw,
    const void* weight_scales_raw,
    void* output_raw,
    int num_groups,
    int group_m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    using ElementInput = cutlass::float_e4m3_t;
    using ElementScale = cutlass::float_ue8m0_t;
    using ElementMainloop = cutlass::mx_float8_t<ElementInput>;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementC = void;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;
    using ClusterShape = Shape<_1, _1, _1>;
    using ThreadBlockShape = Shape<_128, _32, _128>;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInput>::value;
    constexpr int AlignmentB = AlignmentA;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    if (k % AlignmentA || n % AlignmentD) {
        return false;
    }

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        ThreadBlockShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutD*,
        AlignmentD,
        ElementD,
        LayoutD*,
        AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        ElementMainloop,
        LayoutA*,
        AlignmentA,
        ElementMainloop,
        LayoutB*,
        AlignmentB,
        ElementAccumulator,
        ThreadBlockShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GroupProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        GroupProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;
    using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
    constexpr int ScaleGranularity = Gemm::GemmKernel::CollectiveMainloop::TiledMma::SFVecSize;
    static_assert(ScaleGranularity == 32);

    WorkspaceArena arena(workspace, workspace_size);
    auto problem_sizes =
        arena.allocate<typename GroupProblemShape::UnderlyingProblemShape>(num_groups);
    auto a_ptr = arena.allocate<const typename Gemm::ElementA*>(num_groups);
    auto b_ptr = arena.allocate<const typename Gemm::ElementB*>(num_groups);
    auto output_ptr =
        arena.allocate<typename Gemm::EpilogueOutputOp::ElementOutput*>(num_groups);
    auto scale_a_ptr = arena.allocate<const ElementScale*>(num_groups);
    auto scale_b_ptr = arena.allocate<const ElementScale*>(num_groups);
    auto stride_a = arena.allocate<StrideA>(num_groups);
    auto stride_b = arena.allocate<StrideB>(num_groups);
    auto stride_d = arena.allocate<StrideD>(num_groups);
    auto layout_scale_a = arena.allocate<LayoutSFA>(num_groups);
    auto layout_scale_b = arena.allocate<LayoutSFB>(num_groups);

    const int threads = num_groups < 256 ? num_groups : 256;
    const int blocks = (num_groups + threads - 1) / threads;
    prepare_grouped_mxfp8_args<
        ScaleGranularity,
        ScaleConfig,
        typename Gemm::ElementA,
        typename Gemm::ElementB,
        ElementScale,
        ElementScale,
        ElementD,
        typename GroupProblemShape::UnderlyingProblemShape,
        StrideA,
        StrideB,
        StrideD,
        LayoutSFA,
        LayoutSFB><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<typename Gemm::ElementA*>(const_cast<void*>(activations_raw)),
        reinterpret_cast<typename Gemm::ElementB*>(const_cast<void*>(weights_raw)),
        reinterpret_cast<ElementScale*>(const_cast<void*>(activation_scales_raw)),
        reinterpret_cast<ElementScale*>(const_cast<void*>(weight_scales_raw)),
        static_cast<ElementD*>(output_raw),
        group_m,
        n,
        k,
        num_groups,
        problem_sizes,
        a_ptr,
        b_ptr,
        scale_a_ptr,
        scale_b_ptr,
        output_ptr,
        stride_a,
        stride_b,
        stride_d,
        layout_scale_a,
        layout_scale_b);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    thread_local int cached_device = -1;
    thread_local int cached_sm_count = 0;
    if (cached_device != device) {
        cached_device = device;
        cached_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    }
    cutlass::KernelHardwareInfo hardware_info;
    hardware_info.device_id = device;
    hardware_info.sm_count = cached_sm_count;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_groups, problem_sizes, nullptr},
        {a_ptr, stride_a, b_ptr, stride_b, scale_a_ptr, layout_scale_a, scale_b_ptr,
         layout_scale_b},
        {{}, nullptr, nullptr, output_ptr, stride_d},
        hardware_info};
    auto& fusion = arguments.epilogue.thread;
    fusion.alpha = 1.0f;
    fusion.beta = 0.0f;

    Gemm gemm;
    const size_t gemm_workspace_size = Gemm::get_workspace_size(arguments);
    void* gemm_workspace = arena.allocate_bytes(gemm_workspace_size);
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
        return false;
    }
    if (gemm.initialize(arguments, gemm_workspace, stream) != cutlass::Status::kSuccess) {
        return false;
    }
    return gemm.run(stream) == cutlass::Status::kSuccess;
#else
    (void)activations_raw;
    (void)activation_scales_raw;
    (void)weights_raw;
    (void)weight_scales_raw;
    (void)output_raw;
    (void)num_groups;
    (void)group_m;
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

extern "C" bool launch_cutlass_grouped_gemm_mxfp8(
    const void* activation_ptr,
    const void* activation_scale_ptr,
    const void* weight_ptr,
    const void* weight_scale_ptr,
    void* output_ptr,
    int64_t num_groups,
    int64_t group_m,
    int64_t n,
    int64_t k,
    int out_dtype_code,
    void* workspace_ptr,
    int64_t workspace_size,
    cudaStream_t stream) {
#ifdef COMFY_HAVE_CUTLASS
    if (num_groups <= 0) {
        return true;
    }
    if (out_dtype_code == 1) {
        return comfy::run_grouped_mxfp8<cutlass::half_t>(
            activation_ptr, activation_scale_ptr, weight_ptr, weight_scale_ptr, output_ptr,
            static_cast<int>(num_groups), static_cast<int>(group_m), static_cast<int>(n),
            static_cast<int>(k), workspace_ptr, static_cast<size_t>(workspace_size), stream);
    }
    if (out_dtype_code == 2) {
        return comfy::run_grouped_mxfp8<cutlass::bfloat16_t>(
            activation_ptr, activation_scale_ptr, weight_ptr, weight_scale_ptr, output_ptr,
            static_cast<int>(num_groups), static_cast<int>(group_m), static_cast<int>(n),
            static_cast<int>(k), workspace_ptr, static_cast<size_t>(workspace_size), stream);
    }
#else
    (void)activation_ptr;
    (void)activation_scale_ptr;
    (void)weight_ptr;
    (void)weight_scale_ptr;
    (void)output_ptr;
    (void)num_groups;
    (void)group_m;
    (void)n;
    (void)k;
    (void)out_dtype_code;
    (void)workspace_ptr;
    (void)workspace_size;
    (void)stream;
#endif
    return false;
}
