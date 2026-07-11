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

/*
 * The narrow-N mainloop builder below is adapted from NVIDIA CUTLASS's
 * sm120_blockscaled_mma_builder.inl.
 *
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

namespace cutlass_collective = cutlass::gemm::collective;
namespace cutlass_collective_detail = cutlass::gemm::collective::detail;

template <class TileShapeMNK, class ClusterShapeMNK, class StageCountType>
struct DgNarrowMxfp8MainloopBuilder {
    using ElementPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementScale = typename cutlass_collective_detail::blockscaled::blockscaled_type<
        cutlass_collective::KernelScheduleAuto, ElementPair>::sf_type;
    using ElementA = typename cutlass_collective_detail::blockscaled::blockscaled_type<
        cutlass_collective::KernelScheduleAuto, ElementPair>::data_type;
    using ElementB = ElementA;
    using ElementAccumulator = float;
    using GmemLayoutATag = cutlass::layout::RowMajor*;
    using GmemLayoutBTag = cutlass::layout::ColumnMajor*;
    static constexpr int SFVectorSize =
        cutlass_collective_detail::blockscaled::blockscaled_type<
            cutlass_collective::KernelScheduleAuto, ElementPair>::SfVectorSize;

    static constexpr cute::UMMA::Major UmmaMajorA =
        cutlass_collective_detail::tag_to_umma_major_A<GmemLayoutATag>();
    static constexpr cute::UMMA::Major UmmaMajorB =
        cutlass_collective_detail::tag_to_umma_major_B<GmemLayoutBTag>();
    static_assert(UmmaMajorA == cute::UMMA::Major::K &&
                  UmmaMajorB == cute::UMMA::Major::K);
    static_assert(cute::is_static_v<TileShapeMNK>);
    static_assert(cute::is_static_v<ClusterShapeMNK>);
    static_assert(cute::size(ClusterShapeMNK{}) == cute::Int<1>{});
    static_assert(cute::size<1>(TileShapeMNK{}) == 16);

    static constexpr auto Instr = cutlass_collective_detail::blockscaled::select_instr<
        ElementPair,
        ElementPair,
        ElementAccumulator,
        UmmaMajorA,
        UmmaMajorB,
        cutlass_collective::KernelScheduleAuto>();
    static constexpr bool UseMxf8f6f4 =
        Instr == cutlass_collective_detail::blockscaled::BlockScaledInstr::MXF4F6F8;
    static_assert(UseMxf8f6f4);

    using PermTileM = decltype(cute::min(cute::size<0>(TileShapeMNK{}), cute::_128{}));
    // CUTLASS's generic block-scaled builder uses a 32-column permutation tile
    // and therefore rejects CTA N=16.  The SM120 MMA atom is 8 columns wide;
    // the cooperative 2-way N layout natively covers this 16-column tile.
    using PermTileN = cute::_16;
    using PermTileK = cute::_32;
    using AtomLayoutMNK = cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>;
    using TiledMma = decltype(cute::make_tiled_mma(
        cute::rr_blockscaled_op_selector_sm120<
            ElementA,
            ElementB,
            ElementAccumulator,
            ElementScale,
            SFVectorSize,
            UseMxf8f6f4>(),
        AtomLayoutMNK{},
        cute::Tile<PermTileM, PermTileN, PermTileK>{}));

    static constexpr int MmaScaleFactors =
        cute::size<2>(typename TiledMma::AtomShape_MNK{}) / SFVectorSize;
    using SmemAllocTypeA = typename TiledMma::ValTypeA;
    using SmemAllocTypeB = typename TiledMma::ValTypeB;
    using SmemAllocTypeScale = ElementScale;
    using GmemTiledCopyPairA = decltype(cute::make_tuple(SM90_TMA_LOAD{}, SM90_TMA_LOAD{}));
    using GmemTiledCopyPairB = GmemTiledCopyPairA;
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;

    using SmemLayoutAtomA = decltype(cutlass_collective_detail::sm120_rr_smem_selector<
        SmemAllocTypeA, decltype(cute::size<2>(TileShapeMNK{}))>());
    using SmemLayoutAtomB = decltype(cutlass_collective_detail::sm120_rr_smem_selector<
        SmemAllocTypeB, decltype(cute::size<2>(TileShapeMNK{}))>());
    using SmemCopyAtomA = cute::Copy_Atom<
        decltype(cutlass_collective_detail::sm120_rr_smem_copy_selector_A<
            ElementA, ElementB, UseMxf8f6f4>()),
        SmemAllocTypeA>;
    using SmemCopyAtomB = cute::Copy_Atom<
        decltype(cutlass_collective_detail::sm120_rr_smem_copy_selector_B<
            ElementA, ElementB, UseMxf8f6f4>()),
        SmemAllocTypeB>;
    using SmemCopyAtomScale =
        cute::Copy_Atom<cute::UniversalCopy<SmemAllocTypeScale>, SmemAllocTypeScale>;
    using SmemCopyAtomsA = decltype(cute::make_tuple(SmemCopyAtomA{}, SmemCopyAtomScale{}));
    using SmemCopyAtomsB = decltype(cute::make_tuple(SmemCopyAtomB{}, SmemCopyAtomScale{}));

    using ScaleBlockMN = typename Sm1xxBlkScaledConfig::Blk_MN;
    using ScaleBlockSF = typename Sm1xxBlkScaledConfig::Blk_SF;
    using ScaleBlockElements = decltype(ScaleBlockMN{} * ScaleBlockSF{});
    using ScaleBasicMNShape = cute::Shape<cute::_32, cute::_4>;
    using ScaleBasicMNStride = cute::Stride<cute::_16, cute::_4>;
    using ScaleBasicKShape = cute::Shape<
        cute::Int<SFVectorSize>, cute::Int<MmaScaleFactors>>;
    using ScaleBasicKStride = cute::Stride<cute::_0, cute::_1>;

    using ScaleAShapeM = decltype(cute::prepend(
        cute::size<0>(TileShapeMNK{}) / ScaleBlockMN{}, ScaleBasicMNShape{}));
    using ScaleStrideMN = decltype(cute::prepend(
        ScaleBlockElements{}, ScaleBasicMNStride{}));
    using ScaleShapeK = decltype(cute::prepend(
        cute::make_shape(
            ScaleBlockSF{} / cute::Int<MmaScaleFactors>{},
            cute::size<2>(TileShapeMNK{}) / cute::Int<SFVectorSize>{} /
                ScaleBlockSF{}),
        ScaleBasicKShape{}));
    using ScaleAStrideK = decltype(cute::prepend(
        cute::make_stride(
            cute::Int<MmaScaleFactors>{},
            cute::size<0>(TileShapeMNK{}) / ScaleBlockMN{} * ScaleBlockElements{}),
        ScaleBasicKStride{}));
    using ScaleAShape = decltype(cute::make_shape(ScaleAShapeM{}, ScaleShapeK{}));
    using ScaleAStride = decltype(cute::make_stride(ScaleStrideMN{}, ScaleAStrideK{}));
    using SmemLayoutAtomScaleA =
        decltype(cute::make_layout(ScaleAShape{}, ScaleAStride{}));

    using ScaleBTileN = cute::Int<cute::max(cute::size<1>(TileShapeMNK{}), 128)>;
    using ScaleBShapeN = decltype(cute::prepend(
        ScaleBTileN{} / ScaleBlockMN{}, ScaleBasicMNShape{}));
    using ScaleBStrideK = decltype(cute::prepend(
        cute::make_stride(
            cute::Int<MmaScaleFactors>{},
            ScaleBTileN{} / ScaleBlockMN{} * ScaleBlockElements{}),
        ScaleBasicKStride{}));
    using ScaleBShape = decltype(cute::make_shape(ScaleBShapeN{}, ScaleShapeK{}));
    using ScaleBStride = decltype(cute::make_stride(ScaleStrideMN{}, ScaleBStrideK{}));
    using SmemLayoutAtomScaleB =
        decltype(cute::make_layout(ScaleBShape{}, ScaleBStride{}));
    using SmemLayoutAtomsA = decltype(cute::make_tuple(
        SmemLayoutAtomA{}, SmemLayoutAtomScaleA{}));
    using SmemLayoutAtomsB = decltype(cute::make_tuple(
        SmemLayoutAtomB{}, SmemLayoutAtomScaleB{}));

    using StrideA = cutlass::gemm::TagToStrideA_t<GmemLayoutATag>;
    using StrideB = cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>;
    using InternalStrideA = cute::remove_pointer_t<StrideA>;
    using InternalStrideB = cute::remove_pointer_t<StrideB>;
    using InternalLayoutSFA = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFA());
    using InternalLayoutSFB = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFB());
    using LayoutSFA = InternalLayoutSFA*;
    using LayoutSFB = InternalLayoutSFB*;
    using StridePairA = decltype(cute::make_tuple(StrideA{}, LayoutSFA{}));
    using StridePairB = decltype(cute::make_tuple(StrideB{}, LayoutSFB{}));

    static constexpr uint32_t SchedulerPipelineStageCount = 3;
    static constexpr int SchedulerPipelineStorage =
        sizeof(cutlass::PipelineDetail::PipelineAsyncSharedStorage<8>);
    static constexpr int TensorMapStorage = sizeof(cute::TmaDescriptor) * 2;
    static constexpr int TensorMapReadyPipelineStorage = sizeof(
        typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);
    static constexpr int ReducedSmemCapacityBytes =
        cutlass_collective_detail::sm120_smem_capacity_bytes - SchedulerPipelineStorage -
        TensorMapStorage - TensorMapReadyPipelineStorage;
    static constexpr int PipelineStages =
        cutlass_collective_detail::sm100_compute_stage_count_or_override_blockscaled<
        ReducedSmemCapacityBytes,
        SmemAllocTypeA,
        SmemAllocTypeB,
        TileShapeMNK,
        SmemLayoutAtomScaleA,
        SmemLayoutAtomScaleB>(StageCountType{});

    using KernelSchedule = cutlass::gemm::
        KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<
            SchedulerPipelineStageCount>;
    using DispatchPolicy = cutlass::gemm::MainloopSm120ArrayTmaWarpSpecializedBlockScaled<
        PipelineStages,
        SchedulerPipelineStageCount,
        ClusterShapeMNK,
        KernelSchedule>;
    using CollectiveOp = cutlass_collective::CollectiveMma<
        DispatchPolicy,
        TileShapeMNK,
        cute::tuple<ElementA, ElementScale>,
        StridePairA,
        cute::tuple<ElementB, ElementScale>,
        StridePairB,
        TiledMma,
        GmemTiledCopyPairA,
        SmemLayoutAtomsA,
        SmemCopyAtomsA,
        cute::identity,
        GmemTiledCopyPairB,
        SmemLayoutAtomsB,
        SmemCopyAtomsB,
        cute::identity>;
};

template <bool IsNarrow, class TileShapeMNK, class ClusterShapeMNK, class StageCountType>
struct DgMxfp8MainloopSelector;

template <class TileShapeMNK, class ClusterShapeMNK, class StageCountType>
struct DgMxfp8MainloopSelector<true, TileShapeMNK, ClusterShapeMNK, StageCountType> {
    using CollectiveOp = typename DgNarrowMxfp8MainloopBuilder<
        TileShapeMNK, ClusterShapeMNK, StageCountType>::CollectiveOp;
};

template <class TileShapeMNK, class ClusterShapeMNK, class StageCountType>
struct DgMxfp8MainloopSelector<false, TileShapeMNK, ClusterShapeMNK, StageCountType> {
    using ElementMainloop = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        ElementMainloop,
        cutlass::layout::RowMajor*,
        16,
        ElementMainloop,
        cutlass::layout::ColumnMajor*,
        16,
        float,
        TileShapeMNK,
        ClusterShapeMNK,
        StageCountType,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
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
    const int32_t* m_indptr,
    int scale_group_m,
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
    int m = group_m;
    size_t activation_row = static_cast<size_t>(group) * group_m;
    if (m_indptr != nullptr) {
        const int32_t start = m_indptr[group];
        const int32_t end = m_indptr[group + 1];
        m = end - start;
        activation_row = static_cast<size_t>(start);
    }

    // Swap A/B so the large expert output dimension is CUTLASS M and the
    // small routed-token bucket is CUTLASS N.
    problem_sizes[group] = ProblemShape(n, m, k);
    stride_a[group] = cutlass::make_cute_packed_stride(StrideA{}, {n, k, 1});
    stride_b[group] = cutlass::make_cute_packed_stride(StrideB{}, {m, k, 1});
    stride_d[group] = cutlass::make_cute_packed_stride(StrideD{}, {n, m, 1});
    a_ptr[group] = weights + static_cast<size_t>(group) * n * k;
    b_ptr[group] = activations + activation_row * static_cast<size_t>(k);
    output_ptr[group] = output + activation_row * static_cast<size_t>(n);

    layout_scale_a[group] = ScaleConfig::tile_atom_to_shape_SFA(
        make_shape(static_cast<int>(scale_n), m, static_cast<int>(swizzled_k), 1));
    scale_a_ptr[group] =
        weight_scales + static_cast<size_t>(group) * scale_n * scale_k;
    layout_scale_b[group] = ScaleConfig::tile_atom_to_shape_SFB(
        make_shape(static_cast<int>(scale_n), m, static_cast<int>(swizzled_k), 1));
    scale_b_ptr[group] =
        activation_scales + static_cast<size_t>(group) * scale_group_m * scale_k;
}

template <int TileM, int TileN, int TileK, class ElementD>
bool run_grouped_mxfp8(
    const void* activations_raw,
    const void* activation_scales_raw,
    const void* weights_raw,
    const void* weight_scales_raw,
    void* output_raw,
    int num_groups,
    int group_m,
    const int32_t* m_indptr,
    int scale_group_m,
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
    // The grouped problem swaps A/B so TileM spans the expert output and
    // TileN spans the routed-token bucket.
    using ThreadBlockShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using EpilogueTile = cute::conditional_t<
        TileN == 16,
        Shape<_64, _16>,
        cutlass::epilogue::collective::EpilogueTileAuto>;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInput>::value;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    if (k % AlignmentA || n % AlignmentD) {
        return false;
    }

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        ThreadBlockShape,
        ClusterShape,
        EpilogueTile,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutD*,
        AlignmentD,
        ElementD,
        LayoutD*,
        AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using MainloopStageCount = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
    using CollectiveMainloop = typename DgMxfp8MainloopSelector<
        TileN == 16,
        ThreadBlockShape,
        ClusterShape,
        MainloopStageCount>::CollectiveOp;

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
        m_indptr,
        scale_group_m,
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
    (void)m_indptr;
    (void)scale_group_m;
    (void)n;
    (void)k;
    (void)workspace;
    (void)workspace_size;
    (void)stream;
    return false;
#endif
}

template <class ElementD>
bool run_selected_grouped_mxfp8(
    const void* activations_raw,
    const void* activation_scales_raw,
    const void* weights_raw,
    const void* weight_scales_raw,
    void* output_raw,
    int num_groups,
    int group_m,
    const int32_t* m_indptr,
    int scale_group_m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
    if (n == 2816 && k == 704) {
        return run_grouped_mxfp8<128, 16, 128, ElementD>(
            activations_raw, activation_scales_raw, weights_raw, weight_scales_raw,
            output_raw, num_groups, group_m, m_indptr, scale_group_m, n, k, workspace,
            workspace_size, stream);
    }
    return run_grouped_mxfp8<128, 32, 128, ElementD>(
        activations_raw, activation_scales_raw, weights_raw, weight_scales_raw,
        output_raw, num_groups, group_m, m_indptr, scale_group_m, n, k, workspace,
        workspace_size, stream);
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
        return comfy::run_selected_grouped_mxfp8<cutlass::half_t>(
            activation_ptr, activation_scale_ptr, weight_ptr, weight_scale_ptr, output_ptr,
            static_cast<int>(num_groups), static_cast<int>(group_m), nullptr,
            static_cast<int>(group_m), static_cast<int>(n), static_cast<int>(k), workspace_ptr,
            static_cast<size_t>(workspace_size), stream);
    }
    if (out_dtype_code == 2) {
        return comfy::run_selected_grouped_mxfp8<cutlass::bfloat16_t>(
            activation_ptr, activation_scale_ptr, weight_ptr, weight_scale_ptr, output_ptr,
            static_cast<int>(num_groups), static_cast<int>(group_m), nullptr,
            static_cast<int>(group_m), static_cast<int>(n), static_cast<int>(k), workspace_ptr,
            static_cast<size_t>(workspace_size), stream);
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

extern "C" bool launch_cutlass_grouped_gemm_mxfp8_variable(
    const void* activation_ptr,
    const void* activation_scale_ptr,
    const void* weight_ptr,
    const void* weight_scale_ptr,
    void* output_ptr,
    const int32_t* m_indptr_ptr,
    int64_t num_groups,
    int64_t scale_group_m,
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
        return comfy::run_selected_grouped_mxfp8<cutlass::half_t>(
            activation_ptr, activation_scale_ptr, weight_ptr, weight_scale_ptr, output_ptr,
            static_cast<int>(num_groups), 0, m_indptr_ptr, static_cast<int>(scale_group_m),
            static_cast<int>(n), static_cast<int>(k), workspace_ptr,
            static_cast<size_t>(workspace_size), stream);
    }
    if (out_dtype_code == 2) {
        return comfy::run_selected_grouped_mxfp8<cutlass::bfloat16_t>(
            activation_ptr, activation_scale_ptr, weight_ptr, weight_scale_ptr, output_ptr,
            static_cast<int>(num_groups), 0, m_indptr_ptr, static_cast<int>(scale_group_m),
            static_cast<int>(n), static_cast<int>(k), workspace_ptr,
            static_cast<size_t>(workspace_size), stream);
    }
#else
    (void)activation_ptr;
    (void)activation_scale_ptr;
    (void)weight_ptr;
    (void)weight_scale_ptr;
    (void)output_ptr;
    (void)m_indptr_ptr;
    (void)num_groups;
    (void)scale_group_m;
    (void)n;
    (void)k;
    (void)out_dtype_code;
    (void)workspace_ptr;
    (void)workspace_size;
    (void)stream;
#endif
    return false;
}
