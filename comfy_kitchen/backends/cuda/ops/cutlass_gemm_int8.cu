/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * INT8 GEMM with a FUSED dequant epilogue via CUTLASS (EVT):
 *   D[m,n] = (sum_k A[m,k]*B[n,k]) * x_scale[m] * w_scale[n] + bias[n]   -> out dtype
 *
 * Replaces cuBLAS-GEMM(int32) + separate dequant with one near-peak kernel.
 * Multiple tile configs are instantiated and the fastest for each (M,N,K) is
 * picked at runtime and cached (like Triton's autotuner / cuBLAS's heuristic),
 * so it adapts to the GPU instead of relying on one hand-tuned tile.
 * Falls back to cuBLAS when CUTLASS is unavailable or no config can run.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <climits>
#include <cstddef>
#include <cstdint>

#ifdef COMFY_HAVE_CUTLASS

#include <map>
#include <tuple>
#include <mutex>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

namespace {
using namespace cute;

template <typename ThreadMap, bool Scalar>
struct WeightScaleBroadcast;

template <typename ThreadMap>
struct WeightScaleBroadcast<ThreadMap, false> {
    using Type = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, float, cute::Stride<_0, _1, int32_t>>;

    static typename Type::Arguments arguments(const float* scale, int n) {
        return {scale, 0.f, {_0{}, _1{}, n}};
    }
};

template <typename ThreadMap>
struct WeightScaleBroadcast<ThreadMap, true> {
    using Type = cutlass::epilogue::threadblock::VisitorScalarBroadcast<float>;

    static typename Type::Arguments arguments(const float* scale, int) {
        typename Type::Arguments result{};
        result.scalar_ptrs[0] = scale;
        return result;
    }
};

// One fused int8 GEMM, parameterized on output type AND tile/warp/stage config.
template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm80, typename ElementBias = float,
          bool ScalarWeightScale = false>
struct FusedInt8Gemm {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;   // B[N,K] row == [K,N] col
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = 16, AlignB = 16;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
    using Bias   = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementBias, cute::Stride<_0, _1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using Add2 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT2 = cutlass::epilogue::threadblock::Sm80EVT<Add2, EVT1, Bias>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT2>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const ElementBias* bias, ElementOutput* D, int M, int N, int K, cudaStream_t stream) {
        return run_strided(A, B, xs, ws, bias, D, M, N, K, N, stream);
    }

    static bool run_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            const ElementBias* bias, ElementOutput* D, int M, int N, int K,
                            int output_stride, cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(M, N, K);
        const auto weight_scale_args = WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { {  { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
                 weight_scale_args, {} },
              {const_cast<ElementBias*>(bias), ElementBias(0), {_0{}, _1{}, N}}, {} },
            {D, {output_stride, _1{}, M * output_stride}} };
        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
            const_cast<int8_t*>(A), const_cast<int8_t*>(B), nullptr, nullptr,
            (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        if (Gemm::get_workspace_size(args) != 0) return false;  // kGemm mode -> 0; bail if not
        if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

template <typename ElementOutput, int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages,
          typename ArchTag = cutlass::arch::Sm80, bool ScalarWeightScale = false>
struct FusedInt8GemmNoBias {
    using ElementA = int8_t; using ElementB = int8_t;
    using ElementC = ElementOutput;
    using ElementAcc = int32_t; using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = 16, AlignB = 16;
    static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using TB   = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
    using Accum  = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = typename WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::Type;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT1>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignA,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignB,
        ElementC, LayoutC, AlignC,
        ElementAcc, ElementCompute,
        cutlass::arch::OpClassTensorOp, ArchTag,
        TB, Warp, Inst, EVTD,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages, cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    ElementOutput* D, int M, int N, int K, cudaStream_t stream) {
        return run_strided(A, B, xs, ws, D, M, N, K, N, stream);
    }

    static bool run_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            ElementOutput* D, int M, int N, int K, int output_stride,
                            cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(M, N, K);
        const auto weight_scale_args = WeightScaleBroadcast<ThreadMap, ScalarWeightScale>::arguments(ws, N);
        typename EVTD::Arguments cb{
            { { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
              weight_scale_args, {} },
            {D, {output_stride, _1{}, M * output_stride}} };
        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
            const_cast<int8_t*>(A), const_cast<int8_t*>(B), nullptr, nullptr,
            (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        if (Gemm::get_workspace_size(args) != 0) return false;
        if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
        return gemm(stream) == cutlass::Status::kSuccess;
    }

    static bool run_batched(const int8_t* A, const int8_t* B, const float* xs,
                            const float* ws, ElementOutput* D, int groups, int M,
                            int N, int K, cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(M, N, K);
        typename EVTD::Arguments cb{
            { { {}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {} },
              {const_cast<float*>(ws), 0.f, {_0{}, _1{}, N}}, {} },
            {D, {N, _1{}, M * N}} };
        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kBatched, problem, groups, cb,
            const_cast<int8_t*>(A), const_cast<int8_t*>(B), nullptr, nullptr,
            static_cast<int64_t>(M) * K, static_cast<int64_t>(N) * K, 0, 0,
            K, K, 0, 0);

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        if (Gemm::get_workspace_size(args) != 0) return false;
        if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

// Autotuning dispatcher: try each tile config, time it, cache the fastest per
// (M,N,K). First call for a shape pays the tuning cost; the rest hit the cache.
template <typename OutT>
bool dispatch_fused(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                    const float* bias, OutT* D, int M, int N, int K, cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, const float*, OutT*, int, int, int, cudaStream_t);
    // Tile configs spanning big-GPU/large-M (wide) to small-GPU/small-M (more CTAs).
    static const Fn runners[] = {
        &FusedInt8Gemm<OutT, 128, 256, 64, 64, 64, 64, 3>::run,
        &FusedInt8Gemm<OutT, 128, 128, 64, 64, 64, 64, 4>::run,
        &FusedInt8Gemm<OutT,  64, 128, 64, 32, 64, 64, 4>::run,
    };
    constexpr int NC = sizeof(runners) / sizeof(runners[0]);

    static std::mutex mtx;
    static std::map<std::tuple<int, int, int>, int> cache;   // (M,N,K) -> best config (or -1 = none)
    const std::tuple<int, int, int> key{M, N, K};

    static thread_local int last_m = -1;
    static thread_local int last_n = -1;
    static thread_local int last_k = -1;
    static thread_local int last_best = -2;
    if (M == last_m && N == last_n && K == last_k) {
        if (last_best < 0) return false;
        return runners[last_best](A, B, xs, ws, bias, D, M, N, K, stream);
    }

    int best;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(key);
        best = (it != cache.end()) ? it->second : -2;
    }
    if (best == -2) {  // not tuned yet
        best = -1;
        float best_ms = 1e30f;
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        for (int i = 0; i < NC; ++i) {
            if (!runners[i](A, B, xs, ws, bias, D, M, N, K, stream)) continue;  // can't run / failed
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int r = 0; r < 3; ++r) runners[i](A, B, xs, ws, bias, D, M, N, K, stream);
            cudaEventRecord(e, stream); cudaEventSynchronize(e);
            float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
            if (ms < best_ms) { best_ms = ms; best = i; }
        }
        cudaEventDestroy(s); cudaEventDestroy(e);
        std::lock_guard<std::mutex> lk(mtx);
        cache[key] = best;
    }
    last_m = M;
    last_n = N;
    last_k = K;
    last_best = best;
    if (best < 0) return false;                              // fall back to cuBLAS
    return runners[best](A, B, xs, ws, bias, D, M, N, K, stream);  // final, correct write with best config
}

template <typename OutT>
bool dispatch_fused_no_bias(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            OutT* D, int M, int N, int K, cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, OutT*, int, int, int, cudaStream_t);
    static const Fn runners[] = {
        &FusedInt8GemmNoBias<OutT, 128, 256, 64, 64, 64, 64, 3>::run,
        &FusedInt8GemmNoBias<OutT, 128, 128, 64, 64, 64, 64, 4>::run,
        &FusedInt8GemmNoBias<OutT,  64, 128, 64, 32, 64, 64, 4>::run,
    };
    constexpr int NC = sizeof(runners) / sizeof(runners[0]);

    static std::mutex mtx;
    static std::map<std::tuple<int, int, int>, int> cache;
    const std::tuple<int, int, int> key{M, N, K};

    static thread_local int last_m = -1;
    static thread_local int last_n = -1;
    static thread_local int last_k = -1;
    static thread_local int last_best = -2;
    if (M == last_m && N == last_n && K == last_k) {
        if (last_best < 0) return false;
        return runners[last_best](A, B, xs, ws, D, M, N, K, stream);
    }

    int best;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(key);
        best = (it != cache.end()) ? it->second : -2;
    }
    if (best == -2) {
        best = -1;
        float best_ms = 1e30f;
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        for (int i = 0; i < NC; ++i) {
            if (!runners[i](A, B, xs, ws, D, M, N, K, stream)) continue;
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int r = 0; r < 3; ++r) runners[i](A, B, xs, ws, D, M, N, K, stream);
            cudaEventRecord(e, stream); cudaEventSynchronize(e);
            float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
            if (ms < best_ms) { best_ms = ms; best = i; }
        }
        cudaEventDestroy(s); cudaEventDestroy(e);
        std::lock_guard<std::mutex> lk(mtx);
        cache[key] = best;
    }
    last_m = M;
    last_n = N;
    last_k = K;
    last_best = best;
    if (best < 0) return false;
    return runners[best](A, B, xs, ws, D, M, N, K, stream);
}

template <typename OutT>
bool dispatch_fused_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                            const float* bias, OutT* D, int M, int N, int K, int output_stride,
                            cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, const float*, OutT*, int, int, int, int, cudaStream_t);
    static const Fn runners[] = {
        &FusedInt8Gemm<OutT, 128, 256, 64, 64, 64, 64, 3>::run_strided,
        &FusedInt8Gemm<OutT, 128, 128, 64, 64, 64, 64, 4>::run_strided,
        &FusedInt8Gemm<OutT,  64, 128, 64, 32, 64, 64, 4>::run_strided,
    };
    constexpr int NC = sizeof(runners) / sizeof(runners[0]);

    static std::mutex mtx;
    static std::map<std::tuple<int, int, int>, int> cache;
    const std::tuple<int, int, int> key{M, N, K};

    int best;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(key);
        best = (it != cache.end()) ? it->second : -2;
    }
    if (best == -2) {
        best = -1;
        float best_ms = 1e30f;
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        for (int i = 0; i < NC; ++i) {
            if (!runners[i](A, B, xs, ws, bias, D, M, N, K, output_stride, stream)) continue;
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int r = 0; r < 3; ++r) runners[i](A, B, xs, ws, bias, D, M, N, K, output_stride, stream);
            cudaEventRecord(e, stream); cudaEventSynchronize(e);
            float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
            if (ms < best_ms) { best_ms = ms; best = i; }
        }
        cudaEventDestroy(s); cudaEventDestroy(e);
        std::lock_guard<std::mutex> lk(mtx);
        cache[key] = best;
    }
    if (best < 0) return false;
    return runners[best](A, B, xs, ws, bias, D, M, N, K, output_stride, stream);
}

template <typename OutT>
bool dispatch_fused_no_bias_strided(const int8_t* A, const int8_t* B, const float* xs, const float* ws,
                                    OutT* D, int M, int N, int K, int output_stride,
                                    cudaStream_t stream) {
    using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, OutT*, int, int, int, int, cudaStream_t);
    static const Fn runners[] = {
        &FusedInt8GemmNoBias<OutT, 128, 256, 64, 64, 64, 64, 3>::run_strided,
        &FusedInt8GemmNoBias<OutT, 128, 128, 64, 64, 64, 64, 4>::run_strided,
        &FusedInt8GemmNoBias<OutT,  64, 128, 64, 32, 64, 64, 4>::run_strided,
    };
    constexpr int NC = sizeof(runners) / sizeof(runners[0]);

    static std::mutex mtx;
    static std::map<std::tuple<int, int, int>, int> cache;
    const std::tuple<int, int, int> key{M, N, K};

    int best;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(key);
        best = (it != cache.end()) ? it->second : -2;
    }
    if (best == -2) {
        best = -1;
        float best_ms = 1e30f;
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        for (int i = 0; i < NC; ++i) {
            if (!runners[i](A, B, xs, ws, D, M, N, K, output_stride, stream)) continue;
            cudaStreamSynchronize(stream);
            cudaEventRecord(s, stream);
            for (int r = 0; r < 3; ++r) runners[i](A, B, xs, ws, D, M, N, K, output_stride, stream);
            cudaEventRecord(e, stream); cudaEventSynchronize(e);
            float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
            if (ms < best_ms) { best_ms = ms; best = i; }
        }
        cudaEventDestroy(s); cudaEventDestroy(e);
        std::lock_guard<std::mutex> lk(mtx);
        cache[key] = best;
    }
    if (best < 0) return false;
    return runners[best](A, B, xs, ws, D, M, N, K, output_stride, stream);
}

template <typename OutT>
bool dispatch_fused_no_bias_batched(
    const int8_t* A, const int8_t* B, const float* xs, const float* ws,
    OutT* D, int groups, int M, int N, int K, cudaStream_t stream) {
    using Fn = bool (*)(
        const int8_t*, const int8_t*, const float*, const float*, OutT*,
        int, int, int, int, cudaStream_t);
    static const Fn small_m_runners[] = {
        &FusedInt8GemmNoBias<OutT,  64, 128, 64, 32, 64, 64, 4>::run_batched,
        &FusedInt8GemmNoBias<OutT, 128, 128, 64, 64, 64, 64, 4>::run_batched,
        &FusedInt8GemmNoBias<OutT, 128, 256, 64, 64, 64, 64, 3>::run_batched,
    };
    static const Fn large_m_runners[] = {
        &FusedInt8GemmNoBias<OutT, 128, 128, 64, 64, 64, 64, 4>::run_batched,
        &FusedInt8GemmNoBias<OutT, 128, 256, 64, 64, 64, 64, 3>::run_batched,
        &FusedInt8GemmNoBias<OutT,  64, 128, 64, 32, 64, 64, 4>::run_batched,
    };
    const Fn* runners = M <= 64 ? small_m_runners : large_m_runners;
    for (int index = 0; index < 3; ++index) {
        if (runners[index](A, B, xs, ws, D, groups, M, N, K, stream)) {
            return true;
        }
    }
    return false;
}
}  // namespace

namespace packed_int8 {

using PackedInt8Epilogue = cutlass::epilogue::thread::LinearCombination<
    int32_t, 4, int32_t, int32_t>;
template <int Stages>
using PackedInt8KernelT = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    int8_t,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    16,
    int8_t,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    16,
    int32_t,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    PackedInt8Epilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    Stages,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
using PackedInt8Kernel = PackedInt8KernelT<4>;
using PackedInt8Gemm = cutlass::gemm::device::GemmGrouped<PackedInt8Kernel>;
using PackedInt8GemmStage3 = cutlass::gemm::device::GemmGrouped<PackedInt8KernelT<3>>;

constexpr size_t kPackedWorkspaceAlignment = 16;

size_t align_packed_workspace(size_t offset) {
    return (offset + kPackedWorkspaceAlignment - 1) & ~(kPackedWorkspaceAlignment - 1);
}

template <typename T>
size_t append_packed_workspace(size_t offset, size_t count) {
    return align_packed_workspace(offset) + sizeof(T) * count;
}

size_t packed_grouped_int8_workspace_size(int64_t groups, int64_t rows) {
    if (groups < 0 || rows < 0) {
        return 0;
    }
    size_t offset = 0;
    offset = append_packed_workspace<cutlass::gemm::GemmCoord>(offset, groups);
    offset = append_packed_workspace<int8_t*>(offset, groups);
    offset = append_packed_workspace<int8_t*>(offset, groups);
    offset = append_packed_workspace<int32_t*>(offset, groups);
    offset = append_packed_workspace<int32_t*>(offset, groups);
    offset = append_packed_workspace<int64_t>(offset, groups);
    offset = append_packed_workspace<int64_t>(offset, groups);
    offset = append_packed_workspace<int64_t>(offset, groups);
    offset = append_packed_workspace<int64_t>(offset, groups);
    offset = append_packed_workspace<int32_t>(offset, rows);
    return align_packed_workspace(offset);
}

class PackedWorkspaceArena {
public:
    PackedWorkspaceArena(void* workspace, size_t workspace_size)
        : base_(static_cast<uint8_t*>(workspace)), size_(workspace_size) {}

    template <typename T>
    T* allocate(size_t count) {
        offset_ = align_packed_workspace(offset_);
        const size_t bytes = sizeof(T) * count;
        if (offset_ > size_ || bytes > size_ - offset_) {
            return nullptr;
        }
        T* result = reinterpret_cast<T*>(base_ + offset_);
        offset_ += bytes;
        return result;
    }

private:
    uint8_t* base_;
    size_t size_;
    size_t offset_{0};
};

__global__ void prepare_packed_grouped_int8_args(
    int8_t* activations,
    int8_t* weights,
    int32_t* accumulator,
    const int32_t* expert_indptr,
    int32_t* row_expert,
    int groups,
    int rows,
    int n,
    int k,
    cutlass::gemm::GemmCoord* problem_sizes,
    int8_t** activation_ptrs,
    int8_t** weight_ptrs,
    int32_t** accumulator_c_ptrs,
    int32_t** accumulator_d_ptrs,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc,
    int64_t* ldd) {
    const int expert = blockIdx.x;
    if (expert >= groups) {
        return;
    }

    const int32_t start = expert_indptr[expert];
    const int32_t end = expert_indptr[expert + 1];
    if (threadIdx.x == 0) {
        problem_sizes[expert] = cutlass::gemm::GemmCoord(end - start, n, k);
        activation_ptrs[expert] = activations + static_cast<size_t>(start) * k;
        weight_ptrs[expert] = weights + static_cast<size_t>(expert) * n * k;
        accumulator_c_ptrs[expert] = accumulator + static_cast<size_t>(start) * n;
        accumulator_d_ptrs[expert] = accumulator + static_cast<size_t>(start) * n;
        lda[expert] = k;
        ldb[expert] = k;
        ldc[expert] = n;
        ldd[expert] = n;
    }
    for (int row = start + threadIdx.x; row < end && row < rows; row += blockDim.x) {
        row_expert[row] = expert;
    }
}

template <typename ElementOutput>
__global__ void dequantize_packed_grouped_int8(
    const int32_t* accumulator,
    const float* activation_scales,
    const float* weight_scales,
    const int32_t* row_expert,
    ElementOutput* output,
    int64_t elements,
    int n) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int row = static_cast<int>(index / n);
    const int column = static_cast<int>(index - static_cast<int64_t>(row) * n);
    const int expert = row_expert[row];
    float value = static_cast<float>(accumulator[index]) * activation_scales[row];
    value = value * weight_scales[static_cast<size_t>(expert) * n + column];
    output[index] = static_cast<ElementOutput>(value);
}

template <typename ElementOutput>
__global__ void dequantize_packed_grouped_int8_vec4(
    const int32_t* accumulator,
    const float* activation_scales,
    const float* weight_scales,
    const int32_t* row_expert,
    ElementOutput* output,
    int64_t vectors,
    int n,
    int n4) {
    const int64_t index4 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index4 >= vectors) {
        return;
    }
    const int row = static_cast<int>(index4 / n4);
    const int column4 = static_cast<int>(index4 - static_cast<int64_t>(row) * n4);
    const int expert = row_expert[row];
    const float activation_scale = activation_scales[row];
    const int4 values = reinterpret_cast<const int4*>(accumulator)[index4];
    const float4 scales = reinterpret_cast<const float4*>(
        weight_scales + static_cast<size_t>(expert) * n)[column4];
    const int64_t index = index4 * 4;
    output[index] = static_cast<ElementOutput>(
        static_cast<float>(values.x) * activation_scale * scales.x);
    output[index + 1] = static_cast<ElementOutput>(
        static_cast<float>(values.y) * activation_scale * scales.y);
    output[index + 2] = static_cast<ElementOutput>(
        static_cast<float>(values.z) * activation_scale * scales.z);
    output[index + 3] = static_cast<ElementOutput>(
        static_cast<float>(values.w) * activation_scale * scales.w);
}

template <typename ElementOutput>
bool run_packed_grouped_int8(
    const void* activations_raw,
    const void* weights_raw,
    const void* activation_scales_raw,
    const void* weight_scales_raw,
    const int32_t* expert_indptr,
    void* accumulator_raw,
    void* output_raw,
    int groups,
    int rows,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
    if (rows == 0) {
        return true;
    }
    if (groups <= 0 || n <= 0 || k <= 0) {
        return false;
    }

    PackedWorkspaceArena arena(workspace, workspace_size);
    auto problem_sizes = arena.allocate<cutlass::gemm::GemmCoord>(groups);
    auto activation_ptrs = arena.allocate<int8_t*>(groups);
    auto weight_ptrs = arena.allocate<int8_t*>(groups);
    auto accumulator_c_ptrs = arena.allocate<int32_t*>(groups);
    auto accumulator_d_ptrs = arena.allocate<int32_t*>(groups);
    auto lda = arena.allocate<int64_t>(groups);
    auto ldb = arena.allocate<int64_t>(groups);
    auto ldc = arena.allocate<int64_t>(groups);
    auto ldd = arena.allocate<int64_t>(groups);
    auto row_expert = arena.allocate<int32_t>(rows);
    if (problem_sizes == nullptr || activation_ptrs == nullptr || weight_ptrs == nullptr ||
        accumulator_c_ptrs == nullptr || accumulator_d_ptrs == nullptr || lda == nullptr ||
        ldb == nullptr || ldc == nullptr || ldd == nullptr || row_expert == nullptr) {
        return false;
    }

    constexpr int prepare_threads = 128;
    prepare_packed_grouped_int8_args<<<groups, prepare_threads, 0, stream>>>(
        const_cast<int8_t*>(static_cast<const int8_t*>(activations_raw)),
        const_cast<int8_t*>(static_cast<const int8_t*>(weights_raw)),
        static_cast<int32_t*>(accumulator_raw),
        expert_indptr,
        row_expert,
        groups,
        rows,
        n,
        k,
        problem_sizes,
        activation_ptrs,
        weight_ptrs,
        accumulator_c_ptrs,
        accumulator_d_ptrs,
        lda,
        ldb,
        ldc,
        ldd);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    constexpr int threadblock_count = 256;
    const bool gate_up_shape = groups == 128 && n == 1408 && k == 2816;
    if (gate_up_shape) {
        typename PackedInt8GemmStage3::EpilogueOutputOp::Params epilogue(1, 0);
        typename PackedInt8GemmStage3::Arguments arguments(
            problem_sizes,
            groups,
            threadblock_count,
            epilogue,
            activation_ptrs,
            weight_ptrs,
            accumulator_c_ptrs,
            accumulator_d_ptrs,
            lda,
            ldb,
            ldc,
            ldd,
            nullptr);
        PackedInt8GemmStage3 gemm;
        if (gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess ||
            gemm.run(stream) != cutlass::Status::kSuccess) {
            return false;
        }
    } else {
        typename PackedInt8Gemm::EpilogueOutputOp::Params epilogue(1, 0);
        typename PackedInt8Gemm::Arguments arguments(
            problem_sizes,
            groups,
            threadblock_count,
            epilogue,
            activation_ptrs,
            weight_ptrs,
            accumulator_c_ptrs,
            accumulator_d_ptrs,
            lda,
            ldb,
            ldc,
            ldd,
            nullptr);
        PackedInt8Gemm gemm;
        if (gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess ||
            gemm.run(stream) != cutlass::Status::kSuccess) {
            return false;
        }
    }

    constexpr int dequant_threads = 256;
    const int64_t elements = static_cast<int64_t>(rows) * n;
    if ((n & 3) == 0) {
        const int n4 = n / 4;
        const int64_t vectors = elements / 4;
        const int blocks = static_cast<int>((vectors + dequant_threads - 1) / dequant_threads);
        dequantize_packed_grouped_int8_vec4<<<blocks, dequant_threads, 0, stream>>>(
            static_cast<const int32_t*>(accumulator_raw),
            static_cast<const float*>(activation_scales_raw),
            static_cast<const float*>(weight_scales_raw),
            row_expert,
            static_cast<ElementOutput*>(output_raw),
            vectors,
            n,
            n4);
    } else {
        const int blocks = static_cast<int>((elements + dequant_threads - 1) / dequant_threads);
        dequantize_packed_grouped_int8<<<blocks, dequant_threads, 0, stream>>>(
            static_cast<const int32_t*>(accumulator_raw),
            static_cast<const float*>(activation_scales_raw),
            static_cast<const float*>(weight_scales_raw),
            row_expert,
            static_cast<ElementOutput*>(output_raw),
            elements,
            n);
    }
    return cudaGetLastError() == cudaSuccess;
}
}  // namespace packed_int8

namespace int8_routes {
constexpr int kThreads = 256;

__global__ void count_and_rank(
    const int64_t* expert_ids,
    int32_t* counts,
    int32_t* route_rank,
    int routes,
    int num_experts) {
    for (int route = blockIdx.x * blockDim.x + threadIdx.x;
         route < routes;
         route += blockDim.x * gridDim.x) {
        const int64_t expert = expert_ids[route];
        route_rank[route] = expert >= 0 && expert < num_experts
            ? atomicAdd(counts + expert, 1)
            : -1;
    }
}

__global__ void prefix_and_place(
    const int64_t* expert_ids,
    const int32_t* counts,
    const int32_t* route_rank,
    int32_t* expert_indptr,
    int64_t* route_order,
    int64_t* route_dest,
    int routes,
    int num_experts) {
    if (threadIdx.x == 0) {
        int32_t running = 0;
        for (int expert = 0; expert < num_experts; ++expert) {
            expert_indptr[expert] = running;
            running += counts[expert];
        }
        expert_indptr[num_experts] = running;
    }
    __syncthreads();

    for (int route = threadIdx.x; route < routes; route += blockDim.x) {
        const int64_t expert = expert_ids[route];
        const int32_t rank = route_rank[route];
        const int64_t destination = expert >= 0 && expert < num_experts && rank >= 0
            ? static_cast<int64_t>(expert_indptr[expert]) + rank
            : -1;
        route_dest[route] = destination;
        if (destination >= 0) {
            route_order[destination] = route;
        }
    }
}

bool prepare(
    const int64_t* expert_ids,
    int32_t* counts,
    int32_t* route_rank,
    int32_t* expert_indptr,
    int64_t* route_order,
    int64_t* route_dest,
    int routes,
    int num_experts,
    cudaStream_t stream) {
    if (cudaMemsetAsync(counts, 0, static_cast<size_t>(num_experts) * sizeof(int32_t), stream) !=
        cudaSuccess) {
        return false;
    }
    const int blocks = (routes + kThreads - 1) / kThreads;
    count_and_rank<<<blocks, kThreads, 0, stream>>>(
        expert_ids, counts, route_rank, routes, num_experts);
    prefix_and_place<<<1, kThreads, 0, stream>>>(
        expert_ids, counts, route_rank, expert_indptr, route_order, route_dest,
        routes, num_experts);
    return cudaPeekAtLastError() == cudaSuccess;
}
}  // namespace int8_routes

extern "C" {
bool launch_cutlass_int8_dequant(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int out_dtype_code, cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    const float* bs = static_cast<const float*>(bias);
    if (bs == nullptr) {
        switch (out_dtype_code) {
            case 0: return dispatch_fused_no_bias<float>(a, b, x, w, static_cast<float*>(D), M, N, K, stream);
            case 1: return dispatch_fused_no_bias<cutlass::half_t>(a, b, x, w, static_cast<cutlass::half_t*>(D), M, N, K, stream);
            case 2: return dispatch_fused_no_bias<cutlass::bfloat16_t>(a, b, x, w, static_cast<cutlass::bfloat16_t*>(D), M, N, K, stream);
            default: return false;
        }
    }
    switch (out_dtype_code) {
        case 0: return dispatch_fused<float>(a, b, x, w, bs, static_cast<float*>(D), M, N, K, stream);
        case 1: return dispatch_fused<cutlass::half_t>(a, b, x, w, bs, static_cast<cutlass::half_t*>(D), M, N, K, stream);
        case 2: return dispatch_fused<cutlass::bfloat16_t>(a, b, x, w, bs, static_cast<cutlass::bfloat16_t*>(D), M, N, K, stream);
        default: return false;
    }
}

bool launch_cutlass_int8_dequant_strided(
    const void* A, const void* B, const void* xs, const void* ws, const void* bias,
    void* D, int64_t M, int64_t N, int64_t K, int64_t output_stride, int out_dtype_code,
    cudaStream_t stream)
{
    if (M == 0 || N == 0 || K == 0) return true;
    if (output_stride < N) return false;
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    const float* bs = static_cast<const float*>(bias);
    if (bs == nullptr) {
        switch (out_dtype_code) {
            case 0: return dispatch_fused_no_bias_strided<float>(a, b, x, w, static_cast<float*>(D), M, N, K, output_stride, stream);
            case 1: return dispatch_fused_no_bias_strided<cutlass::half_t>(a, b, x, w, static_cast<cutlass::half_t*>(D), M, N, K, output_stride, stream);
            case 2: return dispatch_fused_no_bias_strided<cutlass::bfloat16_t>(a, b, x, w, static_cast<cutlass::bfloat16_t*>(D), M, N, K, output_stride, stream);
            default: return false;
        }
    }
    switch (out_dtype_code) {
        case 0: return dispatch_fused_strided<float>(a, b, x, w, bs, static_cast<float*>(D), M, N, K, output_stride, stream);
        case 1: return dispatch_fused_strided<cutlass::half_t>(a, b, x, w, bs, static_cast<cutlass::half_t*>(D), M, N, K, output_stride, stream);
        case 2: return dispatch_fused_strided<cutlass::bfloat16_t>(a, b, x, w, bs, static_cast<cutlass::bfloat16_t*>(D), M, N, K, output_stride, stream);
        default: return false;
    }
}

bool launch_cutlass_grouped_int8_dequant(
    const void* A, const void* B, const void* xs, const void* ws, void* D,
    int64_t groups, int64_t M, int64_t N, int64_t K, int out_dtype_code,
    cudaStream_t stream) {
    if (groups == 0 || M == 0 || N == 0 || K == 0) return true;
    if (groups < 0 || M < 0 || N < 0 || K < 0 ||
        groups > INT32_MAX || M > INT32_MAX || N > INT32_MAX || K > INT32_MAX) {
        return false;
    }
    const int8_t* a = static_cast<const int8_t*>(A);
    const int8_t* b = static_cast<const int8_t*>(B);
    const float* x = static_cast<const float*>(xs);
    const float* w = static_cast<const float*>(ws);
    switch (out_dtype_code) {
        case 0:
            return dispatch_fused_no_bias_batched<float>(
                a, b, x, w, static_cast<float*>(D), groups, M, N, K, stream);
        case 1:
            return dispatch_fused_no_bias_batched<cutlass::half_t>(
                a, b, x, w, static_cast<cutlass::half_t*>(D), groups, M, N, K, stream);
        case 2:
            return dispatch_fused_no_bias_batched<cutlass::bfloat16_t>(
                a, b, x, w, static_cast<cutlass::bfloat16_t*>(D), groups, M, N, K, stream);
        default:
            return false;
    }
}

size_t cutlass_grouped_int8_dequant_packed_workspace_size(int64_t groups, int64_t rows) {
    return packed_int8::packed_grouped_int8_workspace_size(groups, rows);
}

bool launch_cutlass_grouped_int8_dequant_packed(
    const void* activations,
    const void* weights,
    const void* activation_scales,
    const void* weight_scales,
    const int32_t* expert_indptr,
    void* accumulator,
    void* output,
    int64_t groups,
    int64_t rows,
    int64_t n,
    int64_t k,
    void* workspace,
    size_t workspace_size,
    int out_dtype_code,
    cudaStream_t stream) {
    if (groups < 0 || rows < 0 || n < 0 || k < 0 || groups > INT32_MAX ||
        rows > INT32_MAX || n > INT32_MAX || k > INT32_MAX) {
        return false;
    }
    if (workspace_size < packed_int8::packed_grouped_int8_workspace_size(groups, rows)) {
        return false;
    }
    switch (out_dtype_code) {
        case 0:
            return packed_int8::run_packed_grouped_int8<float>(
                activations, weights, activation_scales, weight_scales, expert_indptr,
                accumulator, output, groups, rows, n, k, workspace, workspace_size, stream);
        case 1:
            return packed_int8::run_packed_grouped_int8<cutlass::half_t>(
                activations, weights, activation_scales, weight_scales, expert_indptr,
                accumulator, output, groups, rows, n, k, workspace, workspace_size, stream);
        case 2:
            return packed_int8::run_packed_grouped_int8<cutlass::bfloat16_t>(
                activations, weights, activation_scales, weight_scales, expert_indptr,
                accumulator, output, groups, rows, n, k, workspace, workspace_size, stream);
        default:
            return false;
    }
}

bool launch_prepare_int8_moe_routes(
    const int64_t* expert_ids,
    int32_t* counts,
    int32_t* route_rank,
    int32_t* expert_indptr,
    int64_t* route_order,
    int64_t* route_dest,
    int64_t routes,
    int64_t num_experts,
    cudaStream_t stream) {
    if (routes <= 0 || num_experts <= 0 || routes > INT_MAX || num_experts > INT_MAX) {
        return false;
    }
    return int8_routes::prepare(
        expert_ids, counts, route_rank, expert_indptr, route_order, route_dest,
        static_cast<int>(routes), static_cast<int>(num_experts), stream);
}
}  // extern "C"

#else  // !COMFY_HAVE_CUTLASS -- stub; caller falls back to cuBLAS + separate dequant.

extern "C" bool launch_cutlass_int8_dequant(
    const void*, const void*, const void*, const void*, const void*,
    void*, int64_t, int64_t, int64_t, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_int8_dequant_strided(
    const void*, const void*, const void*, const void*, const void*,
    void*, int64_t, int64_t, int64_t, int64_t, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_cutlass_grouped_int8_dequant(
    const void*, const void*, const void*, const void*, void*,
    int64_t, int64_t, int64_t, int64_t, int, cudaStream_t) {
    return false;
}

extern "C" size_t cutlass_grouped_int8_dequant_packed_workspace_size(int64_t, int64_t) {
    return 0;
}

extern "C" bool launch_cutlass_grouped_int8_dequant_packed(
    const void*, const void*, const void*, const void*, const int32_t*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, void*, size_t, int, cudaStream_t) {
    return false;
}

extern "C" bool launch_prepare_int8_moe_routes(
    const int64_t*, int32_t*, int32_t*, int32_t*, int64_t*, int64_t*,
    int64_t, int64_t, cudaStream_t) {
    return false;
}

#endif
