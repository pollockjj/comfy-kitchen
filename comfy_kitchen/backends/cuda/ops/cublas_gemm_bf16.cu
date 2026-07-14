/*
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../cublaslt_runtime.h"

#define CUBLAS_CHECK(call)                                                        \
  do {                                                                            \
    const cublasStatus_t status = (call);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                         \
      throw std::runtime_error("cuBLASLt error: " + std::to_string(status));      \
    }                                                                             \
  } while (0)

namespace {

thread_local cublasLtHandle_t bf16_handle = nullptr;
thread_local cublasHandle_t bf16_gemm_ex_handle = nullptr;

cublasLtHandle_t get_handle() {
  auto& runtime = comfy::CublasLtRuntime::instance();
  if (!runtime.is_available()) {
    throw std::runtime_error("cuBLASLt unavailable: " + runtime.error_message());
  }
  if (bf16_handle == nullptr) {
    CUBLAS_CHECK(runtime.cublasLtCreate(&bf16_handle));
  }
  return bf16_handle;
}

cublasHandle_t get_gemm_ex_handle() {
  if (bf16_gemm_ex_handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&bf16_gemm_ex_handle));
  }
  return bf16_gemm_ex_handle;
}

struct Bf16Plan {
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  int64_t workspace_size = 0;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t input = nullptr;
  cublasLtMatrixLayout_t output = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> algorithms;

  Bf16Plan(int64_t m, int64_t n, int64_t k, int64_t workspace_bytes)
      : M(m), N(n), K(k), workspace_size(workspace_bytes) {
    auto& runtime = comfy::CublasLtRuntime::instance();
    const cublasOperation_t trans_weight = CUBLAS_OP_T;
    const cublasOperation_t trans_input = CUBLAS_OP_N;

    CUBLAS_CHECK(runtime.cublasLtMatmulDescCreate(
        &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(runtime.cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_weight, sizeof(trans_weight)));
    CUBLAS_CHECK(runtime.cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_input, sizeof(trans_input)));

    CUBLAS_CHECK(runtime.cublasLtMatrixLayoutCreate(
        &weight, CUDA_R_16BF, K, N, K));
    CUBLAS_CHECK(runtime.cublasLtMatrixLayoutCreate(
        &input, CUDA_R_16BF, K, M, K));
    CUBLAS_CHECK(runtime.cublasLtMatrixLayoutCreate(
        &output, CUDA_R_16BF, N, M, N));

    cublasLtMatmulPreference_t preference = nullptr;
    CUBLAS_CHECK(runtime.cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(runtime.cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    algorithms.resize(32);
    int returned = 0;
    CUBLAS_CHECK(runtime.cublasLtMatmulAlgoGetHeuristic(
        get_handle(), operation, weight, input, output, output, preference,
        static_cast<int>(algorithms.size()), algorithms.data(), &returned));
    CUBLAS_CHECK(runtime.cublasLtMatmulPreferenceDestroy(preference));
    algorithms.resize(returned);
    if (algorithms.empty()) {
      throw std::runtime_error("cuBLASLt returned no BF16 linear algorithms");
    }
  }

  ~Bf16Plan() {
    auto& runtime = comfy::CublasLtRuntime::instance();
    if (output != nullptr) runtime.cublasLtMatrixLayoutDestroy(output);
    if (input != nullptr) runtime.cublasLtMatrixLayoutDestroy(input);
    if (weight != nullptr) runtime.cublasLtMatrixLayoutDestroy(weight);
    if (operation != nullptr) runtime.cublasLtMatmulDescDestroy(operation);
  }
};

thread_local std::unique_ptr<Bf16Plan> bf16_plan;

Bf16Plan& get_plan(int64_t M, int64_t N, int64_t K, int64_t workspace_size) {
  if (!bf16_plan || bf16_plan->M != M || bf16_plan->N != N ||
      bf16_plan->K != K || bf16_plan->workspace_size != workspace_size) {
    bf16_plan = std::make_unique<Bf16Plan>(M, N, K, workspace_size);
  }
  return *bf16_plan;
}

}  // namespace

extern "C" int bf16_cublaslt_algorithm_count(
    int64_t M, int64_t N, int64_t K, int64_t workspace_size) {
  return static_cast<int>(get_plan(M, N, K, workspace_size).algorithms.size());
}

extern "C" void launch_bf16_cublaslt_linear(
    const void* input,
    const void* weight,
    void* output,
    int64_t M,
    int64_t N,
    int64_t K,
    int algorithm_index,
    void* workspace,
    int64_t workspace_size,
    cudaStream_t stream) {
  Bf16Plan& plan = get_plan(M, N, K, workspace_size);
  if (algorithm_index < 0 || algorithm_index >= static_cast<int>(plan.algorithms.size())) {
    throw std::runtime_error("BF16 cuBLASLt algorithm index out of range");
  }
  auto& runtime = comfy::CublasLtRuntime::instance();
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(runtime.cublasLtMatmul(
      get_handle(), plan.operation, &alpha, weight, plan.weight, input, plan.input,
      &beta, output, plan.output, output, plan.output,
      &plan.algorithms[algorithm_index].algo, workspace,
      static_cast<size_t>(workspace_size), stream));
}

extern "C" void launch_bf16_cublas_gemm_ex(
    const void* input,
    const void* weight,
    void* output,
    int64_t M,
    int64_t N,
    int64_t K,
    int algorithm,
    cudaStream_t stream) {
  cublasHandle_t handle = get_gemm_ex_handle();
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N,
      static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
      &alpha, weight, CUDA_R_16BF, static_cast<int>(K),
      input, CUDA_R_16BF, static_cast<int>(K),
      &beta, output, CUDA_R_16BF, static_cast<int>(N),
      CUBLAS_COMPUTE_32F, static_cast<cublasGemmAlgo_t>(algorithm)));
}
