#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "../cublaslt_runtime.h"

#define CUBLAS_CHECK(call) \
  do { \
    const cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      throw std::runtime_error(std::string("cuBLASLt BF16 error: ") + std::to_string(status)); \
    } \
  } while (0)

namespace comfy {
namespace {

thread_local cublasLtHandle_t bf16_handle = nullptr;

struct Bf16Plan {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t input = nullptr;
  cublasLtMatrixLayout_t output = nullptr;
  cublasLtMatmulAlgo_t algo = {};
  bool initialized = false;
};

thread_local Bf16Plan gate_up_plan;

cublasLtHandle_t get_handle() {
  auto& runtime = CublasLtRuntime::instance();
  if (!runtime.is_available()) {
    throw std::runtime_error("cuBLASLt unavailable: " + runtime.error_message());
  }
  if (bf16_handle == nullptr) {
    CUBLAS_CHECK(runtime.cublasLtCreate(&bf16_handle));
  }
  return bf16_handle;
}

Bf16Plan& get_gate_up_plan() {
  if (gate_up_plan.initialized) {
    return gate_up_plan;
  }

  constexpr int64_t M = 3;
  constexpr int64_t N = 20480;
  constexpr int64_t K = 2560;
  auto& runtime = CublasLtRuntime::instance();
  cublasLtHandle_t handle = get_handle();
  auto& plan = gate_up_plan;

  CUBLAS_CHECK(runtime.cublasLtMatmulDescCreate(&plan.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans_weight = CUBLAS_OP_T;
  const cublasOperation_t trans_input = CUBLAS_OP_N;
  CUBLAS_CHECK(runtime.cublasLtMatmulDescSetAttribute(
      plan.operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_weight, sizeof(trans_weight)));
  CUBLAS_CHECK(runtime.cublasLtMatmulDescSetAttribute(
      plan.operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_input, sizeof(trans_input)));

  CUBLAS_CHECK(runtime.cublasLtMatrixLayoutCreate(&plan.weight, CUDA_R_16BF, K, N, K));
  CUBLAS_CHECK(runtime.cublasLtMatrixLayoutCreate(&plan.input, CUDA_R_16BF, K, M, K));
  CUBLAS_CHECK(runtime.cublasLtMatrixLayoutCreate(&plan.output, CUDA_R_16BF, N, M, N));

  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(runtime.cublasLtMatmulPreferenceCreate(&preference));
  size_t workspace_bytes = 0;
  CUBLAS_CHECK(runtime.cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes, sizeof(workspace_bytes)));

  cublasLtMatmulHeuristicResult_t results[16] = {};
  int returned = 0;
  CUBLAS_CHECK(runtime.cublasLtMatmulAlgoGetHeuristic(
      handle, plan.operation, plan.weight, plan.input, plan.output, plan.output,
      preference, 16, results, &returned));
  CUBLAS_CHECK(runtime.cublasLtMatmulPreferenceDestroy(preference));

  bool found = false;
  for (int i = 0; i < returned; ++i) {
    int algo_id = -1;
    int tile_id = -1;
    int split_k = -1;
    int stages = -1;
    size_t written = 0;
    const auto& algo = results[i].algo;
    CUBLAS_CHECK(runtime.cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &written));
    CUBLAS_CHECK(runtime.cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id, sizeof(tile_id), &written));
    CUBLAS_CHECK(runtime.cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k, sizeof(split_k), &written));
    CUBLAS_CHECK(runtime.cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), &written));
    if (algo_id == 21 && tile_id == 5 && split_k == 1 && stages == 19 &&
        results[i].workspaceSize == 0 && results[i].state == CUBLAS_STATUS_SUCCESS) {
      plan.algo = algo;
      found = true;
      break;
    }
  }
  if (!found) {
    throw std::runtime_error("exact tuned BF16 E4B gate/up cuBLASLt algorithm unavailable");
  }
  plan.initialized = true;
  return plan;
}

void tuned_gate_up(
    const void* input,
    const void* weight,
    void* output,
    cudaStream_t stream) {
  auto& runtime = CublasLtRuntime::instance();
  auto& plan = get_gate_up_plan();
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(runtime.cublasLtMatmul(
      get_handle(), plan.operation, &alpha,
      weight, plan.weight,
      input, plan.input,
      &beta,
      output, plan.output,
      output, plan.output,
      &plan.algo, nullptr, 0, stream));
}

}  // namespace
}  // namespace comfy

extern "C" void launch_bf16_tuned_gate_up_linear(
    const void* input,
    const void* weight,
    void* output,
    int64_t M,
    int64_t N,
    int64_t K,
    cudaStream_t stream) {
  if (M != 3 || N != 20480 || K != 2560) {
    throw std::runtime_error("tuned BF16 gate/up requires M=3, N=20480, K=2560");
  }
  comfy::tuned_gate_up(input, weight, output, stream);
}
