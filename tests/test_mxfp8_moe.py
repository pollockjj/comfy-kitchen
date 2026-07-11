import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck

cuda_status = ck.list_backends().get("cuda", {})
sm120_grouped_mxfp8_available = (
    hasattr(torch, "float8_e8m0fnu")
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability(0) == (12, 0)
    and "grouped_scaled_mm_mxfp8" in set(cuda_status.get("capabilities", ()))
)
sm120_fused_mxfp8_available = (
    sm120_grouped_mxfp8_available
    and "fused_moe_mxfp8" in set(cuda_status.get("capabilities", ()))
)

NUM_EXPERTS = 128
HIDDEN_SIZE = 2816
INTERMEDIATE_SIZE = 704
TOP_K = 8


def _quantize_expert_templates(templates):
    quantized = [ck.quantize_mxfp8(template) for template in templates]
    qdata = torch.stack([item[0] for item in quantized]).repeat(64, 1, 1)
    block_scales = torch.stack([item[1] for item in quantized]).repeat(64, 1, 1)
    return qdata, block_scales


@pytest.fixture(scope="module")
def fused_expert_bank():
    generator = torch.Generator(device="cuda:0").manual_seed(5770521)
    with ck.use_backend("cuda"):
        fc1 = torch.randn(
            2,
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda:0",
        ).mul_(2**-6)
        fc2 = torch.randn(
            2,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda:0",
        ).mul_(2**-6)
        fc1_qdata, fc1_block_scales = _quantize_expert_templates(fc1)
        fc2_qdata, fc2_block_scales = _quantize_expert_templates(fc2)
    return fc1_qdata, fc1_block_scales, fc2_qdata, fc2_block_scales


def _fused_inputs(dtype):
    generator = torch.Generator(device="cuda:0").manual_seed(5770777)
    x = torch.randn(
        256,
        HIDDEN_SIZE,
        generator=generator,
        dtype=dtype,
        device="cuda:0",
    )
    logits = torch.randn(
        256,
        NUM_EXPERTS,
        generator=generator,
        dtype=torch.float32,
        device="cuda:0",
    )
    probabilities = torch.softmax(logits, dim=-1)
    router_weights, expert_ids = torch.topk(probabilities, TOP_K, dim=-1)
    router_weights.div_(router_weights.sum(dim=-1, keepdim=True))
    return x, expert_ids.contiguous(), router_weights.contiguous()


def _grouped_mxfp8_reference(x, expert_ids, router_weights, expert_bank):
    fc1_qdata, fc1_block_scales, fc2_qdata, fc2_block_scales = expert_bank
    num_tokens = x.shape[0]
    flat_experts = expert_ids.reshape(-1)
    counts = torch.bincount(flat_experts, minlength=NUM_EXPERTS)
    group_size = -(-int(counts.max()) // 128) * 128
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    rank = torch.arange(num_tokens * TOP_K, device=x.device) - (
        counts.cumsum(0) - counts
    )[sorted_experts]
    slot = sorted_experts * group_size + rank
    gather_tokens = torch.zeros(NUM_EXPERTS * group_size, dtype=torch.long, device=x.device)
    gather_tokens[slot] = order // TOP_K

    with ck.use_backend("cuda"):
        qx, x_block_scales = ck.quantize_mxfp8(x[gather_tokens])
        gate_up = ck.grouped_scaled_mm_mxfp8(
            qx,
            fc1_qdata,
            x_block_scales,
            fc1_block_scales,
            group_size,
            out_dtype=x.dtype,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = functional.gelu(gate, approximate="tanh") * up
        qi, intermediate_block_scales = ck.quantize_mxfp8(intermediate.flatten(0, 1))
        routed_output = ck.grouped_scaled_mm_mxfp8(
            qi,
            fc2_qdata,
            intermediate_block_scales,
            fc2_block_scales,
            group_size,
            out_dtype=x.dtype,
        )

    pair_order = torch.empty(num_tokens * TOP_K, dtype=torch.long, device=x.device)
    pair_order[order] = slot
    routed_output = routed_output.reshape(NUM_EXPERTS * group_size, HIDDEN_SIZE)[pair_order]
    return (
        routed_output.float()
        .mul_(router_weights.reshape(-1, 1))
        .view(num_tokens, TOP_K, HIDDEN_SIZE)
        .sum(dim=1)
        .to(x.dtype)
    )


@pytest.mark.skipif(not sm120_grouped_mxfp8_available, reason="SM120 grouped MXFP8 required")
def test_grouped_scaled_mm_mxfp8_matches_scalar_mxfp8():
    generator = torch.Generator(device="cuda:0").manual_seed(5770521)
    groups, group_size, k, n = 2, 128, 128, 128
    x = torch.randn(
        groups * group_size,
        k,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda:0",
    ).mul_(2**-3)
    weights = torch.randn(
        groups,
        n,
        k,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda:0",
    ).mul_(2**-3)

    with ck.use_backend("cuda"):
        qx, x_scales = ck.quantize_mxfp8(x)
        quantized_weights = [ck.quantize_mxfp8(weight) for weight in weights]
        qw = torch.stack([item[0] for item in quantized_weights])
        weight_scales = torch.stack([item[1] for item in quantized_weights])
        candidate = ck.grouped_scaled_mm_mxfp8(
            qx,
            qw,
            x_scales,
            weight_scales,
            group_size,
        )

    reference = torch.stack(
        [
            ck.scaled_mm_mxfp8(
                qx[index * group_size:(index + 1) * group_size],
                qw[index],
                x_scales[index * group_size:(index + 1) * group_size],
                weight_scales[index],
            )
            for index in range(groups)
        ]
    )
    delta = candidate.float() - reference.float()
    relative_rmse = delta.square().mean().sqrt() / reference.float().square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        candidate.float().flatten(1), reference.float().flatten(1), dim=-1
    ).mean()

    assert torch.isfinite(candidate).all()
    assert delta.abs().max() <= 0.0625
    assert relative_rmse <= 1.0e-3
    assert cosine >= 0.99999


@pytest.mark.skipif(not sm120_grouped_mxfp8_available, reason="SM120 grouped MXFP8 required")
def test_grouped_scaled_mm_mxfp8_rejects_unaligned_group_size():
    qdata = torch.empty((64, 32), dtype=torch.float8_e4m3fn, device="cuda:0")
    scales = torch.empty((128, 4), dtype=torch.float8_e8m0fnu, device="cuda:0")
    weights = torch.empty((1, 128, 32), dtype=torch.float8_e4m3fn, device="cuda:0")
    weight_scales = torch.empty((1, 128, 4), dtype=torch.float8_e8m0fnu, device="cuda:0")

    with pytest.raises(ValueError, match="positive multiple of 128"):
        ck.grouped_scaled_mm_mxfp8(qdata, weights, scales, weight_scales, 64)


def test_grouped_scaled_mm_mxfp8_has_no_eager_fallback():
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("PyTorch does not expose E8M0")
    qdata = torch.empty((128, 32), dtype=torch.float8_e4m3fn)
    scales = torch.empty((128, 4), dtype=torch.float8_e8m0fnu)
    weights = torch.empty((1, 128, 32), dtype=torch.float8_e4m3fn)
    weight_scales = torch.empty((1, 128, 4), dtype=torch.float8_e8m0fnu)

    with ck.use_backend("eager"), pytest.raises(ck.NoCapableBackendError):
        ck.grouped_scaled_mm_mxfp8(qdata, weights, scales, weight_scales, 128)


@pytest.mark.skipif(not sm120_fused_mxfp8_available, reason="SM120 fused MXFP8 required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_moe_mxfp8_matches_grouped_reference(dtype, fused_expert_bank):
    x, expert_ids, router_weights = _fused_inputs(dtype)
    candidate = ck.fused_moe_mxfp8(
        x, expert_ids, router_weights, *fused_expert_bank
    )
    reference = _grouped_mxfp8_reference(
        x, expert_ids, router_weights, fused_expert_bank
    )
    delta = candidate.float() - reference.float()
    relative_rmse = delta.square().mean().sqrt() / reference.float().square().mean().sqrt()
    cosine = functional.cosine_similarity(candidate.float(), reference.float(), dim=-1).mean()

    assert torch.isfinite(candidate).all()
    assert delta.abs().max() <= 0.0625
    assert relative_rmse <= 1.0e-3
    assert cosine >= 0.99999


@pytest.mark.skipif(not sm120_fused_mxfp8_available, reason="SM120 fused MXFP8 required")
def test_fused_moe_mxfp8_marks_invalid_route_nan(fused_expert_bank):
    x, expert_ids, router_weights = _fused_inputs(torch.bfloat16)
    expert_ids[0, 0] = NUM_EXPERTS
    output = ck.fused_moe_mxfp8(
        x, expert_ids, router_weights, *fused_expert_bank
    )

    assert output[0].isnan().all()
    assert output[1:].isfinite().all()


def test_fused_moe_mxfp8_has_no_eager_fallback():
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("PyTorch does not expose E8M0")
    x = torch.empty((1, 32), dtype=torch.bfloat16)
    expert_ids = torch.empty((1, TOP_K), dtype=torch.int32)
    router_weights = torch.empty((1, TOP_K), dtype=torch.float32)
    qdata = torch.empty((1, 32, 32), dtype=torch.float8_e4m3fn)
    scales = torch.empty((1, 128, 4), dtype=torch.float8_e8m0fnu)

    with ck.use_backend("eager"), pytest.raises(ck.NoCapableBackendError):
        ck.fused_moe_mxfp8(
            x, expert_ids, router_weights, qdata, scales, qdata, scales
        )


@pytest.mark.skipif(not sm120_fused_mxfp8_available, reason="SM120 fused MXFP8 required")
def test_fused_moe_workspace_can_be_released(fused_expert_bank):
    stream = torch.cuda.current_stream()
    ck.release_cuda_stream_workspaces(stream)
    assert ck.reserve_cuda_stream_workspaces(stream)
    assert not ck.reserve_cuda_stream_workspaces(stream)
    x, expert_ids, router_weights = _fused_inputs(torch.bfloat16)
    ck.fused_moe_mxfp8(x, expert_ids, router_weights, *fused_expert_bank)
    assert ck.release_cuda_stream_workspaces(stream)
    assert not ck.release_cuda_stream_workspaces(stream)
