import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck

NUM_EXPERTS = 128
HIDDEN_SIZE = 2816
INTERMEDIATE_SIZE = 704
TOP_K = 8

cuda_status = ck.list_backends().get("cuda", {})
sm120_moe_available = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability(0) == (12, 0)
    and {"grouped_scaled_mm_nvfp4", "fused_moe_nvfp4"}
    <= set(cuda_status.get("capabilities", ()))
)
pytestmark = pytest.mark.skipif(not sm120_moe_available, reason="SM120 NVFP4 MoE required")


def _quantize_expert_templates(templates, scale):
    quantized = [ck.quantize_nvfp4(template, scale, hi_first=False) for template in templates]
    qdata = torch.stack([item[0] for item in quantized]).repeat(64, 1, 1)
    block_scales = torch.stack([item[1] for item in quantized]).repeat(64, 1, 1)
    return qdata, block_scales


@pytest.fixture(scope="module")
def expert_bank():
    generator = torch.Generator(device="cuda:0").manual_seed(5770521)
    weight_scale = torch.ones(1, dtype=torch.float32, device="cuda:0")
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
        fc1_qdata, fc1_block_scales = _quantize_expert_templates(fc1, weight_scale)
        fc2_qdata, fc2_block_scales = _quantize_expert_templates(fc2, weight_scale)
    return weight_scale, fc1_qdata, fc1_block_scales, fc2_qdata, fc2_block_scales


def _inputs(num_tokens):
    generator = torch.Generator(device="cuda:0").manual_seed(5770521 + num_tokens)
    x = torch.randn(
        num_tokens,
        HIDDEN_SIZE,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    logits = torch.randn(
        num_tokens,
        NUM_EXPERTS,
        generator=generator,
        dtype=torch.float32,
        device="cuda:0",
    )
    probabilities = torch.softmax(logits, dim=-1)
    router_weights, expert_ids = torch.topk(probabilities, TOP_K, dim=-1)
    router_weights.div_(router_weights.sum(dim=-1, keepdim=True))
    return x, expert_ids.contiguous(), router_weights.contiguous()


def _grouped_reference(x, expert_ids, router_weights, expert_bank):
    weight_scale, fc1_qdata, fc1_block_scales, fc2_qdata, fc2_block_scales = expert_bank
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

    input_scale = torch.full((1,), 2**-9, dtype=torch.float32, device=x.device)
    intermediate_scale = torch.full((1,), 2**-2, dtype=torch.float32, device=x.device)
    alpha1 = input_scale.mul(weight_scale).expand(NUM_EXPERTS).contiguous()
    alpha2 = intermediate_scale.mul(weight_scale).expand(NUM_EXPERTS).contiguous()
    with ck.use_backend("cuda"):
        qx, x_block_scales = ck.quantize_nvfp4(
            x[gather_tokens], input_scale, hi_first=False
        )
        gate_up = ck.grouped_scaled_mm_nvfp4(
            qx,
            fc1_qdata,
            input_scale,
            weight_scale,
            x_block_scales,
            fc1_block_scales,
            group_size,
            alpha=alpha1,
        )
        up, gate = gate_up.chunk(2, dim=-1)
        intermediate = functional.gelu(gate, approximate="tanh") * up
        qi, intermediate_block_scales = ck.quantize_nvfp4(
            intermediate.flatten(0, 1), intermediate_scale, hi_first=False
        )
        routed_output = ck.grouped_scaled_mm_nvfp4(
            qi,
            fc2_qdata,
            intermediate_scale,
            weight_scale,
            intermediate_block_scales,
            fc2_block_scales,
            group_size,
            alpha=alpha2,
        )

    pair_order = torch.empty(num_tokens * TOP_K, dtype=torch.long, device=x.device)
    pair_order[order] = slot
    routed_output = routed_output.reshape(NUM_EXPERTS * group_size, HIDDEN_SIZE)[pair_order]
    return (
        routed_output.float()
        .mul_(router_weights.reshape(-1, 1))
        .view(num_tokens, TOP_K, HIDDEN_SIZE)
        .sum(dim=1)
        .to(torch.bfloat16)
    )


def _fused(x, expert_ids, router_weights, expert_bank):
    weight_scale, fc1_qdata, fc1_block_scales, fc2_qdata, fc2_block_scales = expert_bank
    input_scale = torch.full((1,), 2**-9, dtype=torch.float32, device=x.device)
    intermediate_scale = torch.full((1,), 2**-2, dtype=torch.float32, device=x.device)
    return ck.fused_moe_nvfp4(
        x,
        expert_ids,
        router_weights,
        fc1_qdata,
        fc1_block_scales,
        fc2_qdata,
        fc2_block_scales,
        input_scale,
        intermediate_scale,
        input_scale.mul(weight_scale).expand(NUM_EXPERTS).contiguous(),
        intermediate_scale.mul(weight_scale).expand(NUM_EXPERTS).contiguous(),
    )


def test_grouped_scaled_mm_nvfp4_matches_scalar_nvfp4():
    generator = torch.Generator(device="cuda:0").manual_seed(5770521)
    groups, group_size, k, n = 2, 128, 128, 128
    scale = torch.ones(1, dtype=torch.float32, device="cuda:0")
    x = torch.randn(
        groups * group_size, k, generator=generator, dtype=torch.bfloat16, device="cuda:0"
    )
    weights = torch.randn(
        groups, n, k, generator=generator, dtype=torch.bfloat16, device="cuda:0"
    )
    with ck.use_backend("cuda"):
        qx, x_block_scales = ck.quantize_nvfp4(x, scale, hi_first=False)
        quantized_weights = [
            ck.quantize_nvfp4(weight, scale, hi_first=False) for weight in weights
        ]
        qw = torch.stack([item[0] for item in quantized_weights])
        weight_block_scales = torch.stack([item[1] for item in quantized_weights])
        alpha = torch.ones(groups, dtype=torch.float32, device="cuda:0")
        candidate = ck.grouped_scaled_mm_nvfp4(
            qx,
            qw,
            scale,
            scale,
            x_block_scales,
            weight_block_scales,
            group_size,
            alpha=alpha,
        )
        reference = torch.stack([
            ck.scaled_mm_nvfp4(
                qx[index * group_size:(index + 1) * group_size],
                qw[index],
                scale,
                scale,
                x_block_scales[index * group_size:(index + 1) * group_size],
                weight_block_scales[index],
                alpha=alpha[index:index + 1],
            )
            for index in range(groups)
        ])
    assert torch.equal(candidate, reference)


@pytest.mark.parametrize("num_tokens", [256, 340])
def test_fused_moe_nvfp4_matches_grouped_reference(num_tokens, expert_bank):
    x, expert_ids, router_weights = _inputs(num_tokens)
    candidate = _fused(x, expert_ids, router_weights, expert_bank)
    reference = _grouped_reference(x, expert_ids, router_weights, expert_bank)
    delta = candidate.float() - reference.float()
    relative_rmse = delta.square().mean().sqrt() / reference.float().square().mean().sqrt()
    cosine = functional.cosine_similarity(candidate.float(), reference.float(), dim=-1).mean()

    assert torch.isfinite(candidate).all()
    assert delta.abs().max() <= 0.0625
    assert relative_rmse <= 1.0e-3
    assert cosine >= 0.99999


def test_fused_moe_nvfp4_marks_invalid_route_nan(expert_bank):
    x, expert_ids, router_weights = _inputs(256)
    expert_ids[0, 0] = NUM_EXPERTS
    output = _fused(x, expert_ids, router_weights, expert_bank)

    assert output[0].isnan().all()
    assert output[1:].isfinite().all()
