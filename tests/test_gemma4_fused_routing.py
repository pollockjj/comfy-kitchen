import pytest
import torch

import comfy_kitchen as ck


def _cuda_routing_available():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        return False
    backend = next(item for item in ck.list_backends() if item["name"] == "cuda")
    return "gemma4_fused_routing" in backend["capabilities"]


pytestmark = pytest.mark.skipif(
    not _cuda_routing_available(), reason="Gemma4 fused CUDA routing is unavailable"
)


def _reference(logits, scales):
    ids = torch.argsort(logits.float(), dim=-1, descending=True, stable=True)[:, :8]
    selected = torch.gather(logits.float(), 1, ids)
    weights = torch.softmax(selected, dim=-1) * scales[ids]
    return weights, ids


@pytest.mark.parametrize("tokens", [1, 256, 340])
def test_gemma4_fused_routing_matches_deterministic_reference(tokens):
    generator = torch.Generator(device="cuda").manual_seed(5770521)
    logits = torch.randn(tokens, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
    scales = torch.randn(128, generator=generator, device="cuda", dtype=torch.bfloat16)

    weights, ids = ck.gemma4_fused_routing(logits, scales)
    reference_weights, reference_ids = _reference(logits, scales)

    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    assert torch.equal(ids.to(torch.int64), reference_ids)
    torch.testing.assert_close(weights, reference_weights, rtol=2e-6, atol=2e-7)


def test_gemma4_fused_routing_breaks_ties_by_expert_id():
    logits = torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16)
    scales = torch.arange(1, 129, device="cuda", dtype=torch.bfloat16)

    weights, ids = ck.gemma4_fused_routing(logits, scales)

    assert torch.equal(ids.cpu(), torch.arange(8, dtype=torch.int32).unsqueeze(0))
    torch.testing.assert_close(weights.cpu(), scales[:8].float().cpu().unsqueeze(0) / 8)


def test_gemma4_fused_routing_accepts_empty_batch():
    logits = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    scales = torch.ones(128, device="cuda", dtype=torch.bfloat16)

    weights, ids = ck.gemma4_fused_routing(logits, scales)

    assert weights.shape == (0, 8)
    assert ids.shape == (0, 8)
