import pytest
import torch

import comfy_kitchen


def test_bf16_silu_mul_cpu_is_exact():
    torch.manual_seed(7)
    gate = torch.randn((1, 1, 9728), dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    torch.testing.assert_close(
        comfy_kitchen.bf16_silu_mul(gate, up),
        torch.nn.functional.silu(gate) * up,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_silu_mul_cuda_is_exact():
    torch.manual_seed(7)
    gate = torch.randn((1, 1, 9728), device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    torch.testing.assert_close(
        comfy_kitchen.bf16_silu_mul(gate, up),
        torch.nn.functional.silu(gate) * up,
        rtol=0,
        atol=0,
    )
