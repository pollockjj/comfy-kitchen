import pytest
import torch

import comfy_kitchen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_tuned_gate_up_is_exact():
    if torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("SM120 required")
    torch.manual_seed(7)
    x = torch.randn((1, 3, 2560), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((20480, 2560), device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, weight)
    actual = comfy_kitchen.bf16_tuned_gate_up_linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
