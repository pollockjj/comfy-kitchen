import pytest
import torch

import comfy_kitchen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n,k", [(3072, 2560), (256, 2560)])
def test_bf16_small_m_linear_is_exact(n, k):
    torch.manual_seed(7)
    x = torch.randn((1, 3, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x, weight)
    actual = comfy_kitchen.bf16_small_m_linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_small_m_linear_rejects_strided_rows():
    storage = torch.randn((1, 3, 2577), device="cuda", dtype=torch.bfloat16)
    x = storage[..., :2560]
    weight = torch.randn((3072, 2560), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires contiguous inputs"):
        comfy_kitchen.bf16_small_m_linear(x, weight)
