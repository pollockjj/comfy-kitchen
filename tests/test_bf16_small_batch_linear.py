import pytest
import torch

import comfy_kitchen


@pytest.mark.parametrize("m", [1, 3])
def test_bf16_small_batch_linear_cpu_matches_linear(m):
    torch.manual_seed(7)
    x = torch.randn((1, m, 256), dtype=torch.bfloat16)
    weight = torch.randn((384, 256), dtype=torch.bfloat16)
    torch.testing.assert_close(
        comfy_kitchen.bf16_small_batch_linear(x, weight),
        torch.nn.functional.linear(x, weight),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m", [1, 3])
def test_bf16_small_batch_linear_cuda_matches_linear(m):
    torch.manual_seed(7)
    x = torch.randn((1, m, 256), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((384, 256), device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        comfy_kitchen.bf16_small_batch_linear(x, weight),
        torch.nn.functional.linear(x, weight),
        rtol=1e-2,
        atol=1e-2,
    )
