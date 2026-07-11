# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import comfy_kitchen as ck
from tests.conftest import get_capable_backends


@pytest.mark.parametrize("rows", [256, 340])
@pytest.mark.parametrize("backend", ["eager", "cuda"])
def test_rmsnorm_quantize_mxfp8_matches_composed_reference(
    rows, backend, cuda_available, seed
):
    device = "cuda" if backend == "cuda" else "cpu"
    if backend == "cuda" and not cuda_available:
        pytest.skip("CUDA unavailable")
    if backend not in get_capable_backends("rmsnorm_quantize_mxfp8", device):
        pytest.skip(f"backend '{backend}' is not capable")

    x = torch.randn((rows, 2816), dtype=torch.bfloat16, device=device)
    weight = torch.randn((2816,), dtype=torch.bfloat16, device=device)
    with ck.use_backend(backend):
        normalized = torch.nn.functional.rms_norm(x, (2816,), weight, 1e-6)
        qdata_ref, scales_ref = ck.quantize_mxfp8(normalized, pad_32x=rows % 32 != 0)
        qdata, scales = ck.rmsnorm_quantize_mxfp8(x, weight)

    assert qdata.shape == (((rows + 31) // 32) * 32, 2816)
    assert scales.shape == (((rows + 127) // 128) * 128, 88)
    assert qdata.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float8_e8m0fnu
    assert torch.equal(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), scales_ref.view(torch.uint8))


def test_rmsnorm_quantize_mxfp8_rejects_non_dg_shape():
    with pytest.raises(ValueError, match=r"\[256\|340, 2816\]"):
        ck.rmsnorm_quantize_mxfp8(
            torch.empty((32, 2816), dtype=torch.bfloat16),
            torch.empty((2816,), dtype=torch.bfloat16),
        )
