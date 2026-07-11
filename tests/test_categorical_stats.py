# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import comfy_kitchen as ck
from tests.conftest import get_capable_backends


@pytest.mark.parametrize("backend", ["eager", "cuda"])
def test_categorical_stats_matches_torch(backend, cuda_available, seed):
    device = "cuda" if backend == "cuda" else "cpu"
    if backend == "cuda" and not cuda_available:
        pytest.skip("CUDA unavailable")
    if backend not in get_capable_backends("categorical_stats", device):
        pytest.skip(f"backend '{backend}' is not capable")

    logits = torch.randn((2, 3, 4096), dtype=torch.float32, device=device)
    reference = torch.distributions.Categorical(logits=logits)
    with ck.use_backend(backend):
        probs, entropy, argmax = ck.categorical_stats(logits)

    assert probs.shape == logits.shape
    assert entropy.shape == logits.shape[:-1]
    assert argmax.shape == logits.shape[:-1]
    assert argmax.dtype == torch.int64
    torch.testing.assert_close(probs, reference.probs, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(entropy, reference.entropy(), rtol=1e-5, atol=1e-5)
    assert torch.equal(argmax, logits.argmax(dim=-1))


def test_categorical_stats_preserves_single_row_shape(seed):
    logits = torch.randn(97, dtype=torch.float32)
    probs, entropy, argmax = ck.categorical_stats(logits)
    assert probs.shape == logits.shape
    assert entropy.shape == torch.Size([])
    assert argmax.shape == torch.Size([])


def test_categorical_stats_rejects_unsupported_storage():
    with pytest.raises(ValueError, match="float32"):
        ck.categorical_stats(torch.randn(2, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="contiguous"):
        ck.categorical_stats(torch.randn(2, 8).t())
