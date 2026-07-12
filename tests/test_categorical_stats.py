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

    assert (probs.shape, entropy.shape, argmax.shape) == (
        logits.shape, logits.shape[:-1], logits.shape[:-1]
    )
    assert argmax.dtype == torch.int64
    torch.testing.assert_close(probs, reference.probs, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(entropy, reference.entropy(), rtol=1e-5, atol=1e-5)
    assert torch.equal(argmax, logits.argmax(dim=-1))

def test_categorical_stats_rejects_unsupported_storage():
    with pytest.raises(ValueError, match="float32"):
        ck.categorical_stats(torch.randn(2, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="contiguous"):
        ck.categorical_stats(torch.randn(2, 8).t())


@pytest.mark.parametrize("backend", ["eager", "cuda"])
def test_categorical_stats_sample_matches_multinomial(backend, cuda_available, seed):
    device = "cuda" if backend == "cuda" else "cpu"
    if backend == "cuda" and not cuda_available:
        pytest.skip("CUDA unavailable")
    if backend not in get_capable_backends("categorical_stats_sample", device):
        pytest.skip(f"backend '{backend}' is not capable")

    logits = torch.randn((2, 3, 257), dtype=torch.float32, device=device)
    reference_generator = torch.Generator(device=device).manual_seed(5770521)
    noise_generator = torch.Generator(device=device).manual_seed(5770521)
    noise = torch.empty_like(logits).exponential_(generator=noise_generator)

    with ck.use_backend(backend):
        old_probs, old_entropy, old_argmax = ck.categorical_stats(logits)
        reference = torch.multinomial(
            old_probs.reshape(-1, logits.shape[-1]), 1, generator=reference_generator
        ).reshape(logits.shape[:-1])
        entropy, argmax, sample = ck.categorical_stats_sample(logits, noise)

    assert torch.equal(entropy, old_entropy)
    assert torch.equal(argmax, old_argmax)
    assert torch.equal(sample, reference)
    assert torch.equal(noise_generator.get_state(), reference_generator.get_state())


@pytest.mark.parametrize("backend", ["eager", "cuda"])
def test_softcap_scale_is_bit_exact(backend, cuda_available):
    device = "cuda" if backend == "cuda" else "cpu"
    if backend == "cuda" and not cuda_available:
        pytest.skip("CUDA unavailable")
    if backend not in get_capable_backends("softcap_scale", device):
        pytest.skip(f"backend '{backend}' is not capable")

    raw_logits = torch.linspace(-64, 64, 1542, device=device).to(torch.bfloat16).reshape(2, 3, 257)
    reference = torch.tanh(raw_logits.float() / 30.0) * 30.0 * 1.25
    with ck.use_backend(backend):
        output = ck.softcap_scale(raw_logits, 30.0, 1.25)

    assert output.dtype == torch.float32
    assert output.shape == raw_logits.shape
    assert torch.equal(output, reference)


@pytest.mark.parametrize("backend", ["eager", "cuda"])
def test_softcap_categorical_stats_sample_matches_composed_ops(backend, cuda_available):
    device = "cuda" if backend == "cuda" else "cpu"
    if backend == "cuda" and not cuda_available:
        pytest.skip("CUDA unavailable")
    if backend not in get_capable_backends("softcap_categorical_stats_sample", device):
        pytest.skip(f"backend '{backend}' is not capable")
    if backend not in get_capable_backends("softcap_categorical_stats_sample_bf16", device):
        pytest.skip(f"backend '{backend}' is not capable")

    raw = torch.linspace(-64, 64, 1542, device=device).to(torch.bfloat16).reshape(2, 3, 257)
    raw[..., 3] = raw[..., 7] = 64
    noise = torch.linspace(1, 2, raw.numel(), dtype=torch.float32, device=device).reshape(raw.shape)
    noise[..., 3] = noise[..., 7] = 0.5
    with ck.use_backend(backend):
        processed_ref = ck.softcap_scale(raw, 30.0, 1.25)
        entropy_ref, argmax_ref, sample_ref = ck.categorical_stats_sample(processed_ref, noise)
        processed, self_conditioning, entropy, argmax, sample = (
            ck.softcap_categorical_stats_sample(raw, noise, 30.0, 1.25)
        )
        compact_self_conditioning, compact_entropy, compact_argmax, compact_sample = (
            ck.softcap_categorical_stats_sample_bf16(raw, noise, 30.0, 1.25)
        )

    assert torch.equal(processed, processed_ref)
    assert torch.equal(self_conditioning, processed_ref.to(torch.bfloat16))
    torch.testing.assert_close(entropy, entropy_ref, rtol=1e-5, atol=1e-5)
    assert torch.equal(argmax, argmax_ref)
    assert torch.equal(sample, sample_ref)
    assert torch.equal(compact_self_conditioning, self_conditioning)
    assert torch.equal(compact_entropy, entropy)
    assert torch.equal(compact_argmax, argmax)
    assert torch.equal(compact_sample, sample)
    assert torch.equal(argmax, torch.full_like(argmax, 3))
    assert torch.equal(sample, torch.full_like(sample, 3))
