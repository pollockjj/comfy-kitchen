# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from comfy_kitchen.registry import registry


def categorical_stats(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Categorical probabilities, entropy, and modal token indices."""
    distribution = torch.distributions.Categorical(logits=logits)
    return distribution.probs, distribution.entropy(), logits.argmax(dim=-1)


def categorical_stats_sample(
    logits: torch.Tensor,
    exponential_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return categorical entropy, argmax, sample, and an invalid-input flag."""
    distribution = torch.distributions.Categorical(logits=logits)
    sample = (distribution.probs / exponential_noise).argmax(dim=-1)
    valid = (
        torch.isfinite(logits).all()
        & torch.isfinite(exponential_noise).all()
        & (exponential_noise > 0).all()
    )
    return distribution.entropy(), logits.argmax(dim=-1), sample, (~valid).to(torch.int32)


def softcap_scale(
    raw_logits: torch.Tensor,
    cap: float,
    inverse_temperature: float,
) -> torch.Tensor:
    """Apply logit softcapping in FP32 and scale by inverse temperature."""
    logits = raw_logits.to(torch.float32)
    return torch.tanh(logits / cap) * cap * inverse_temperature


@torch.library.custom_op("comfy_kitchen::categorical_stats", mutates_args=())
def _op_categorical_stats(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs = {"logits": logits}
    impl = registry.get_implementation("categorical_stats", kwargs=kwargs)
    return impl(**kwargs)


@_op_categorical_stats.register_fake
def _op_categorical_stats_fake(logits):
    rows = logits.shape[0]
    return (
        torch.empty_like(logits),
        torch.empty((rows,), dtype=logits.dtype, device=logits.device),
        torch.empty((rows,), dtype=torch.int64, device=logits.device),
    )


@torch.library.custom_op("comfy_kitchen::categorical_stats_sample", mutates_args=())
def _op_categorical_stats_sample(
    logits: torch.Tensor,
    exponential_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs = {"logits": logits, "exponential_noise": exponential_noise}
    impl = registry.get_implementation("categorical_stats_sample", kwargs=kwargs)
    return impl(**kwargs)


@_op_categorical_stats_sample.register_fake
def _op_categorical_stats_sample_fake(logits, exponential_noise):
    rows = logits.shape[0]
    return (
        torch.empty((rows,), dtype=logits.dtype, device=logits.device),
        torch.empty((rows,), dtype=torch.int64, device=logits.device),
        torch.empty((rows,), dtype=torch.int64, device=logits.device),
        torch.empty((), dtype=torch.int32, device=logits.device),
    )


@torch.library.custom_op("comfy_kitchen::softcap_scale", mutates_args=())
def _op_softcap_scale(
    raw_logits: torch.Tensor,
    cap: float,
    inverse_temperature: float,
) -> torch.Tensor:
    kwargs = {
        "raw_logits": raw_logits,
        "cap": cap,
        "inverse_temperature": inverse_temperature,
    }
    impl = registry.get_implementation("softcap_scale", kwargs=kwargs)
    return impl(**kwargs)


@_op_softcap_scale.register_fake
def _op_softcap_scale_fake(raw_logits, cap, inverse_temperature):
    return torch.empty_like(raw_logits, dtype=torch.float32)
