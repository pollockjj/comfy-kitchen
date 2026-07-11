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
