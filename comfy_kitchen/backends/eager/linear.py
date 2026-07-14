# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from comfy_kitchen.registry import registry


def bf16_small_batch_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight)


@torch.library.custom_op("comfy_kitchen::bf16_small_batch_linear", mutates_args=())
def _op_bf16_small_batch_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    kwargs = {"x": x, "weight": weight}
    implementation = registry.get_implementation(
        "bf16_small_batch_linear", kwargs=kwargs
    )
    return implementation(**kwargs)


@_op_bf16_small_batch_linear.register_fake
def _op_bf16_small_batch_linear_fake(x, weight):
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype, device=x.device)
