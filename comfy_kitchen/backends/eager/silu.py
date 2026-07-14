# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from comfy_kitchen.registry import registry


def bf16_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


@torch.library.custom_op("comfy_kitchen::bf16_silu_mul", mutates_args=())
def _op_bf16_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    kwargs = {"gate": gate, "up": up}
    implementation = registry.get_implementation("bf16_silu_mul", kwargs=kwargs)
    return implementation(**kwargs)


@_op_bf16_silu_mul.register_fake
def _op_bf16_silu_mul_fake(gate, up):
    return torch.empty_like(gate)
