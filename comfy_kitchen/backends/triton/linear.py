# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_small_m_linear_kernel(
    x,
    weight,
    out,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    output_block = tl.program_id(0)
    rows = tl.arange(0, 16)
    cols = output_block * block_n + tl.arange(0, block_n)
    inner = tl.arange(0, block_k)
    acc = tl.zeros((16, block_n), tl.float32)
    for k_base in range(0, k, block_k):
        x_tile = tl.load(
            x + rows[:, None] * stride_xm + (k_base + inner[None, :]) * stride_xk,
            mask=rows[:, None] < m,
            other=0.0,
        )
        weight_tile = tl.load(
            weight + cols[None, :] * stride_wn + (k_base + inner[:, None]) * stride_wk,
            mask=cols[None, :] < n,
            other=0.0,
        )
        acc += tl.dot(x_tile, weight_tile, input_precision="ieee")
    tl.store(
        out + rows[:, None] * n + cols[None, :],
        acc,
        mask=(rows[:, None] < m) & (cols[None, :] < n),
    )


def _kernel_config(n: int, k: int) -> tuple[int, int, int, int]:
    if (n, k) == (20480, 2560):
        return 128, 64, 8, 5
    if (n, k) == (256, 2560):
        return 32, 128, 2, 5
    return 64, 128, 4, 5


@torch.library.triton_op(
    "comfy_kitchen::bf16_small_m_linear",
    mutates_args={},
)
def bf16_small_m_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    if not x_2d.is_contiguous() or not weight.is_contiguous():
        raise ValueError("bf16_small_m_linear requires contiguous inputs")
    out = torch.empty(
        (x_2d.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
    )
    block_n, block_k, num_warps, num_stages = _kernel_config(
        weight.shape[0], weight.shape[1]
    )

    def grid(meta):
        return (triton.cdiv(weight.shape[0], meta["block_n"]),)

    torch.library.wrap_triton(_bf16_small_m_linear_kernel)[grid](
        x_2d,
        weight,
        out,
        stride_xm=x_2d.stride(0),
        stride_xk=x_2d.stride(1),
        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),
        m=x_2d.shape[0],
        n=weight.shape[0],
        k=weight.shape[1],
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.view(*x.shape[:-1], weight.shape[0])
