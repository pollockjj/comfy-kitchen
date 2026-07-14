import torch

from comfy_kitchen.backends.cuda import bf16_tuned_gate_up_linear as _cuda_bf16_tuned_gate_up_linear


def bf16_tuned_gate_up_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _cuda_bf16_tuned_gate_up_linear(x, weight)


@torch.library.custom_op(
    "comfy_kitchen::bf16_tuned_gate_up_linear",
    mutates_args=(),
)
def _op_bf16_tuned_gate_up_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return bf16_tuned_gate_up_linear(x, weight)


@_op_bf16_tuned_gate_up_linear.register_fake
def _op_bf16_tuned_gate_up_linear_fake(x, weight):
    return torch.empty(*x.shape[:-1], weight.shape[0], dtype=x.dtype, device=x.device)
