import torch

from comfy_kitchen.registry import registry


def bf16_tuned_gate_up_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight)


@torch.library.custom_op(
    "comfy_kitchen::bf16_tuned_gate_up_linear",
    mutates_args=(),
)
def _op_bf16_tuned_gate_up_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    kwargs = {"x": x, "weight": weight}
    impl = registry.get_implementation("bf16_tuned_gate_up_linear", kwargs=kwargs)
    return impl(**kwargs)


@_op_bf16_tuned_gate_up_linear.register_fake
def _op_bf16_tuned_gate_up_linear_fake(x, weight):
    return torch.empty(*x.shape[:-1], weight.shape[0], dtype=x.dtype, device=x.device)
