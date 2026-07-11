import pytest
import torch

import comfy_kitchen as ck


cuda_status = ck.list_backends().get("cuda", {})
sm120_grouped_mxfp8_available = (
    hasattr(torch, "float8_e8m0fnu")
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability(0) == (12, 0)
    and "grouped_scaled_mm_mxfp8" in set(cuda_status.get("capabilities", ()))
)


@pytest.mark.skipif(not sm120_grouped_mxfp8_available, reason="SM120 grouped MXFP8 required")
def test_grouped_scaled_mm_mxfp8_matches_scalar_mxfp8():
    generator = torch.Generator(device="cuda:0").manual_seed(5770521)
    groups, group_size, k, n = 2, 128, 128, 128
    x = torch.randn(
        groups * group_size,
        k,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda:0",
    ).mul_(2**-3)
    weights = torch.randn(
        groups,
        n,
        k,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda:0",
    ).mul_(2**-3)

    with ck.use_backend("cuda"):
        qx, x_scales = ck.quantize_mxfp8(x)
        quantized_weights = [ck.quantize_mxfp8(weight) for weight in weights]
        qw = torch.stack([item[0] for item in quantized_weights])
        weight_scales = torch.stack([item[1] for item in quantized_weights])
        candidate = ck.grouped_scaled_mm_mxfp8(
            qx,
            qw,
            x_scales,
            weight_scales,
            group_size,
        )

    reference = torch.stack(
        [
            ck.scaled_mm_mxfp8(
                qx[index * group_size:(index + 1) * group_size],
                qw[index],
                x_scales[index * group_size:(index + 1) * group_size],
                weight_scales[index],
            )
            for index in range(groups)
        ]
    )
    delta = candidate.float() - reference.float()
    relative_rmse = delta.square().mean().sqrt() / reference.float().square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        candidate.float().flatten(1), reference.float().flatten(1), dim=-1
    ).mean()

    assert torch.isfinite(candidate).all()
    assert delta.abs().max() <= 0.0625
    assert relative_rmse <= 1.0e-3
    assert cosine >= 0.99999


@pytest.mark.skipif(not sm120_grouped_mxfp8_available, reason="SM120 grouped MXFP8 required")
def test_grouped_scaled_mm_mxfp8_rejects_unaligned_group_size():
    qdata = torch.empty((64, 32), dtype=torch.float8_e4m3fn, device="cuda:0")
    scales = torch.empty((128, 4), dtype=torch.float8_e8m0fnu, device="cuda:0")
    weights = torch.empty((1, 128, 32), dtype=torch.float8_e4m3fn, device="cuda:0")
    weight_scales = torch.empty((1, 128, 4), dtype=torch.float8_e8m0fnu, device="cuda:0")

    with pytest.raises(ValueError, match="positive multiple of 128"):
        ck.grouped_scaled_mm_mxfp8(qdata, weights, scales, weight_scales, 64)


def test_grouped_scaled_mm_mxfp8_has_no_eager_fallback():
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("PyTorch does not expose E8M0")
    qdata = torch.empty((128, 32), dtype=torch.float8_e4m3fn)
    scales = torch.empty((128, 4), dtype=torch.float8_e8m0fnu)
    weights = torch.empty((1, 128, 32), dtype=torch.float8_e4m3fn)
    weight_scales = torch.empty((1, 128, 4), dtype=torch.float8_e8m0fnu)

    with ck.use_backend("eager"), pytest.raises(ck.NoCapableBackendError):
        ck.grouped_scaled_mm_mxfp8(qdata, weights, scales, weight_scales, 128)
