import pytest
import torch

import comfy_kitchen as ck

cuda_status = ck.list_backends().get("cuda", {})


def _reference(qweight, block_scales, indices, output_type):
    full = ck.dequantize_mxfp8(qweight, block_scales, output_type)
    return full.index_select(0, indices.reshape(-1)).reshape(*indices.shape, qweight.shape[1])


def test_mxfp8_embedding_eager_preserves_rows_and_transpose_values():
    torch.manual_seed(19)
    weight = torch.randn(128, 64, dtype=torch.bfloat16).mul_(2**-5)
    with ck.use_backend("eager"):
        qweight, block_scales = ck.quantize_mxfp8(weight)
        indices = torch.tensor([[127, 3, 3], [64, 0, 31]], dtype=torch.int64)
        candidate = ck.mxfp8_embedding(qweight, block_scales, indices)
        reference = _reference(qweight, block_scales, indices, torch.bfloat16)
        transposed, transposed_scales = ck.requantize_mxfp8_transpose(qweight, block_scales)
        actual = ck.dequantize_mxfp8(transposed, transposed_scales, torch.bfloat16)
        expected = ck.dequantize_mxfp8(qweight, block_scales, torch.bfloat16).t()
    assert torch.equal(candidate, reference)
    torch.testing.assert_close(actual, expected, rtol=0.125, atol=0.001)


def test_mxfp8_embedding_eager_rejects_out_of_range_index():
    weight = torch.zeros(128, 64, dtype=torch.bfloat16)
    with ck.use_backend("eager"):
        qweight, block_scales = ck.quantize_mxfp8(weight)
        with pytest.raises(IndexError):
            ck.mxfp8_embedding(qweight, block_scales, torch.tensor([128]))


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not cuda_status.get("available", False)
    or "mxfp8_embedding" not in cuda_status.get("capabilities", ()),
    reason="native CUDA MXFP8 embedding kernel required",
)
@pytest.mark.parametrize("output_type", [torch.float32, torch.float16, torch.bfloat16])
def test_mxfp8_embedding_cuda_is_bit_exact(output_type):
    torch.manual_seed(23)
    weight = torch.randn(256, 2816, dtype=torch.bfloat16, device="cuda")
    qweight, block_scales = ck.quantize_mxfp8(weight)
    indices = torch.tensor([255, 17, 17, 128, 0], dtype=torch.int64, device="cuda")
    with ck.use_backend("cuda"):
        candidate = ck.mxfp8_embedding(qweight, block_scales, indices, output_type)
    with ck.use_backend("eager"):
        reference = _reference(qweight, block_scales, indices, output_type)
    assert torch.equal(candidate, reference)
