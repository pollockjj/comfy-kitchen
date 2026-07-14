import pytest
import torch

import comfy_kitchen as ck

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not ck._cuda_backend._EXT_AVAILABLE,
    reason="Comfy Kitchen CUDA extension is unavailable",
)


def test_cuda_runtime_graph_update_and_lifecycle():
    stream = torch.cuda.Stream()
    static_input = torch.zeros(1024, device="cuda")

    with torch.cuda.stream(stream):
        ck.begin_cuda_graph_capture(stream)
        output = static_input + 1
        graph = ck.end_cuda_graph_capture(stream, output, static_input)
    static_input.fill_(4)
    graph.replay(stream)
    stream.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 5))

    with torch.cuda.stream(stream):
        ck.begin_cuda_graph_capture(stream)
        updated_output = static_input * 2
        graph.update_from_capture(stream, updated_output, static_input)
    static_input.fill_(7)
    graph.replay(stream)
    stream.synchronize()
    torch.testing.assert_close(updated_output, torch.full_like(updated_output, 14))

    graph.reset()
    assert not graph.valid
    with pytest.raises(RuntimeError, match="has been reset"):
        graph.replay(stream)
