import gc
import weakref

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cuda_backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not cuda_backend._EXT_AVAILABLE, reason="Comfy Kitchen CUDA extension is unavailable")


def test_cuda_runtime_graph_lifecycle():
    stream = torch.cuda.Stream()
    static_input = torch.zeros(1024, device="cuda")
    with torch.cuda.stream(stream):
        ck.begin_cuda_graph_capture(stream)
        captured_output = static_input + 1
        graph = ck.end_cuda_graph_capture(stream, captured_output)
    output_ref = weakref.ref(captured_output)
    del captured_output
    gc.collect()
    static_input.fill_(4)
    torch.cuda.synchronize()
    graph.replay(stream)
    stream.synchronize()
    assert graph.valid
    torch.testing.assert_close(output_ref(), torch.full_like(static_input, 5))
    graph.reset()
    assert output_ref() is None
    with pytest.raises(RuntimeError, match="has been reset"):
        graph.replay(stream)
    with torch.cuda.stream(stream):
        ck.begin_cuda_graph_capture(stream)
        ck.abort_cuda_graph_capture(stream)
        result = torch.ones(1, device="cuda") + 1
    stream.synchronize()
    torch.testing.assert_close(result, torch.tensor([2.0], device="cuda"))
    with pytest.raises(RuntimeError, match="capture is not active"):
        ck.abort_cuda_graph_capture(stream)
