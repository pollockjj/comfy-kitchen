import weakref

import pytest
import torch

import comfy_kitchen as ck

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not ck._cuda_backend._EXT_AVAILABLE, reason="Comfy Kitchen CUDA extension is unavailable")


def test_cuda_runtime_graph_lifecycle():
    stream = torch.cuda.Stream()
    static_input = torch.zeros(1024, device="cuda")
    with torch.cuda.stream(stream):
        ck.begin_cuda_graph_capture(stream)
        torch.cuda._sleep(20_000_000)
        captured_output = static_input + 1
        graph = ck.end_cuda_graph_capture(stream, captured_output)
    for value in (4, 7):
        static_input.fill_(value)
        graph.replay(torch.cuda.current_stream())
        torch.cuda.synchronize()
        torch.testing.assert_close(captured_output, torch.full_like(static_input, value + 1))
    output_ref = weakref.ref(captured_output)
    del captured_output
    static_input.fill_(9)
    graph.replay(torch.cuda.current_stream())
    replay_complete = torch.cuda.Event()
    replay_complete.record()
    assert output_ref() is not None and not replay_complete.query()
    graph.reset()
    assert replay_complete.query()
    assert output_ref() is None
    with pytest.raises(RuntimeError, match="has been reset"):
        graph.replay(stream)
    ck.begin_cuda_graph_capture(stream)
    ck.abort_cuda_graph_capture(stream)
    with pytest.raises(RuntimeError, match="capture is not active"):
        ck.abort_cuda_graph_capture(stream)
