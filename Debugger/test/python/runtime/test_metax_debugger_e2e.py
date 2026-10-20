# FlagPrism：verify MACA hidden arguments, stream ordering and tensor payload contents.
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl
import flagtree.language as ftl
from flagtree import debugger


@pytest.fixture(autouse=True)
def metax_device():
    if not torch.cuda.is_available():
        pytest.skip("MACA device required")
    if triton.runtime.driver.active.get_current_target().backend != "maca":
        pytest.skip("MetaX target required")
    assert debugger.is_available(), "MetaX debugger bindings unavailable"
    debugger.clear_exported_runs()
    debugger.reset_config()
    yield
    debugger.deactivate()
    debugger.reset_config()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_metax_level2_value_and_address_contents(tmp_path, dtype):
    @triton.jit
    def capture_copy(X, Y):
        offset = tl.arange(0, 16)
        ftl.debug_collect_start(level=2, addr_level=2)
        value = tl.load(X + offset)
        tl.store(Y + offset, value)
        ftl.debug_collect_end()

    x = torch.arange(-8, 8, device="cuda", dtype=dtype) + 0.25
    y = torch.empty_like(x)
    debugger.activate(level=2, addr_level=2, timeline=False, output_dir=tmp_path)
    capture_copy[(1,)](x, y, num_warps=1)
    torch.cuda.synchronize()
    runs = debugger.take_exported_runs()
    assert torch.equal(x, y)
    assert len(runs) == 1
    assert runs[0]["meta"]["backend_kind"] == 7
    assert runs[0]["decoded"]["header"]["overflow_count"] == 0
    artifacts = runs[0]["runtime_metadata"]["full_dump_artifacts"]
    values = [a for a in artifacts if a["kind"] == "value"]
    addresses = [a for a in artifacts if a["kind"] == "memory_address"]
    assert values and addresses
    expected = x.float().cpu().numpy()
    for artifact in values:
        assert Path(artifact["path"]).is_file()
        tensor = np.load(artifact["path"], allow_pickle=False)
        # Pointer arithmetic may also be observed; compare all float value dumps.
        if tensor.dtype == np.float32:
            np.testing.assert_array_equal(tensor.reshape(-1), expected)
    assert any(np.load(a["path"], allow_pickle=False).dtype == np.float32 for a in values)
    bases = (x.data_ptr(), y.data_ptr())
    for artifact in addresses:
        actual = np.load(artifact["path"], allow_pickle=False).reshape(-1)
        contiguous = any(np.array_equal(actual, base + np.arange(16, dtype=np.uint64) * x.element_size())
                         for base in bases)
        # A tt.splat pointer result is the broadcast base, before tt.addptr.
        broadcast = any(np.array_equal(actual, np.full(16, base, dtype=np.uint64))
                        for base in bases)
        assert contiguous or (artifact["source"] == "result" and broadcast), (
            "Captured pointer result or memory-access addresses are incorrect")
    accesses = [a for a in addresses if a["source"] == "address"]
    assert len(accesses) == 2, "Both the load and store must expose actual lane addresses"


def test_metax_masked_non_default_stream_launches():
    @triton.jit
    def masked_copy(X, Y, N: tl.constexpr):
        offset = tl.program_id(0) * 32 + tl.arange(0, 32)
        ftl.debug_collect_start(level=1, addr_level=1)
        value = tl.load(X + offset, offset < N, other=0.0)
        tl.store(Y + offset, value, offset < N)
        ftl.debug_collect_end()

    n = 37
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    outputs = [torch.empty_like(x), torch.empty_like(x)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    ready = torch.cuda.current_stream()
    for stream in streams:
        stream.wait_stream(ready)
    debugger.activate(level=1, addr_level=1, timeline=False, output_dir=None)
    for stream, output in zip(streams, outputs):
        with torch.cuda.stream(stream):
            masked_copy[(2,)](x, output, n, num_warps=1)
    for stream in streams:
        stream.synchronize()
    runs = debugger.take_exported_runs()
    assert all(torch.equal(x, output) for output in outputs)
    assert len(runs) == 2
    assert all(run["meta"]["backend_kind"] == 7 for run in runs)
    assert all(run["decoded"]["records"] for run in runs)
    assert all(run["decoded"]["header"]["overflow_count"] == 0 for run in runs)
# FlagPrism end
