"""Enflame session isolation, including kernels without Triton scopes."""

import json

import pytest


def test_enflame_session_launch_membership(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_gcu")
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from flagtree import profiler

    if triton.runtime.driver.active.get_current_target().backend != "gcu":
        pytest.skip("Enflame device required")

    @triton.jit
    def session_probe(X, Y, TAG: tl.constexpr):
        i = tl.arange(0, 32)
        x = tl.load(X + i)
        tl.store(Y + i, x + TAG)

    x = torch.ones(32, device="gcu")
    y = torch.empty_like(x)
    # Compile before collecting to avoid timing compilation or initialization.
    for tag in range(5):
        session_probe[(1, )](x, y, tag)
    torch.gcu.synchronize()
    paths = [tmp_path / "first", tmp_path / "second"]
    # hook=None exercises raw TOPSPTI activities without scope correlation.
    first = profiler.start(
        str(paths[0]),
        backend="enflame",
        context="shadow",
        data="tree",
        mode="runtime_base:runtime_host_timing_fallback=false",
    )
    second = None
    try:
        session_probe[(1, )](x, y, 0)  # first only
        profiler.deactivate(first)
        session_probe[(1, )](x, y, 1)  # paused: neither
        # Vendor adapters currently reject overlapping sessions, even paused
        # ones. Preserve that API contract rather than enabling it implicitly.
        with pytest.raises(RuntimeError, match="does not support overlapping"):
            profiler.start(
                str(paths[1]),
                backend="enflame",
                context="shadow",
                data="tree",
                mode="runtime_base:runtime_host_timing_fallback=false",
            )
        profiler.activate(first)
        session_probe[(1, )](x, y, 3)  # resumed first
        session_probe[(1, )](x, y, 4)  # first only
    finally:
        profiler.finalize(first)
        if second is not None:
            profiler.finalize(second)
    second = profiler.start(
        str(paths[1]),
        backend="enflame",
        context="shadow",
        data="tree",
        mode="runtime_base:runtime_host_timing_fallback=false",
    )
    try:
        session_probe[(1, )](x, y, 2)
    finally:
        profiler.finalize(second)
    events = []
    for path in paths:
        document = json.loads(path.with_suffix(".vendor.json").read_text())
        # Assert exact launch membership, not merely a nonempty timing report.
        associations = document["associations"]
        events.append({
            a["runtime_event"]["correlation_id"]
            for a in associations
            if "session_probe" in a["runtime_event"]["op_name"]
        })
    assert len(events[0]) == 3
    assert len(events[1]) == 1
    assert not events[0] & events[1]
