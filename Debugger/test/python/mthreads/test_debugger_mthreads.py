import os
from pathlib import Path

import numpy as np
import pytest


def _require_real_mthreads_environment():
    if os.getenv("FLAGTREE_RUN_MTHREADS_HARDWARE_TESTS") != "1":
        pytest.skip("set FLAGTREE_RUN_MTHREADS_HARDWARE_TESTS=1 to run")
    try:
        import torch
        import torch_musa  # noqa: F401
    except Exception as exc:
        pytest.skip(f"torch_musa is not available: {exc!r}")
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA is installed, but no Moore Threads GPU is available")


def test_mthreads_debugger_level_and_address_modes(tmp_path):
    _require_real_mthreads_environment()
    import torch
    import triton
    import triton.language as tl
    from flagtree import debugger
    from flagtree import language as ftl

    @triton.jit
    def no_address_kernel(x_ptr, out_ptr, block: tl.constexpr):
        offsets = tl.arange(0, block)
        ftl.debug_collect_start(level=1, addr_level=0)
        observed = tl.load(x_ptr + offsets)
        tl.store(out_ptr + offsets, observed)
        ftl.debug_collect_end()

    @triton.jit
    def address_kernel(x_ptr, out_ptr, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.arange(0, block)
        mask = offsets < n
        ftl.debug_collect_start(level=1, addr_level=1)
        observed = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, observed, mask=mask)
        ftl.debug_collect_end()

    @triton.jit
    def full_dump_kernel(x_ptr, out_ptr, block: tl.constexpr):
        offsets = tl.arange(0, block)
        ftl.debug_collect_start(level=2, addr_level=2)
        observed = tl.load(x_ptr + offsets)
        tl.store(out_ptr + offsets, observed)
        ftl.debug_collect_end()

    def run_case(name, level, addr_level, kernel, size, expected, kernel_args):
        debugger.clear_exported_runs()
        debugger.reset_config()
        debugger.activate(
            level=level,
            addr_level=addr_level,
            output_dir=tmp_path / name,
            record_capacity=8192,
            export_raw_records=False,
        )
        x = torch.arange(size, dtype=torch.float32, device="musa")
        output = torch.full_like(x, -1.0)
        try:
            kernel[(1, )](x, output, *kernel_args, num_warps=1)
            torch.musa.synchronize()
            torch.testing.assert_close(output.cpu(), expected)
            runs = debugger.take_exported_runs()
        finally:
            debugger.deactivate()
            debugger.reset_config()
        assert len(runs) == 1
        return x, runs[0]

    _, no_address = run_case(
        "level1_addr0",
        1,
        0,
        no_address_kernel,
        8,
        torch.arange(8, dtype=torch.float32),
        (8, ),
    )
    no_address_records = no_address["decoded"]["records"]
    no_address_kinds = {record["record_kind"] for record in no_address_records}
    assert "MEMORY_EVENT" not in no_address_kinds
    assert "SUMMARY_COUNT_BUNDLE_U64" in no_address_kinds
    assert "SUMMARY_VALUE_BUNDLE_F32" in no_address_kinds

    expected = torch.full((16, ), -1.0, dtype=torch.float32)
    expected[:5] = torch.arange(5, dtype=torch.float32)
    _, address = run_case(
        "level1_addr1",
        1,
        1,
        address_kernel,
        16,
        expected,
        (5, 16),
    )
    address_records = [
        record for record in address["decoded"]["records"]
        if record["record_kind"] == "MEMORY_EVENT"
    ]
    event_names = {
        1: "last_aligned_addr",
        2: "base_aligned_addr",
        3: "first_addr",
        4: "last_addr",
        5: "min_addr",
        6: "max_addr",
        7: "active_lane_count",
        8: "address_span_bytes",
    }
    event_values = {}
    for record in address_records:
        event_kind = record["event_kind"]
        name = (event_kind.lower() if isinstance(event_kind, str) else
                event_names[int(event_kind)])
        event_values.setdefault(name, []).append(int(record["addr"]))
    assert 5 in event_values.get("active_lane_count", [])
    assert 20 in event_values.get("address_span_bytes", [])

    full_dump_x, full_dump = run_case(
        "level2_addr2",
        2,
        2,
        full_dump_kernel,
        8,
        torch.arange(8, dtype=torch.float32),
        (8, ),
    )
    artifacts = full_dump["runtime_metadata"]["full_dump_artifacts"]
    assert all(Path(artifact["path"]).is_file() for artifact in artifacts)
    value_arrays = [
        np.load(artifact["path"]) for artifact in artifacts
        if artifact["kind"] == "value"
    ]
    address_arrays = [
        np.load(artifact["path"]) for artifact in artifacts
        if artifact["kind"] == "memory_address"
    ]
    expected_values = np.arange(8, dtype=np.float32)
    expected_addresses = (
        np.arange(8, dtype=np.uint64) * full_dump_x.element_size() +
        full_dump_x.data_ptr())
    assert any(
        np.array_equal(array.reshape(-1), expected_values)
        for array in value_arrays)
    assert any(
        np.array_equal(array.reshape(-1), expected_addresses)
        for array in address_arrays)
