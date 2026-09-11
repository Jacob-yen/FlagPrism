import csv
import json
import os
import subprocess

import pytest

from flagtree.profiler import mthreads

# TODO(FlagPrism): Run the retained MCU helper tests against a compatible Moore
# Threads MCU, MUSA SDK, and driver environment before enabling the integration.


def test_mcu_command_exposes_counter_collection_options(monkeypatch):
    monkeypatch.setattr(mthreads,
                        "find_mcu_cli",
                        lambda explicit=None: "/opt/mthreads/mcu")

    command = mthreads.build_mcu_command(
        ["python3", "workload.py"],
        devices="0,1",
        sections="LaunchStats,SpeedOfLight",
        metrics="mp__cycles_elapsed.max,sm_efficiency",
        kernel_name="regex:vector_add",
        launch_count=2,
        launch_skip=1,
        output="profile.mcu-rep",
    )

    assert command == [
        "/opt/mthreads/mcu",
        "--devices",
        "0,1",
        "--sections",
        "LaunchStats,SpeedOfLight",
        "--metrics",
        "mp__cycles_elapsed.max,sm_efficiency",
        "--kernel-name",
        "regex:vector_add",
        "--launch-count",
        "2",
        "--launch-skip",
        "1",
        "--output",
        "profile.mcu-rep",
        "--force-overwrite",
        "python3",
        "workload.py",
    ]


def test_run_mcu_profile_preserves_output_and_log(monkeypatch, tmp_path):
    monkeypatch.setattr(mthreads, "MCU_INTEGRATION_ENABLED", True)
    monkeypatch.setattr(mthreads, "build_mcu_command",
                        lambda *args, **kwargs: ["mcu", "workload"])
    monkeypatch.setattr(
        mthreads.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "counter output\n", "diagnostic\n"),
    )
    log = tmp_path / "profile.mcu.log"

    result = mthreads.run_mcu_profile(["workload"], log_path=str(log))

    assert result.returncode == 0
    assert log.read_text(encoding="utf-8") == "counter output\ndiagnostic\n"


def test_mcu_long_csv_preserves_hardware_counter_names(tmp_path):
    path = tmp_path / "long.csv"
    path.write_text(
        "Kernel Name,Start Time Us,Duration Us,Metric Name,Metric Value\n"
        "vector_add,10,2,mp__cycles_elapsed.max,12345\n",
        encoding="utf-8",
    )

    associations = mthreads._parse_csv(path)

    assert len(associations) == 1
    association = associations[0]
    assert association["source"] == "mcu_csv"
    assert association["runtime_event"]["start_time_ns"] == 10_000
    assert association["runtime_event"]["end_time_ns"] == 12_000
    assert association["metrics"]["mthreads.mp__cycles_elapsed.max"] == 12345


def test_mcu_wide_csv_imports_instruction_bandwidth_and_utilization(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text(
        "Kernel Name,Instructions Executed,gld_throughput,gst_throughput,"
        "sm_efficiency\n"
        "vector_add,4096,120.5,64.25,87.5%\n",
        encoding="utf-8",
    )

    metrics = mthreads._parse_csv(path)[0]["metrics"]

    assert metrics["mthreads.instructions_executed"] == 4096
    assert metrics["mthreads.gld_throughput"] == 120.5
    assert metrics["mthreads.gst_throughput"] == 64.25
    assert metrics["mthreads.sm_efficiency"] == 87.5


def test_merge_mcu_csv_updates_vendor_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(mthreads, "MCU_INTEGRATION_ENABLED", True)
    base = tmp_path / "profile"
    vendor = base.with_suffix(".vendor.json")
    vendor.write_text(
        json.dumps({
            "backend":
            "mthreads",
            "enabled_metrics": ["cycles", "hardware_counters"],
            "raw_inputs": [],
            "associations": [],
            "degrade_reasons": [
                "MThreads metric 'cycles' was enabled but the capture "
                "produced no value for it."
            ],
        }),
        encoding="utf-8",
    )
    report = tmp_path / "profile.mcu-rep"
    report.write_bytes(b"mcu report")
    csv_path = tmp_path / "profile.csv"
    csv_path.write_text(
        "Kernel Name,Metric,Value\n"
        "vector_add,mp__cycles_elapsed.max,12345\n",
        encoding="utf-8",
    )

    merged = mthreads.merge_mcu_vendor_artifact(str(base), str(report),
                                                str(csv_path))

    assert merged == vendor
    artifact = json.loads(vendor.read_text(encoding="utf-8"))
    assert str(report) in artifact["raw_inputs"]
    assert str(csv_path) in artifact["raw_inputs"]
    assert artifact["summary"]["counts_by_source"]["mcu_csv"] == 1
    assert "mthreads.mp__cycles_elapsed.max" in artifact["enabled_metrics"]
    assert not artifact["degrade_reasons"]


def test_mcu_integration_is_frozen():
    assert not mthreads.MCU_INTEGRATION_ENABLED
    with pytest.raises(RuntimeError, match="integration is frozen"):
        mthreads.run_mcu_profile(["workload"])
    with pytest.raises(RuntimeError, match="integration is frozen"):
        mthreads.merge_mcu_vendor_artifact("profile", "profile.mcu-rep")


def test_mcu_cli_is_hidden_while_frozen(monkeypatch, capsys):
    from flagtree.profiler import cli

    monkeypatch.setattr(cli.sys, "argv", ["flagtree-profiler", "--help"])
    with pytest.raises(SystemExit) as error:
        cli.parse_arguments()

    assert error.value.code == 0
    assert "--mcu" not in capsys.readouterr().out


def test_compiled_importer_keeps_mcu_counter_collection_frozen(tmp_path):
    import flagtree.profiler as profiler

    csv_path = tmp_path / "counters.csv"
    csv_path.write_text(
        "Kernel Name,Metric Name,Metric Value\n"
        "vector_add,mp__cycles_elapsed.max,12345\n"
        "vector_add,mp__inst_executed.sum,4096\n"
        "vector_add,dram__bytes_read.sum.per_second,42.5\n"
        "vector_add,mp__throughput.avg.pct_of_peak_sustained_elapsed,87.5\n",
        encoding="utf-8",
    )
    base = tmp_path / "compiled_import"
    session = profiler.start(
        name=str(base),
        context="shadow",
        data="tree",
        backend="mthreads",
        hook="triton",
        mode=("runtime_base:vendor_metrics=instruction_count,cycles,"
              "memory_bandwidth,sm_utilization,hardware_counters:"
              f"mcu_import_path={csv_path}"),
    )
    profiler.finalize(session)

    artifact = json.loads(base.with_suffix(".vendor.json").read_text())
    assert not [
        item for item in artifact["associations"]
        if item["source"] == "mcu_csv"
    ]
    assert not set(artifact["enabled_metrics"]).intersection({
        "instruction_count",
        "cycles",
        "memory_bandwidth",
        "sm_utilization",
        "hardware_counters",
    })
    assert any("MCU integration is frozen" in reason
               for reason in artifact["degrade_reasons"])


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


def test_mthreads_real_gpu_launch_occupancy_and_resources(tmp_path):
    _require_real_mthreads_environment()
    import torch
    import torch_musa  # noqa: F401
    import triton
    import triton.language as tl
    import flagtree.profiler as profiler

    @triton.jit
    def vector_add(x, y, output, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < n
        tl.store(output + offsets,
                 tl.load(x + offsets, mask=mask) +
                 tl.load(y + offsets, mask=mask),
                 mask=mask)

    base = tmp_path / "profile"
    session = profiler.start(
        name=str(base),
        context="shadow",
        data="tree",
        backend="mthreads",
        hook="triton",
        mode=("runtime_base:vendor_metrics=launch_stats,occupancy,"
              "resource_usage,peak_memory_bandwidth"),
    )
    n = 4096
    x = torch.randn(n, device="musa")
    y = torch.randn(n, device="musa")
    output = torch.empty_like(x)
    vector_add[(triton.cdiv(n, 128), )](x, y, output, n=n, block=128)
    torch.musa.synchronize()
    profiler.finalize(session)
    torch.testing.assert_close(output.cpu(), (x + y).cpu())

    with base.with_suffix(".mupti.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row["kernel_name"] == "vector_add"]
    assert len(matches) == 1
    row = matches[0]
    assert row["capture_source"] == "mupti_callback"
    assert row["occupancy_source"] in {
        "musa_driver_occupancy_api", "musa_runtime_occupancy_api"
    }
    assert 0 < float(row["theoretical_occupancy_pct"]) <= 100
    assert int(row["registers_per_thread"]) > 0
    assert int(row["registers_per_sm"]) > 0
    assert int(row["shared_memory_per_sm"]) > 0
    assert float(row["peak_memory_bandwidth_gbps"]) > 0

    meta = json.loads(base.with_suffix(".meta.json").read_text())
    assert not meta["degrade_reasons"]


def test_mthreads_activity_timestamp_or_callback_fallback(tmp_path):
    _require_real_mthreads_environment()
    import torch
    import torch_musa  # noqa: F401
    import triton
    import triton.language as tl
    import flagtree.profiler as profiler

    @triton.jit
    def activity_add(x, y, output, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < n
        tl.store(output + offsets,
                 tl.load(x + offsets, mask=mask) +
                 tl.load(y + offsets, mask=mask),
                 mask=mask)

    base = tmp_path / "activity_profile"
    session = profiler.start(
        name=str(base),
        context="shadow",
        data="tree",
        backend="mthreads",
        hook="triton",
        mode=("runtime_base:vendor_metrics=launch_stats,estimated_cycles:"
              "mupti_activity=true"),
    )
    n = 4096
    x = torch.randn(n, device="musa")
    y = torch.randn(n, device="musa")
    output = torch.empty_like(x)
    activity_add[(triton.cdiv(n, 128), )](x, y, output, n=n, block=128)
    torch.musa.synchronize()
    profiler.finalize(session)
    torch.testing.assert_close(output.cpu(), (x + y).cpu())

    with base.with_suffix(".mupti.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row["kernel_name"] == "activity_add"]
    assert len(matches) == 1
    row = matches[0]
    meta = json.loads(base.with_suffix(".meta.json").read_text())
    reasons = meta["degrade_reasons"]

    if row["capture_source"] == "mupti_activity":
        assert int(row["end_time_ns"]) > int(row["start_time_ns"]) > 0
        assert float(row["estimated_elapsed_cycles"]) > 0
        assert not any("estimated_cycles" in reason for reason in reasons)
    else:
        assert row["capture_source"] == "mupti_callback"
        assert int(row["end_time_ns"]) > int(row["start_time_ns"]) > 0
        assert not row["estimated_elapsed_cycles"]
        assert any("estimated_cycles" in reason for reason in reasons)
