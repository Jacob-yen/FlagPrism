"""Moore Threads profiler helpers.

Moore Perf Compute (MCU) owns hardware-counter replay and therefore wraps the
target process. FlagPrism keeps its in-process runtime trace and merges an
optional structured MCU export into the normal vendor artifact afterwards.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_MCU_SECTIONS = ("LaunchStats,MemoryWorkloadAnalysis,SpeedOfLight")
MCU_INTEGRATION_ENABLED = False
# TODO(FlagPrism): Enable and validate MCU integration when a compatible Moore
# Threads MCU, MUSA SDK, and driver test environment is available.


def _require_mcu_integration() -> None:
    if not MCU_INTEGRATION_ENABLED:
        raise RuntimeError(
            "Moore Perf Compute integration is frozen until it can be "
            "validated on a compatible Moore Threads environment")


def find_mcu_cli(explicit: str | None = None) -> str:
    candidates = [
        explicit,
        os.getenv("FLAGTREE_PROFILER_MTHREADS_MCU_CLI"),
        os.getenv("MCU_CLI"),
        "mcu",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).is_file():
            return str(Path(candidate))
    raise FileNotFoundError(
        "mcu was not found; install Moore Perf Compute or set "
        "FLAGTREE_PROFILER_MTHREADS_MCU_CLI")


def build_mcu_command(
    target: Sequence[str],
    *,
    devices: str | None = "0",
    sections: str | None = DEFAULT_MCU_SECTIONS,
    metrics: str | None = None,
    kernel_name: str | None = None,
    launch_count: int | None = None,
    launch_skip: int | None = None,
    output: str | None = None,
    force_overwrite: bool = True,
    mcu_cli: str | None = None,
) -> list[str]:
    command = [find_mcu_cli(mcu_cli)]
    if devices:
        command.extend(["--devices", str(devices)])
    if sections:
        command.extend(["--sections", str(sections)])
    if metrics:
        command.extend(["--metrics", str(metrics)])
    if kernel_name:
        command.extend(["--kernel-name", str(kernel_name)])
    if launch_count is not None:
        command.extend(["--launch-count", str(launch_count)])
    if launch_skip is not None:
        command.extend(["--launch-skip", str(launch_skip)])
    if output:
        command.extend(["--output", str(output)])
    if force_overwrite:
        command.append("--force-overwrite")
    command.extend(str(item) for item in target)
    return command


def run_mcu_profile(
    target: Sequence[str],
    *,
    devices: str | None = "0",
    sections: str | None = DEFAULT_MCU_SECTIONS,
    metrics: str | None = None,
    kernel_name: str | None = None,
    launch_count: int | None = None,
    launch_skip: int | None = None,
    output: str | None = None,
    force_overwrite: bool = True,
    mcu_cli: str | None = None,
    env: dict[str, str] | None = None,
    log_path: str | None = None,
) -> subprocess.CompletedProcess[str]:
    _require_mcu_integration()
    command = build_mcu_command(
        target,
        devices=devices,
        sections=sections,
        metrics=metrics,
        kernel_name=kernel_name,
        launch_count=launch_count,
        launch_skip=launch_skip,
        output=output,
        force_overwrite=force_overwrite,
        mcu_cli=mcu_cli,
    )
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if log_path:
        Path(log_path).write_text(
            (result.stdout or "") + (result.stderr or ""),
            encoding="utf-8",
        )
    return result


def _normalize(value: str) -> str:
    return "".join(ch.lower() for ch in value.strip() if ch.isalnum())


def _metric_key(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._]+", "_", value)
    return value.strip("_")


def _first(row: dict[str, str], names: Iterable[str]) -> tuple[str, str]:
    normalized = {
        _normalize(key): (key, value or "")
        for key, value in row.items()
    }
    for name in names:
        header, value = normalized.get(_normalize(name), ("", ""))
        if value.strip():
            return header, value.strip()
    return "", ""


def _number(value: str) -> int | float | None:
    cleaned = value.strip().replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _time_ns(value: str, header: str) -> int:
    number = _number(value)
    if number is None:
        return 0
    name = _normalize(header)
    if "ns" in name:
        return int(number)
    if "ms" in name:
        return int(number * 1_000_000)
    return int(number * 1_000)


def _find_csv_files(root: Path) -> list[Path]:
    root = root.expanduser()
    if root.is_file():
        return [root] if root.suffix.lower() == ".csv" else []
    if root.is_dir():
        return sorted(path for path in root.rglob("*.csv") if path.is_file())
    return []


def _parse_csv(path: Path) -> list[dict]:
    associations = []
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        for row in csv.DictReader(stream):
            _, kernel_name = _first(
                row, ("kernel_name", "kernel name", "op_name", "name"))
            start_header, start_value = _first(
                row, ("start_time_ns", "start_time_us", "start_time", "start"))
            end_header, end_value = _first(
                row, ("end_time_ns", "end_time_us", "end_time", "end"))
            duration_header, duration_value = _first(
                row, ("duration_ns", "duration_us", "duration_ms", "duration"))
            start = _time_ns(start_value, start_header)
            end = _time_ns(end_value, end_header)
            if not end and duration_value:
                end = start + _time_ns(duration_value, duration_header)

            metric_values: dict[str, int | float | str] = {
                "mthreads.mcu_source_file": str(path)
            }
            _, metric_name = _first(
                row, ("metric_name", "metric", "counter_name", "counter"))
            _, metric_value = _first(
                row, ("metric_value", "value", "counter_value"))
            if metric_name and metric_value:
                parsed = _number(metric_value)
                metric_values[f"mthreads.{_metric_key(metric_name)}"] = (
                    parsed if parsed is not None else metric_value)

            structural = {
                "kernelname",
                "opname",
                "name",
                "starttimens",
                "starttimeus",
                "starttime",
                "start",
                "endtimens",
                "endtimeus",
                "endtime",
                "end",
                "durationns",
                "durationus",
                "durationms",
                "duration",
                "metricname",
                "metric",
                "countername",
                "counter",
                "metricvalue",
                "value",
                "countervalue",
                "deviceid",
                "device",
                "streamid",
                "stream",
                "correlationid",
                "correlation",
                "corrid",
            }
            for header, value in row.items():
                if _normalize(header) in structural or not (value
                                                            or "").strip():
                    continue
                parsed = _number(value or "")
                metric_values[f"mthreads.{_metric_key(header)}"] = (
                    parsed if parsed is not None else (value or "").strip())

            def identifier(*names: str) -> int:
                _, value = _first(row, names)
                return int(_number(value) or 0)

            associations.append({
                "runtime_event": {
                    "scope_id":
                    0,
                    "op_name":
                    kernel_name,
                    "task_id":
                    identifier("kernel_id", "task_id", "task"),
                    "correlation_id":
                    identifier("correlation_id", "correlation", "corr_id"),
                    "device_id":
                    identifier("device_id", "device"),
                    "stream_id":
                    identifier("stream_id", "stream"),
                    "start_time_ns":
                    start,
                    "end_time_ns":
                    end,
                },
                "state": "collected",
                "source": "mcu_csv",
                "note": "imported from a structured Moore Perf Compute export",
                "metrics": metric_values,
            })
    return associations


def _summarize(artifact: dict) -> None:
    associations = artifact.get("associations", [])
    by_source: dict[str, int] = {}
    by_state: dict[str, int] = {}
    timed = 0
    for association in associations:
        source = str(association.get("source", "unknown"))
        state = str(association.get("state", "unknown"))
        by_source[source] = by_source.get(source, 0) + 1
        by_state[state] = by_state.get(state, 0) + 1
        event = association.get("runtime_event") or {}
        if int(event.get("end_time_ns",
                         0)) > int(event.get("start_time_ns", 0)):
            timed += 1
    artifact.setdefault("summary", {}).update({
        "raw_input_count":
        len(artifact.get("raw_inputs", [])),
        "association_count":
        len(associations),
        "timed_association_count":
        timed,
        "counts_by_source":
        by_source,
        "counts_by_state":
        by_state,
    })


def _has_capability(artifact: dict, capability: str) -> bool:
    names = {
        name
        for association in artifact.get("associations", [])
        for name in (association.get("metrics") or {})
    }
    if capability == "instruction_count":
        return any("instruction" in name or "__inst_" in name
                   for name in names)
    if capability == "cycles":
        return any("cycle" in name and "estimated" not in name
                   for name in names)
    if capability == "memory_bandwidth":
        return any(
            "throughput" in name or ("bytes" in name and "per_second" in name)
            or ("bandwidth" in name and "peak_memory_bandwidth" not in name)
            for name in names)
    if capability == "sm_utilization":
        return any(token in name for name in names
                   for token in ("sm_efficiency", "sm__throughput",
                                 "mp__throughput", "mp_utilization"))
    if capability == "hardware_counters":
        return any(
            association.get("source") == "mcu_csv"
            and len(association.get("metrics") or {}) > 1
            for association in artifact.get("associations", []))
    return False


def merge_mcu_vendor_artifact(
    name: str,
    report_path: str | None,
    csv_import_path: str | None = None,
    log_path: str | None = None,
) -> Path | None:
    """Attach MCU report files and structured CSV rows to a vendor artifact."""
    _require_mcu_integration()
    vendor_path = Path(name).with_suffix(".vendor.json")
    if not vendor_path.exists():
        return None
    artifact = json.loads(vendor_path.read_text(encoding="utf-8"))
    raw_inputs = artifact.setdefault("raw_inputs", [])
    for raw in (report_path, log_path):
        if raw and Path(raw).is_file() and str(Path(raw)) not in raw_inputs:
            raw_inputs.append(str(Path(raw)))

    files = _find_csv_files(Path(csv_import_path)) if csv_import_path else []
    associations = artifact.setdefault("associations", [])
    existing = {
        str(item.get("metrics", {}).get("mthreads.mcu_source_file", ""))
        for item in associations if isinstance(item, dict)
    }
    for path in files:
        if str(path) not in raw_inputs:
            raw_inputs.append(str(path))
        if str(path) not in existing:
            associations.extend(_parse_csv(path))

    if files:
        enabled = artifact.setdefault("enabled_metrics", [])
        for association in associations:
            for metric in (association.get("metrics") or {}):
                if metric.startswith("mthreads.") and metric not in enabled:
                    enabled.append(metric)
        retained_reasons = []
        for reason in artifact.setdefault("degrade_reasons", []):
            match = re.search(r"MThreads metric '([^']+)'", str(reason))
            if (match and _has_capability(artifact, match.group(1))):
                continue
            retained_reasons.append(reason)
        artifact["degrade_reasons"] = retained_reasons
    elif report_path and Path(report_path).is_file():
        reason = (
            "MCU binary report was retained, but no structured CSV export was "
            "provided; hardware counters cannot be merged into vendor.json.")
        reasons = artifact.setdefault("degrade_reasons", [])
        if reason not in reasons:
            reasons.append(reason)

    _summarize(artifact)
    vendor_path.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return vendor_path
