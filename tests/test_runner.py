"""Exercise acceptance decisions without a device or an installed FlagTree."""

import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("fault_at", ["capture", "baseline"])
@pytest.mark.parametrize("source", ["error", "traceback", "log"])
@pytest.mark.parametrize("message", [
    "topsErrorInvalidDevice: device is out of service", "NPU function error",
    "VECTOR CORE EXCEPTION",
    "CUDA error: an illegal memory access was encountered",
    "HIP error: device-side assert triggered"
])
def test_device_fault_never_becomes_warning(tmp_path, monkeypatch, fault_at,
                                            source, message):
    code, summary, calls = _run(tmp_path,
                                monkeypatch,
                                fault_at,
                                source,
                                message=message)
    assert code == 1
    assert summary["status"] != "PASS"
    assert summary["counts"] == {"ERROR": 1, "BLOCKED": 1}
    row = summary["results"][0]
    assert not row["accepted"]
    fault = row if fault_at == "capture" else row["baseline_result"]
    assert fault["device_fault"]
    assert calls == (["debugger"]
                     if fault_at == "capture" else ["debugger", "execute"])


@pytest.mark.parametrize("baseline_passes", [False, True])
def test_ordinary_failure_keeps_baseline_policy(tmp_path, monkeypatch,
                                                baseline_passes):
    code, summary, calls = _run(tmp_path,
                                monkeypatch,
                                baseline_passes=baseline_passes)
    expected = "ERROR" if baseline_passes else "WARNING"
    assert code == int(baseline_passes)
    assert summary["counts"] == {expected: 2}
    assert calls == ["debugger", "execute", "profiler", "execute"]


def _run(tmp_path,
         monkeypatch,
         fault_at=None,
         source=None,
         baseline_passes=False,
         message="topsErrorInvalidDevice",
         unavailable=False,
         concurrent_fault=False,
         extra_args=(),
         setup_at=None):
    path = Path(__file__).resolve().parents[1] / "test.py"
    spec = importlib.util.spec_from_file_location("operator_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []
    original_event = threading.Event
    shared_event = original_event()
    events = []

    def make_event():
        event = original_event() if events else shared_event
        events.append(event)
        return event

    if concurrent_fault:
        monkeypatch.setattr(threading, "Event", make_event)

    def command(argv, log, timeout):
        stage = argv[argv.index("--stage") + 1]
        calls.append(stage)
        if concurrent_fault and stage == "execute":
            shared_event.set()
        result_path = Path(argv[argv.index("--result") + 1])
        case = json.loads(Path(argv[argv.index("--worker") + 1]).read_text())
        passed = stage == "execute" and baseline_passes
        data = dict(op=case["op"],
                    case_id=case["case_id"],
                    stage=stage,
                    status="PASS" if passed else "FAIL")
        if setup_at == stage:
            data.update(status="SETUP_ERROR",
                        error="Runtime initialization failed")
        log.write_text("")
        if not passed:
            data["error"] = "Unsupported operator"
        if ((fault_at == "capture" and stage != "execute")
                or (fault_at == "baseline" and stage == "execute")):
            if source == "log":
                log.write_text(message)
            else:
                data[source] = message
        runner.write_json(result_path, data)
        return 0 if passed else 1

    monkeypatch.setattr(runner, "command", command)
    monkeypatch.setattr(runner, "git_revision", lambda path: "test")

    def probe(*args, **kwargs):
        if unavailable == "timeout":
            raise runner.subprocess.TimeoutExpired("probe", 1)
        return SimpleNamespace(
            returncode=1 if unavailable == "error" else 0,
            stdout="False\n" if unavailable else "True\n",
            stderr="Import failed" if unavailable == "error" else "")

    monkeypatch.setattr(runner.subprocess, "run", probe)
    monkeypatch.setattr(sys, "argv", [
        str(path), "--out",
        str(tmp_path), "--ops", "abs", "--min-ops", "1", "--stages",
        "debugger", "profiler", "--level", "1", *extra_args
    ])
    code = runner.main()
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    return code, summary, calls


@pytest.mark.parametrize("unavailable", [True, "error", "timeout"])
def test_missing_instrumentation_is_error(tmp_path, monkeypatch, unavailable):
    code, summary, calls = _run(tmp_path, monkeypatch, unavailable=unavailable)
    assert code == 1
    assert summary["counts"] == {"ERROR": 2}
    assert calls == []


def test_fault_during_baseline_is_error(tmp_path, monkeypatch):
    code, summary, calls = _run(tmp_path, monkeypatch, concurrent_fault=True)
    assert code == 1
    assert summary["counts"] == {"ERROR": 1, "BLOCKED": 1}
    assert calls == ["debugger", "execute"]


@pytest.mark.parametrize("stage", ["debugger", "profiler"])
@pytest.mark.parametrize("broken_import", [False, True])
def test_worker_checks_tool_before_operator_setup(tmp_path, monkeypatch, stage,
                                                  broken_import):
    path = Path(__file__).resolve().parents[1] / "test.py"
    spec = importlib.util.spec_from_file_location("operator_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps(dict(op="abs", case_id="small", body_sha256="test")))

    def load(name):
        if broken_import:
            raise ImportError("Broken tool installation")
        if name.endswith(".native"):
            raise RuntimeError("Native binding unavailable")
        return SimpleNamespace(is_available=lambda: False)

    monkeypatch.setattr(runner.importlib, "import_module", load)
    result = runner.worker(
        SimpleNamespace(worker=case,
                        stage=stage,
                        level=1,
                        result=tmp_path / "result.json"))
    assert result["status"] == "UNAVAILABLE"
    assert result["error"]


@pytest.mark.parametrize("args,message", [
    (("--jobs", "2"), "requires explicit --devices"),
    (("--jobs", "0"), "--jobs must be positive"),
    (("--devices", "0,0"), "unique non-negative"),
    (("--devices", "-1"), "unique non-negative"),
    (("--devices", "0,x"), "integer indices"),
])
def test_invalid_device_scheduling_fails_before_work(tmp_path, monkeypatch,
                                                     capsys, args, message):
    with pytest.raises(SystemExit) as error:
        _run(tmp_path, monkeypatch, extra_args=args)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_parallel_workers_with_explicit_device_are_allowed(
        tmp_path, monkeypatch):
    code, summary, calls = _run(tmp_path,
                                monkeypatch,
                                extra_args=("--jobs", "2", "--devices", "0"))
    assert code == 0
    assert summary["counts"] == {"WARNING": 2}
    assert all(row["device_index"] == 0 for row in summary["results"])


@pytest.mark.parametrize("stage", ["debugger", "execute"])
def test_setup_failure_is_never_accepted(tmp_path, monkeypatch, stage):
    code, summary, calls = _run(tmp_path, monkeypatch, setup_at=stage)
    assert code == 1
    assert summary["results"][0]["status"] == "ERROR"
    assert not summary["results"][0]["accepted"]
    if stage == "debugger":
        assert "baseline_result" not in summary["results"][0]


@pytest.mark.parametrize("failure",
                         ["torch", "triton", "extension", "driver", "device"])
def test_worker_runtime_setup_errors(tmp_path, monkeypatch, failure):
    import builtins
    path = Path(__file__).resolve().parents[1] / "test.py"
    spec = importlib.util.spec_from_file_location("operator_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def fail(*args, **kwargs):
        raise RuntimeError("broken " + failure)

    interface = SimpleNamespace(
        set_device=fail if failure == "device" else lambda d: None)
    driver = SimpleNamespace(get_device_interface=fail
                             if failure == "driver" else lambda: interface,
                             get_active_torch_device=lambda: "gcu:0",
                             get_current_target=lambda: "gcu")
    modules = {
        "torch":
        SimpleNamespace(device=lambda d: SimpleNamespace(type="gcu")),
        "triton":
        SimpleNamespace(runtime=SimpleNamespace(driver=SimpleNamespace(
            active=driver)))
    }
    original_import = builtins.__import__

    def load(name, *args, **kwargs):
        if name in modules:
            if name == failure:
                raise ImportError("broken " + name)
            return modules[name]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", load)
    monkeypatch.setattr(runner.importlib, "import_module",
                        fail if failure == "extension" else lambda n: None)
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps(dict(op="abs", case_id="small", body_sha256="test")))
    result = runner.worker(
        SimpleNamespace(worker=case,
                        stage="execute",
                        level=1,
                        torch_extension="device_extension",
                        device=0,
                        result=tmp_path / "result.json"))
    assert result["status"] == "SETUP_ERROR"
    assert "broken " + failure in result["error"]
