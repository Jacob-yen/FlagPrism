from types import SimpleNamespace

import pytest

from flagtree.debugger import native


def test_compiler_binding_uses_libtriton_plugin(monkeypatch):
    compiler = object()
    libtriton = SimpleNamespace(debugger=compiler)
    monkeypatch.setattr(
        native,
        "_optional_module",
        lambda name: libtriton if name == "triton._C.libtriton" else None,
    )

    assert native.compiler_binding() is compiler


def test_runtime_binding_uses_standalone_extension(monkeypatch):
    runtime = object()
    monkeypatch.setattr(
        native,
        "_optional_module",
        lambda name: runtime if name == "flagtree.debugger._native" else None,
    )

    assert native.runtime_binding() is runtime


@pytest.mark.parametrize("level", [-1, 0, 3])
def test_runtime_binding_rejects_invalid_record_level(level):
    runtime = native.runtime_binding()
    if runtime is None:
        pytest.skip("flagtree-debugger native binding is unavailable")

    with pytest.raises(ValueError, match="record_level must be 1 or 2"):
        runtime.prepare_launch({"debug_record_level": level}, 0, None)
