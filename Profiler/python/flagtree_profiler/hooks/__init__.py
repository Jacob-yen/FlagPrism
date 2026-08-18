# ruff: noqa
from .hook import HookManager
try:
    from .instrumentation import InstrumentationHook
except (ImportError, RuntimeError):
    class InstrumentationHook:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "FlagTree instrumentation hook requires the Proton compiler "
                "component; use hook='triton' for the MThreads profiler."
            )
from .launch import LaunchHook
