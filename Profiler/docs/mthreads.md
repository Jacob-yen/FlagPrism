# Moore Threads profiler support

## Capability contract

The mthreads adapter has two collection paths. They intentionally share one
FlagPrism artifact schema but do not pretend to provide the same measurements.

| Capability | Collector | Status |
| --- | --- | --- |
| Kernel launch geometry and API duration | MUPTI callback | Implemented and hardware-tested |
| Theoretical occupancy | MUSA occupancy API plus device limits | Implemented and hardware-tested |
| Register/shared/local memory resources | MUSA function attributes plus device limits | Implemented and hardware-tested |
| Theoretical peak memory bandwidth | MUSA memory clock and bus width | Implemented and hardware-tested |
| Kernel duration and estimated elapsed cycles | MUPTI activity timestamps | Implemented; activity is opt-in on MUSA 4.3 |
| Instructions, hardware cycles, achieved bandwidth, MP utilization | Moore Perf Compute replay/export | Frozen pending validation in a compatible environment |
| Arbitrary hardware counters | Moore Perf Compute replay/export | Frozen pending validation in a compatible environment |

Callback duration is exported as `mthreads.launch_api_duration_ns`. Only an
activity record or vendor export can produce `mthreads.kernel_duration_ns`.
This prevents host launch overhead from being mislabeled as device execution
time. Missing values are omitted from CSV cells and reported through artifact
degradation rather than represented as zero.

An activity record is accepted only when it has a non-zero device interval.
If an older runtime returns zero timestamps after an MT-Perf connection
failure, FlagPrism discards that record, retains the matching callback launch
data, and reports the number of invalid activity records as a degradation.

## Native validation

Build FlagTree with `FLAGTREE_BACKEND=mthreads`, `FLAGPRISM_BACKEND=mthreads`,
and `TRITON_BUILD_FLAGPRISM=ON`, then run:

```bash
FLAGTREE_RUN_MTHREADS_HARDWARE_TESTS=1 \
python -m pytest -q third_party/FlagPrism/Profiler/test/test_mthreads.py -s
```

The hardware test verifies the vector-add result and requires exactly one
MUPTI row for that kernel. It also verifies that occupancy came from a MUSA
occupancy API, lies in `(0, 100]`, and that register, shared-memory, and peak
bandwidth fields are non-zero.

## Hardware-counter collection

MCU collection and CSV import are retained as dormant implementation code but
are not exposed through the FlagTree Profiler CLI. Requests for MCU-only
metrics are reported as unavailable rather than producing fabricated values.

<!-- TODO(FlagPrism): Enable and validate MCU integration when a compatible
Moore Threads MCU, MUSA SDK, and driver test environment is available. -->
