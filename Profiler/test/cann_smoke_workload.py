"""Isolated vector-add workload for the CANN direct-finalize regression."""

import argparse
from pathlib import Path

import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--vendor-output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 0:
        parser.error("iters must be positive and warmup must be nonnegative")

    import torch
    import torch_npu  # noqa: F401
    import flagtree.profiler as profiler
    from flagtree.profiler.native import runtime_binding

    torch.npu.set_device(args.device)
    n = 1_048_576
    x = torch.randn(n, device=f"npu:{args.device}", dtype=torch.float32)
    y = torch.randn_like(x)
    out = torch.empty_like(x)

    def launch():
        _vector_add_kernel[(triton.cdiv(n, 1024), )](x,
                                                     y,
                                                     out,
                                                     n,
                                                     BLOCK_SIZE=1024)

    for _ in range(args.warmup + args.iters):
        launch()
    torch.npu.synchronize()
    Path(args.name).parent.mkdir(parents=True, exist_ok=True)
    vendor_output = Path(args.vendor_output)
    vendor_output.mkdir(parents=True, exist_ok=True)
    vendor_output.chmod(0o700)
    session = profiler.start(
        name=args.name,
        context="shadow",
        data="tree",
        hook="triton",
        backend="cann",
        mode=("runtime_base:vendor_metrics=aicore,bandwidth:"
              f"aclprof_output_path={vendor_output}:"
              "runtime_host_timing_fallback=true:aclprof_runtime_enabled=true:"
              "aclprof_auto_export=true:mstx_enabled=true:"
              "mstx_domain=flagtree_profiler"),
    )
    native = runtime_binding()
    scope = "flagtree_profiler_cann_triton::triton_vector_add_fp32"
    try:
        scope_id = native.record_scope()
        native.enter_op(scope_id, scope)
        try:
            for _ in range(args.iters):
                launch()
            torch.npu.synchronize()
        finally:
            native.exit_op(scope_id, scope)
    finally:
        profiler.finalize(session)
    torch.testing.assert_close(out, x + y)


if __name__ == "__main__":
    main()
