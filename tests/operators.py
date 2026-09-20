"""Self-contained, small Triton operators for debugger/profiler acceptance.

These are correctness probes, not optimized production kernels. Each named
operation has a CPU PyTorch reference; all stages use identical fixed launch
parameters. Keep CASES literal so the runner can list it without importing a
vendor runtime. No third-party operator library or dispatch patching is used.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as lib

CASES = [
    {
        "op": "abs",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.abs(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "neg",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = -x",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "exp",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.exp(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "exp2",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.exp2(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "expm1",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.expm1(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op":
        "log",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x.abs() + 1.1",
        "operation":
        "result = torch.log(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op":
        "log2",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x.abs() + 1.1",
        "operation":
        "result = torch.log2(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op":
        "log10",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x.abs() + 1.1",
        "operation":
        "result = torch.log10(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op":
        "log1p",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x / 4.0",
        "operation":
        "result = torch.log1p(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op":
        "sqrt",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x.abs() + 1.1",
        "operation":
        "result = torch.sqrt(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op":
        "rsqrt",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x.abs() + 1.1",
        "operation":
        "result = torch.rsqrt(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op":
        "reciprocal",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x.abs() + 1.1",
        "operation":
        "result = torch.reciprocal(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op": "sin",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.sin(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "cos",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.cos(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "tan",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.tan(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op":
        "asin",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x / 4.0",
        "operation":
        "result = torch.asin(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op":
        "acos",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x / 4.0",
        "operation":
        "result = torch.acos(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op": "atan",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.atan(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "sinh",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.sinh(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "cosh",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.cosh(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "tanh",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.tanh(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "asinh",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.asinh(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op":
        "acosh",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x.abs() + 1.1",
        "operation":
        "result = torch.acosh(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op":
        "atanh",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x / 4.0",
        "operation":
        "result = torch.atanh(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op": "erf",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.erf(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "erfc",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.erfc(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "sigmoid",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.sigmoid(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "relu",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.relu(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "relu6",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.nn.functional.relu6(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "silu",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.nn.functional.silu(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "gelu",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.nn.functional.gelu(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "leaky_relu",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.nn.functional.leaky_relu(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "elu",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.nn.functional.elu(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "selu",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.nn.functional.selu(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "softplus",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.nn.functional.softplus(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "softsign",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.nn.functional.softsign(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "hardsigmoid",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.nn.functional.hardsigmoid(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "hardswish",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.nn.functional.hardswish(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "hardtanh",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.nn.functional.hardtanh(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "floor",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.floor(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "ceil",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.ceil(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "trunc",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.trunc(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op":
        "round",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x[:8] = torch.tensor([-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,-3.5])",
        "operation":
        "result = torch.round(x)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op": "frac",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.frac(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "sign",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.sign(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op":
        "signbit",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        'x[:5] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 0.0])',
        "operation":
        "result = torch.signbit(x)",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "isfinite",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        'x[:5] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 0.0])',
        "operation":
        "result = torch.isfinite(x)",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "isnan",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        'x[:5] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 0.0])',
        "operation":
        "result = torch.isnan(x)",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "isinf",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        'x[:5] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 0.0])',
        "operation":
        "result = torch.isinf(x)",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op": "logical_not",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.logical_not(x)",
        "dtype": "bool",
        "size": 93,
    },
    {
        "op": "square",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.square(x)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "add",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = x+y",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "sub",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = x-y",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "mul",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = x*y",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "div",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = x/y",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "floor_divide",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.floor_divide(x,y)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "remainder",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.remainder(x,y)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op":
        "fmod",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "y[::2] = -y[::2]",
        "operation":
        "result = torch.fmod(x,y)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op":
        "pow",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = x.abs() + 1.1",
        "operation":
        "result = torch.pow(x,y)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op": "minimum",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.minimum(x,y)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "maximum",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.maximum(x,y)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op":
        "eq",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "y[::3] = x[::3]",
        "operation":
        "result = x==y",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "ne",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "y[::3] = x[::3]",
        "operation":
        "result = x!=y",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "lt",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "y[::3] = x[::3]",
        "operation":
        "result = x<y",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "le",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "y[::3] = x[::3]",
        "operation":
        "result = x<=y",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "gt",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "y[::3] = x[::3]",
        "operation":
        "result = x>y",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "ge",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "y[::3] = x[::3]",
        "operation":
        "result = x>=y",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "logical_and",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x[::4] = 0\n"
        "y[::3] = 0",
        "operation":
        "result = torch.logical_and(x,y)",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "logical_or",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x[::4] = 0\n"
        "y[::3] = 0",
        "operation":
        "result = torch.logical_or(x,y)",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "logical_xor",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x[::4] = 0\n"
        "y[::3] = 0",
        "operation":
        "result = torch.logical_xor(x,y)",
        "dtype":
        "bool",
        "size":
        93,
    },
    {
        "op":
        "bitwise_and",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = torch.arange(93,dtype=torch.float32) - 45\n"
        "y = (torch.arange(93)%4).float()",
        "operation":
        "result = x.to(torch.int32)&y.to(torch.int32)",
        "dtype":
        "int32",
        "size":
        93,
    },
    {
        "op":
        "bitwise_or",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = torch.arange(93,dtype=torch.float32) - 45\n"
        "y = (torch.arange(93)%4).float()",
        "operation":
        "result = x.to(torch.int32)|y.to(torch.int32)",
        "dtype":
        "int32",
        "size":
        93,
    },
    {
        "op":
        "bitwise_xor",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = torch.arange(93,dtype=torch.float32) - 45\n"
        "y = (torch.arange(93)%4).float()",
        "operation":
        "result = x.to(torch.int32)^y.to(torch.int32)",
        "dtype":
        "int32",
        "size":
        93,
    },
    {
        "op":
        "bitwise_left_shift",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = torch.arange(93,dtype=torch.float32) - 45\n"
        "y = (torch.arange(93)%4).float()",
        "operation":
        "result = x.to(torch.int32)<<y.to(torch.int32)",
        "dtype":
        "int32",
        "size":
        93,
    },
    {
        "op":
        "bitwise_right_shift",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "x = torch.arange(93,dtype=torch.float32) - 45\n"
        "y = (torch.arange(93)%4).float()",
        "operation":
        "result = x.to(torch.int32)>>y.to(torch.int32)",
        "dtype":
        "int32",
        "size":
        93,
    },
    {
        "op":
        "copysign",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "y[::2] = -y[::2]",
        "operation":
        "result = torch.copysign(x,y)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op":
        "atan2",
        "kind":
        "pointwise",
        "setup":
        "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)\n"
        "y[::2] = -y[::2]",
        "operation":
        "result = torch.atan2(x,y)",
        "dtype":
        "float32",
        "size":
        93,
    },
    {
        "op": "hypot",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.hypot(x,y)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "lerp",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.lerp(x,y,0.25)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "addcmul",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.addcmul(x,y,y,value=0.5)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "addcdiv",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.addcdiv(x,y,y+1.0,value=0.5)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "where",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.where(x>0,x,y)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "clamp",
        "kind": "pointwise",
        "setup": "x = torch.linspace(-2.7, 2.7, 93, dtype=torch.float32)\n"
        "y = torch.linspace(0.3, 1.7, 93, dtype=torch.float32)",
        "operation": "result = torch.clamp(x,-0.5,0.5)",
        "dtype": "float32",
        "size": 93,
    },
    {
        "op": "sum",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.sum(1)",
        "dtype": "float32",
        "size": 4,
    },
    {
        "op": "mean",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.mean(1)",
        "dtype": "float32",
        "size": 4,
    },
    {
        "op": "prod",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.prod(1)",
        "dtype": "float32",
        "size": 4,
    },
    {
        "op": "amin",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.amin(1)",
        "dtype": "float32",
        "size": 4,
    },
    {
        "op": "amax",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.amax(1)",
        "dtype": "float32",
        "size": 4,
    },
    {
        "op": "argmin",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.argmin(1).to(torch.int32)",
        "dtype": "int32",
        "size": 4,
    },
    {
        "op": "argmax",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.argmax(1).to(torch.int32)",
        "dtype": "int32",
        "size": 4,
    },
    {
        "op": "all",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = (x>0).all(1)",
        "dtype": "bool",
        "size": 4,
    },
    {
        "op": "any",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = (x>0).any(1)",
        "dtype": "bool",
        "size": 4,
    },
    {
        "op": "var",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.var(1,correction=0)",
        "dtype": "float32",
        "size": 4,
    },
    {
        "op": "std",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.std(1,correction=0)",
        "dtype": "float32",
        "size": 4,
    },
    {
        "op": "logsumexp",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = torch.logsumexp(x,1)",
        "dtype": "float32",
        "size": 4,
    },
    {
        "op": "softmax",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.softmax(1)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "log_softmax",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.log_softmax(1)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "cumsum",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.cumsum(1)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "cumprod",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = x.cumprod(1)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "layer_norm",
        "kind": "reduction",
        "setup": "x = torch.linspace(-0.8, 1.2, 64).reshape(4,16)",
        "operation": "result = torch.nn.functional.layer_norm(x,(16,))",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "transpose",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = x.reshape(8,8).T.contiguous().reshape(-1)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "flip",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = x.flip(0)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "roll",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.roll(x,3)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "repeat",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = x.repeat(2)",
        "dtype": "float32",
        "size": 128,
    },
    {
        "op": "repeat_interleave",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = x.repeat_interleave(2)",
        "dtype": "float32",
        "size": 128,
    },
    {
        "op": "slice",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = x[::2].clone()",
        "dtype": "float32",
        "size": 32,
    },
    {
        "op": "gather",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.gather(x,0,index.long())",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "index_select",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.index_select(x,0,index.long())",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "cat",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.cat((x,x))",
        "dtype": "float32",
        "size": 128,
    },
    {
        "op": "stack",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.stack((x,x)).reshape(-1)",
        "dtype": "float32",
        "size": 128,
    },
    {
        "op": "diagonal",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = x.reshape(8,8).diagonal()",
        "dtype": "float32",
        "size": 8,
    },
    {
        "op": "tril",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.tril(x.reshape(8,8)).reshape(-1)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "triu",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.triu(x.reshape(8,8)).reshape(-1)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "pad",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.nn.functional.pad(x,(2,2))",
        "dtype": "float32",
        "size": 68,
    },
    {
        "op": "eye",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.eye(8).reshape(-1)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "full",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.full((64,),2.5)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "arange",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.arange(64,dtype=torch.float32)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op": "scatter",
        "kind": "index",
        "setup":
        "x = torch.linspace(-2, 2, 64)\nindex = (torch.arange(64,dtype=torch.int32)*17+3)%64",
        "operation": "result = torch.zeros_like(x).scatter(0,index.long(),x)",
        "dtype": "float32",
        "size": 64,
    },
    {
        "op":
        "mm",
        "kind":
        "matmul",
        "setup":
        "x = torch.randn((16,16)) * 0.2\n"
        "y = torch.randn((16,16)) * 0.2\n"
        "bias = torch.randn((16,16)) * 0.2",
        "operation":
        "result = x @ y",
        "dtype":
        "float32",
        "size":
        256,
    },
    {
        "op":
        "bmm",
        "kind":
        "matmul",
        "setup":
        "x = torch.randn((2,16,16)) * 0.2\n"
        "y = torch.randn((2,16,16)) * 0.2\n"
        "bias = torch.randn((16,16)) * 0.2",
        "operation":
        "result = torch.bmm(x,y)",
        "dtype":
        "float32",
        "size":
        512,
    },
    {
        "op":
        "addmm",
        "kind":
        "matmul",
        "setup":
        "x = torch.randn((16,16)) * 0.2\n"
        "y = torch.randn((16,16)) * 0.2\n"
        "bias = torch.randn((16,16)) * 0.2",
        "operation":
        "result = torch.addmm(bias,x,y)",
        "dtype":
        "float32",
        "size":
        256,
    },
]


@triton.jit
def _pointwise(X, Y, Out, N: tl.constexpr, OP: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, other=1.0)
    y = tl.load(Y + i, i < N, other=1.0)
    if OP == "abs":
        result = tl.abs(x)
    elif OP == "neg":
        result = -x
    elif OP == "exp":
        result = tl.exp(x)
    elif OP == "exp2":
        result = tl.exp2(x)
    elif OP == "expm1":
        result = tl.exp(x) - 1.0
    elif OP == "log":
        result = tl.log(x)
    elif OP == "log2":
        result = tl.log2(x)
    elif OP == "log10":
        result = tl.log(x) * 0.4342944819032518
    elif OP == "log1p":
        result = tl.log(1.0 + x)
    elif OP == "sqrt":
        result = tl.sqrt(x)
    elif OP == "rsqrt":
        result = tl.rsqrt(x)
    elif OP == "reciprocal":
        result = 1.0 / x
    elif OP == "sin":
        result = tl.sin(x)
    elif OP == "cos":
        result = tl.cos(x)
    elif OP == "tan":
        result = tl.sin(x) / tl.cos(x)
    elif OP == "asin":
        result = 1.5707963267948966 - lib.acos(x)
    elif OP == "acos":
        result = lib.acos(x)
    elif OP == "atan":
        result = lib.atan(x)
    elif OP == "sinh":
        result = (tl.exp(x) - tl.exp(-x)) * 0.5
    elif OP == "cosh":
        result = (tl.exp(x) + tl.exp(-x)) * 0.5
    elif OP == "tanh":
        result = 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0
    elif OP == "asinh":
        result = tl.log(x + tl.sqrt(x * x + 1.0))
    elif OP == "acosh":
        result = tl.log(x + tl.sqrt(x * x - 1.0))
    elif OP == "atanh":
        result = 0.5 * tl.log((1.0 + x) / (1.0 - x))
    elif OP == "erf":
        result = tl.erf(x)
    elif OP == "erfc":
        result = 1.0 - tl.erf(x)
    elif OP == "sigmoid":
        result = 1.0 / (1.0 + tl.exp(-x))
    elif OP == "relu":
        result = tl.maximum(x, 0.0)
    elif OP == "relu6":
        result = tl.minimum(tl.maximum(x, 0.0), 6.0)
    elif OP == "silu":
        result = x / (1.0 + tl.exp(-x))
    elif OP == "gelu":
        result = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    elif OP == "leaky_relu":
        result = tl.where(x >= 0, x, 0.01 * x)
    elif OP == "elu":
        result = tl.where(x >= 0, x, tl.exp(x) - 1.0)
    elif OP == "selu":
        result = 1.0507009873554805 * tl.where(
            x >= 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    elif OP == "softplus":
        result = tl.log(1.0 + tl.exp(x))
    elif OP == "softsign":
        result = x / (1.0 + tl.abs(x))
    elif OP == "hardsigmoid":
        result = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    elif OP == "hardswish":
        result = x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    elif OP == "hardtanh":
        result = tl.minimum(tl.maximum(x, -1.0), 1.0)
    elif OP == "floor":
        result = tl.floor(x)
    elif OP == "ceil":
        result = tl.ceil(x)
    elif OP == "trunc":
        result = tl.where(x < 0, tl.ceil(x), tl.floor(x))
    elif OP == "round":
        lower = tl.floor(x)
        fraction = x - lower
        round_up = (fraction > 0.5) | ((fraction == 0.5) &
                                       ((lower.to(tl.int32) & 1) != 0))
        result = tl.where(round_up, lower + 1.0, lower)
    elif OP == "frac":
        result = x - tl.where(x < 0, tl.ceil(x), tl.floor(x))
    elif OP == "sign":
        result = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
    elif OP == "signbit":
        result = x.to(tl.int32, bitcast=True) < 0
    elif OP == "isfinite":
        result = (x == x) & (tl.abs(x) != float("inf"))
    elif OP == "isnan":
        result = x != x
    elif OP == "isinf":
        result = tl.abs(x) == float("inf")
    elif OP == "logical_not":
        result = x == 0
    elif OP == "square":
        result = x * x
    elif OP == "add":
        result = x + y
    elif OP == "sub":
        result = x - y
    elif OP == "mul":
        result = x * y
    elif OP == "div":
        result = x / y
    elif OP == "floor_divide":
        result = tl.floor(x / y)
    elif OP == "remainder":
        result = x - tl.floor(x / y) * y
    elif OP == "fmod":
        quotient = x / y
        quotient = tl.where(quotient < 0, tl.ceil(quotient),
                            tl.floor(quotient))
        remainder = tl.fma(-quotient, y, x)
        # Correct a quotient rounded across an integer boundary; FMA retains
        # the residual's sign even when x/y rounds to that integer in f32.
        crossed = ((remainder > 0) & (x < 0)) | ((remainder < 0) & (x > 0))
        result = tl.where(crossed,
                          remainder + tl.where(x < 0, -tl.abs(y), tl.abs(y)),
                          remainder)
    elif OP == "pow":
        result = tl.exp(y * tl.log(x))
    elif OP == "minimum":
        result = tl.minimum(x, y)
    elif OP == "maximum":
        result = tl.maximum(x, y)
    elif OP == "eq":
        result = x == y
    elif OP == "ne":
        result = x != y
    elif OP == "lt":
        result = x < y
    elif OP == "le":
        result = x <= y
    elif OP == "gt":
        result = x > y
    elif OP == "ge":
        result = x >= y
    elif OP == "logical_and":
        result = (x != 0) & (y != 0)
    elif OP == "logical_or":
        result = (x != 0) | (y != 0)
    elif OP == "logical_xor":
        result = (x != 0) ^ (y != 0)
    elif OP == "bitwise_and":
        result = x.to(tl.int32) & y.to(tl.int32)
    elif OP == "bitwise_or":
        result = x.to(tl.int32) | y.to(tl.int32)
    elif OP == "bitwise_xor":
        result = x.to(tl.int32) ^ y.to(tl.int32)
    elif OP == "bitwise_left_shift":
        result = x.to(tl.int32) << y.to(tl.int32)
    elif OP == "bitwise_right_shift":
        result = x.to(tl.int32) >> y.to(tl.int32)
    elif OP == "copysign":
        result = tl.where(y < 0, -tl.abs(x), tl.abs(x))
    elif OP == "atan2":
        result = lib.atan2(x, y)
    elif OP == "hypot":
        result = tl.sqrt(x * x + y * y)
    elif OP == "lerp":
        result = x + 0.25 * (y - x)
    elif OP == "addcmul":
        result = x + 0.5 * y * y
    elif OP == "addcdiv":
        result = x + 0.5 * y / (y + 1.0)
    elif OP == "where":
        result = tl.where(x > 0, x, y)
    elif OP == "clamp":
        result = tl.minimum(tl.maximum(x, -0.5), 0.5)
    tl.store(Out + i, result, i < N)


@triton.jit
def _multiply(a, b):
    return a * b


@triton.jit
def _reduction(X, Out, OP: tl.constexpr, VECTOR: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, 16)
    x = tl.load(X + row * 16 + i)
    if OP == "sum":
        result = tl.sum(x, 0)
    elif OP == "mean":
        result = tl.sum(x, 0) / 16
    elif OP == "prod":
        result = tl.reduce(x, 0, _multiply)
    elif OP == "amin":
        result = tl.min(x, 0)
    elif OP == "amax":
        result = tl.max(x, 0)
    elif OP == "argmin":
        result = tl.argmin(x, 0)
    elif OP == "argmax":
        result = tl.argmax(x, 0)
    elif OP == "all":
        result = tl.min((x > 0).to(tl.int32), 0)
    elif OP == "any":
        result = tl.max((x > 0).to(tl.int32), 0)
    elif OP == "var":
        result = tl.sum(
            (x - tl.sum(x, 0) / 16) * (x - tl.sum(x, 0) / 16), 0) / 16
    elif OP == "std":
        result = tl.sqrt(
            tl.sum((x - tl.sum(x, 0) / 16) * (x - tl.sum(x, 0) / 16), 0) / 16)
    elif OP == "logsumexp":
        result = tl.log(tl.sum(tl.exp(x), 0))
    elif OP == "softmax":
        result = tl.exp(x - tl.max(x, 0)) / tl.sum(tl.exp(x - tl.max(x, 0)), 0)
    elif OP == "log_softmax":
        result = x - tl.max(x, 0) - tl.log(tl.sum(tl.exp(x - tl.max(x, 0)), 0))
    elif OP == "cumsum":
        result = tl.cumsum(x, 0)
    elif OP == "cumprod":
        result = tl.associative_scan(x, 0, _multiply)
    elif OP == "layer_norm":
        result = (x - tl.sum(x, 0) / 16) * tl.rsqrt(
            tl.sum((x - tl.sum(x, 0) / 16) *
                   (x - tl.sum(x, 0) / 16), 0) / 16 + 1.0e-5)
    if VECTOR:
        tl.store(Out + row * 16 + i, result)
    else:
        tl.store(Out + row, result)


@triton.jit
def _index(X, I, Out, N: tl.constexpr, OP: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < N
    if OP == "transpose":
        offset = (i % 8) * 8 + i // 8
        result = tl.load(X + offset, i < N, other=0.0)
    elif OP == "flip":
        offset = 63 - i
        result = tl.load(X + offset, i < N, other=0.0)
    elif OP == "roll":
        offset = (i + 61) % 64
        result = tl.load(X + offset, i < N, other=0.0)
    elif OP == "repeat":
        offset = i % 64
        result = tl.load(X + offset, i < N, other=0.0)
    elif OP == "repeat_interleave":
        offset = i // 2
        result = tl.load(X + offset, i < N, other=0.0)
    elif OP == "slice":
        offset = i * 2
        result = tl.load(X + offset, i < N, other=0.0)
    elif OP == "gather":
        offset = tl.load(I + i, i < N, other=0)
        result = tl.load(X + offset, i < N, other=0.0)
    elif OP == "index_select":
        offset = tl.load(I + i, i < N, other=0)
        result = tl.load(X + offset, i < N, other=0.0)
    elif OP == "cat":
        offset = i % 64
        result = tl.load(X + offset, i < N, other=0.0)
    elif OP == "stack":
        offset = i % 64
        result = tl.load(X + offset, i < N, other=0.0)
    elif OP == "diagonal":
        offset = i * 9
        result = tl.load(X + offset, i < N, other=0.0)
    elif OP == "tril":
        offset = i
        result = tl.load(X + offset, i < N, other=0.0)
        result = tl.where(i // 8 >= i % 8, result, 0.0)
    elif OP == "triu":
        offset = i
        result = tl.load(X + offset, i < N, other=0.0)
        result = tl.where(i // 8 <= i % 8, result, 0.0)
    elif OP == "pad":
        offset = i - 2
        result = tl.load(X + offset, (i >= 2) & (i < 66), other=0.0)
    elif OP == "eye":
        result = (i // 8 == i % 8).to(tl.float32)
    elif OP == "full":
        result = tl.full((B, ), 2.5, tl.float32)
    elif OP == "arange":
        result = i.to(tl.float32)
    elif OP == "scatter":
        result = tl.load(X + i, i < N, other=0.0)
        i = tl.load(I + i, i < N, other=0)
    tl.store(Out + i, result, mask)


@triton.jit
def _matmul(X, Y, Bias, Out, ADD: tl.constexpr):
    batch = tl.program_id(0)
    i = tl.arange(0, 16)
    a = tl.load(X + batch * 256 + i[:, None] * 16 + i[None, :])
    b = tl.load(Y + batch * 256 + i[:, None] * 16 + i[None, :])
    result = tl.dot(a, b, input_precision="ieee")
    if ADD:
        result += tl.load(Bias + i[:, None] * 16 + i[None, :])
    tl.store(Out + batch * 256 + i[:, None] * 16 + i[None, :], result)


def run(case, namespace):
    """Allocate outside the kernel; execute exactly one fixed-config Triton launch."""
    op, kind = case["op"], case["kind"]
    out = torch.empty((case["size"], ),
                      device=namespace["device"],
                      dtype=getattr(torch, case["dtype"]))
    if kind == "pointwise":
        _pointwise[(triton.cdiv(case["size"],
                                32), )](namespace["x"], namespace["y"], out,
                                        case["size"], op, 32)
    elif kind == "reduction":
        _reduction[(4, )](namespace["x"], out, op, case["size"] == 64)
        if case["size"] == 64:
            out = out.reshape(4, 16)
    elif kind == "index":
        _index[(triton.cdiv(case["size"],
                            16), )](namespace["x"], namespace["index"], out,
                                    case["size"], op, 16)
    elif kind == "matmul":
        _matmul[(2 if op == "bmm" else 1, )](namespace["x"], namespace["y"],
                                             namespace["bias"], out,
                                             op == "addmm")
        out = out.reshape((2, 16, 16) if op == "bmm" else (16, 16))
    else:
        raise ValueError(f"Unknown operator kind: {kind}")
    return out
