# Debugger TODO

## Precision conversion error report

- [x] Add statement-level diagnostics for direct `arith.truncf` and
  `arith.extf` conversions, covering fp16, bf16 and fp32 numeric comparisons.
- [ ] Extend precision diagnostics across longer producer/consumer compute
  chains so accumulated drift can be attributed to the responsible region.

Target presentation:

```text
precision_conversion:
  from              : [operand x] tensor<16xf16>
  to                : [result y] tensor<16xf32>
  comparison_basis  : numeric value before/after conversion
  max_abs_error     : [0 (F32)]
  mean_abs_error    : [0 (F32)]
  max_rel_error     : [0 (F32)]
  rms_error         : [0 (F32)]
  l2_error          : [0 (F32)]
  suspicious_lanes  : []
  worst_lane        : []
  tolerance:
    abs             : [0.001 (F32)]
    rel             : [0.0005 (F32)]
  status            : [ok]
```

Implementation notes:

- Direct conversion diagnostics use level2 post-processing: read the existing statement JSON and
  `*_value.npy` artifacts, identify cast/convert producer-consumer pairs, and
  compute absolute/relative/RMS/L2 error metrics plus worst lane information.
- Direct-cast `precision_conversion` is integrated into statement/op-log text,
  JSON and `tensor_index.json`; `precision_error` or `precision_drift` for
  longer compute chains remains future work.
- Keep the no-error case explicit: near-zero error should render with
  `status: [ok]`, empty `suspicious_lanes`, and no warning wording.
- Current full dump normalizes fp16/bf16/fp32 values to float32 artifacts, which
  is enough for numeric error comparison but not for bit-exact rounding or ULP
  analysis.
- Add raw fp16/bf16 bit-pattern dump support later if bit-level rounding checks
  are required.

## Value layout and stride report cleanup

- [x] Remove meaningless `stride: unknown` and `layout: unknown` rows from
  statement/op reports when the value is a Triton SSA register value or pointer
  lane tensor and no compiler encoding was captured.

Design rules:

- For intermediate register values, keep only facts that are meaningful for
  debugging: `dtype`, `logical_shape`, `element_count`, `dump_shape`,
  `lane_order`, and optional `compiler_layout` when TTGIR/lower metadata has an
  actual encoding.
- Do not infer memory-style stride for register values. A value such as
  `tensor<16xf32>` or `tensor<16x!tt.ptr<f32>>` has logical lanes, not a
  PyTorch-style runtime stride.
- Runtime tensor stride is still important for kernel input/output tensors
  because it identifies contiguous, transposed, or view-like inputs.
- Address stride is important for pointer/address access because it describes
  the actual lane-wise memory access pattern.

Target runtime tensor presentation:

```text
runtime_tensor:
  name   : x
  shape  : [128, 256]
  stride : [256, 1]
  layout : contiguous
```

Target pointer/address access presentation:

```text
address_summary(load from):
  address_span_bytes  : [64]
  active_lane_count   : [16]
  address_stride_bytes: [4]
  address_pattern     : [contiguous]
```

Target register value presentation:

```text
value_layout:
  kind          : register
  logical_shape : [16]
  element_count : [16]
  dump_shape    : [16]
  lane_order    : debugger_flattened
```

If compiler encoding is captured after layout assignment, extend the same block:

```text
value_layout:
  kind           : register
  logical_shape  : [16]
  element_count  : [16]
  dump_shape     : [16]
  lane_order     : debugger_flattened
  compiler_layout: #ttg.blocked<{...}>
  layout_stage   : ttgir
```

## Sensitive-operation diagnostics

- [ ] Add `denom_near_zero_count` for division/remainder-like operations.
- [ ] Add `neg_sqrt_count` for square-root operations.
- [ ] Add bounded finite-value samples where a summary alone is insufficient.

These checks require operation-specific operand capture and tolerances; they
must not infer warnings from unrelated result summaries.
