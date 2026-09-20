# FlagTree Profiler 测试指南

日常跨后端批量算子验收统一使用根目录 `test.py`，见 [统一测试说明](../../docs/TESTING.md)。以下保留组件专项测试和使用说明。

本文保留 CANN 后端的专项测试说明；跨后端批量算子验收使用根目录统一入口。

## 环境准备

在容器或机器中先加载 CANN 环境：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_VISIBLE_DEVICES=0
```

确认基础依赖存在。默认 IR instrumentation 路径不要求 `msprof`；如果要跑 CANN
legacy 或 CSV import 验证，再检查 `which msprof`。

```bash
python3 - <<'PY'
import torch
import torch_npu
import triton
print("torch:", torch.__version__)
print("npu available:", torch.npu.is_available())
print("triton:", triton.__version__)
PY
which msprof  # only required by legacy CANN/msprof tests
```

## 1. 默认 smoke 测试

这是默认自动化测试，包含 CSV import 单元测试和真实 NPU direct-finalize 测试。

```bash
python3 -m pytest -q third_party/FlagPrism/Profiler/test/test_cann_smoke.py -s
```

预期结果：

```text
12 passed
```

这个测试验证：

- `backend="cann"` 可以通过 `profiler.start()` / `profiler.finalize()` 正常工作。
- CANN CSV import 能解析 op summary、MSTX、bandwidth。
- bandwidth 可以从 op summary byte counters 或 supplemental CSV 中获得。
- 真实 Triton kernel 可以通过 `hook="triton"` 被 profile。
- `finalize()` 后直接生成 `vendor.json`、`timeline.json`、`meta.json`。

兼容旧路径的命令仍可收集同一组测试：

```bash
python3 -m pytest -q third_party/FlagPrism/Profiler/test/test_cann_smoke.py -s
```

## 2. 统一算子验收

在 FlagPrism 根目录运行：

```bash
python3 test.py
python3 test.py --stages profiler
```

统一入口使用仓库内自带的 121 个轻量 Triton 算子，不依赖 FlagGems 或 Liger-Kernel。
完整阶段顺序、失败补跑政策、参数及输出见 [统一测试说明](../../docs/TESTING.md)。
CANN 的 CSV 导入、MSTX、bandwidth 和 direct-finalize 专项回归仍由 `test/test_cann_smoke.py` 保留；
它的单算子子进程位于 `test/cann_smoke_workload.py`，不再依赖批量 runner。

## 3. 最小 direct-finalize 手工测试

如果只想验证用户 API 是否可用，可以写一个最小 Triton workload，外层只包：

```python
sid = profiler.start(
    name="/tmp/my_triton_profile/profile",
    context="shadow",
    data="tree",
    backend="cann",
    hook="triton",
    mode=(
        "runtime_base:"
        "vendor_metrics=aicore,bandwidth:"
        "mstx_enabled=true:"
        "mstx_domain=flagtree_profiler"
    ),
)

# run Triton kernels

profiler.finalize(sid)
```

检查输出：

```bash
ls /tmp/my_triton_profile
python3 - <<'PY'
import json
base = "/tmp/my_triton_profile/profile"
vendor = json.load(open(base + ".vendor.json"))
meta = json.load(open(base + ".meta.json"))
print("backend:", meta.get("backend"))
print("raw inputs:", len(vendor.get("raw_inputs", [])))
print("associations:", len(vendor.get("associations", [])))
print("sources:", sorted({a.get("source") for a in vendor.get("associations", []) if a.get("source")}))
PY
```

当前昇腾默认行为下，`hook="triton"` 会启用 IR instrumentation，并关闭 CANN
legacy `aclprof/msprof`。预期仍然只生成标准四个输出文件：

```text
/tmp/my_triton_profile/profile.hatchet
/tmp/my_triton_profile/profile.meta.json
/tmp/my_triton_profile/profile.timeline.json
/tmp/my_triton_profile/profile.vendor.json
```

内部 IR op timeline 会合并进 `profile.timeline.json`，Hatchet 中会增加
`flagtree.ir.*` / `flagtree.internal.*` 指标。最小检查：

```bash
python3 - <<'PY'
import json
base = "/tmp/my_triton_profile/profile"
trace = json.load(open(base + ".timeline.json"))
hatchet = json.load(open(base + ".hatchet"))
events = [
    event for event in trace["traceEvents"]
    if event.get("cat") == "flagtree.kernel_internal"
]
metrics = set()
def walk(node):
    metrics.update(node.get("metrics", {}).keys())
    for child in node.get("children", []):
        walk(child)
walk(hatchet[0])
print("internal_timeline_events:", len(events))
print("first_internal_event:", events[0] if events else None)
print("ir_metrics:", sorted(m for m in metrics if m.startswith("flagtree.ir.")))
PY
```

预期 `internal_timeline_events > 0`，并能看到 `flagtree.ir.duration_cycle`
等指标。这些 records 来自设备端 debug ring buffer，不是静态 IR metadata。
默认 IR 路径没有 CANN kernel event 时，会在 `profile.timeline.json` 中生成
`flagtree.ir_kernel` synthetic kernel event，并把内部 timeline 事件的 `ts` /
`dur` 映射到该窗口内；原始设备 `SYS_CNT` cycle 会保留在
`args.start_cycle`、`args.end_cycle`、`args.duration_cycle` 中。Timeline 默认覆盖
collect region 内非 constant tracked IR op，轻量 op 可能因为计数器分辨率显示为
`duration_cycle=0`。

自动插桩的 record buffer 默认是 32 MiB（`524288` 条、64 bytes/条），每次
kernel launch 导出后释放。设置 `FLAGTREE_PROFILER_IR_RECORD_BUFFER_MB=<MiB>` 可以覆盖默认
值；该值会进入编译 cache key，因此修改后会自动重新编译插桩 kernel。更大的
buffer 能覆盖更多 program instance，但也会增加设备侧插桩扰动、导出时间和
`timeline.json` 体积。

如果要验证旧 CANN legacy 路径，运行前设置：

```bash
export FLAGTREE_PROFILER_CANN_TRITON_HOOK_LEGACY=1
```

此时 `hook="triton"` 会恢复旧行为，不写入 `flagtree.kernel_internal` 事件或
`flagtree.ir.*` 指标。正常情况下，`sources` 至少应包含部分以下来源：

```text
aclprof_op_summary
msprof_mstx
msprof_bandwidth
msprof_api_statistic
msprof_op_statistic
```

## 4. 结果文件怎么看

主要输出文件：

- `*.meta.json`：backend、mode 配置、启用的 vendor metrics、degrade reasons。
- `*.vendor.json`：CANN CSV 导入结果、metric association、bandwidth、MSTX range。
- `*.timeline.json`：Chrome trace 格式时间线。
- `*.hatchet`：FlagTree Profiler/Hatchet 聚合视图。
- `summary.json`：suite 脚本的汇总结果。

常见字段：

- `degrade_reasons`：非致命降级原因。例如 task_time 不完整时使用 host timing fallback。
- `association_sources`：vendor 数据来源统计。
- `bandwidth_association_count`：包含 `bandwidth_gb_s` 的 association 数量。
- `mstx_ranges`：FlagTree Profiler scope / Triton hook 进入 CANN profiler 后导出的 range。
- `top_op_types`：CANN op summary 中出现次数最多的 op 类型。

## 5. 常见问题

### `msprof` 找不到

先确认 CANN 环境：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
which msprof
```

### NPU 不可用

确认：

```bash
python3 - <<'PY'
import torch
import torch_npu
print(torch.npu.is_available())
PY
```

### 输出目录权限问题

CANN/msprof 对输出目录权限较敏感。建议使用独立目录，并确保不是 group/other writable：

```bash
mkdir -p /tmp/my_triton_profile/msprof
chmod 700 /tmp/my_triton_profile /tmp/my_triton_profile/msprof
```

### `degrade_reasons` 不为空

不一定表示测试失败。常见情况是 CANN runtime event 不完整，Profiler 默认使用 host timing fallback 保留基础关联。需要结合 `association_sources`、`bandwidth_association_count` 和 `mstx_ranges` 判断是否采到了核心数据。
