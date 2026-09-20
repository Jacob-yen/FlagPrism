# 统一算子验收

日常验收只运行仓库根目录的一个入口：

```bash
python3 test.py
```

测试不需要 FlagGems。算子实现、固定输入和 PyTorch CPU 参考统一放在
[`tests/operators.py`](../tests/operators.py)，只依赖目标设备的 FlagTree/FlagPrism、PyTorch
及对应设备扩展，以及 NumPy。该文件是轻量功能探针，不是生产算子库或性能 benchmark。

## 覆盖

当前有 **121 个不同名称的算子**：逐元素数学、激活、比较、逻辑/位运算、归约、扫描、
softmax、layer norm、矩阵乘法、索引及数据重排。不用不同 shape、别名、原地/out 变体凑数量。
`cat` 与 `stack` 等 API 可能共享底层访存模式；计数表示算子 API 覆盖，不代表同等数量的编译器路径。

这些是仓库自带的 Triton 实现，替代了以前基于 FlagGems 的 173 ID / 176 case 清单；
**两套清单不能直接比较通过率**。每个算子使用明确固定的轻量 shape/dtype，三阶段和补跑
使用相同的输入、kernel 和配置；不再 monkeypatch autotuner 或进行候选配置搜索。
没有验证所有 shape/dtype、调优配置或厂商生产算子的性能。

```bash
python3 test.py --list         # 生成清单，不需要设备或导入 Torch/Triton
python3 test.py --check-cases  # 验证所有 CPU 参考
```

默认最低覆盖要求是 100 个算子族；单算子诊断需要显式降低门槛：

```bash
python3 test.py --ops abs,add,addmm,sum --min-ops 1
```

## 顺序与判定

每个 case 依次运行：

1. `debugger_l1`：与 CPU 结果比较；检查动态数值记录、无溢出和完整报告。
2. `debugger_l2`：再次比较结果；检查 FULL_VALUE、`.npy` shape/字节数、报告及每次 kernel 的采集覆盖。
3. `profiler`：再次比较结果；要求真实 Triton kernel 对应的设备耗时，host timing fallback 不算通过。

采集失败后，在同一卡的独立进程中补跑无 debugger/profiler 的同一算子：
普通执行也失败 → **WARNING，按约定放行**；普通执行成功 → **ERROR，不放行**。
WARNING 不代表实际采集成功，结果保留两次执行的错误、日志和退出码。
PyTorch/Triton/设备扩展导入或驱动、设备初始化失败（包括 baseline）为 ERROR。
工具缺失、导入失败、原生绑定不可用或预检查失败为 ERROR，不适用算子补跑放行规则。
设备 context/Sip/离线、NPU core 异常及 CUDA/HIP 非法内存访问或 device-side assert 为 ERROR，并阻止后续测试；不能用损坏设备上的补跑失败作 WARNING。

所有阶段均须为 PASS/WARNING，且达到最低算子族门槛，整体才通过。
显式 `--stages execute` 的失败直接判失败。

输入先在 CPU 用固定 seed 生成，再复制到设备。每个 case/阶段使用独立进程，
只执行一次固定配置的被测 kernel，要求存在实际 Triton launch。
L2 的全面内容精度、地址语义、协议 ABI 等边界另有专项回归，不能仅靠 artifact 非空证明。

## 参数

```bash
python3 test.py --stages debugger             # L1 和 L2
python3 test.py --stages debugger --level 2   # 仅 L2
python3 test.py --stages profiler
python3 test.py --stages execute
python3 test.py --devices 0,1,2,3 --jobs 4
python3 test.py --torch-extension torch_gcu --profiler-backend enflame
```

通过 Triton active driver 选择设备与同步接口，可显式指定 PyTorch 扩展和 profiler 后端。
多卡模式每张卡同一时间只运行一个 case；默认单进程串行。`--jobs > 1` 必须显式指定
`--devices`，实际并发数不超过指定设备数量。并行耗时不用于性能比较。
后端名称映射不是跨芯片验证结论，各厂商仍须真机运行。

默认 `addr_level=0`、记录容量 65536，可通过 `--addr-level`、`--record-capacity` 修改。
L2 默认超时 600 秒，其余阶段及普通执行补跑为 180 秒；`--timeout` 可统一覆盖。

## 输出

默认写入 `test-results/<UTC 时间>/`，可用 `--out` 修改。

- `manifest.json`：算子清单、仓库版本、算子文件 SHA256、配置和失败政策。
- `cases/<case>/case.json`：共享输入与 CPU 参考。
- `cases/<case>/<stage>/result.json`、`worker.log`：实际结果和诊断。
- `cases/<case>/<stage>/baseline/`：失败后的普通执行结果。
- `summary.json`：每阶段 PASS/WARNING/ERROR 数量与逐项结果。
- 各阶段目录还保留 Debugger/Profiler 原始报告。

`passed_api_ids` / `passed_operator_families` 包含按政策放行的 WARNING；实际采集成功请查看 PASS 数量。

## 专项回归

`Debugger/test/` 和 `Profiler/test/` 保留编译器、协议与厂商边界测试，包括混合宽度 payload、
L1/L2 混合级别和 TOPSPTI 暂停/恢复及会话拒绝后的重试。这些测试不依赖 FlagGems。
旧批量算子 runner、FlagGems 样例及其专属文档已移除。
厂商专项诊断、报告工具和 examples 保留；examples 仅演示 API。

燧原构建说明见 [Enflame 后端](enflame.md)，实际验证范围以对应运行的清单和结果为准。
