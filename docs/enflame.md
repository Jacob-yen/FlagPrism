# 燧原后端

FlagPrism 的 Enflame 后端通过 FlagTree 联合构建提供 `flagtree.debugger` 与
`flagtree.profiler`。仅安装原始 FlagTree 燧原 wheel 不会包含这些组件。

## 构建

需要 TOPS SDK、TOPSPTI 开发头文件，以及与 FlagTree 燧原分支匹配的 LLVM 工具链。
当前验证环境为 Python 3.12、PyTorch/torch_gcu 2.10、Triton GCU 3.6、TOPS Runtime 1.9.29。

```bash
cd /path/to/FlagTree
export FLAGTREE_BACKEND=enflame FLAGPRISM_BACKEND=enflame
export FLAGPRISM_SOURCE_DIR=/path/to/FlagPrism
export TRITON_BUILD_FLAGPRISM=ON TRITON_BUILD_PROTON=OFF
export KURAMA_LLVM_DIR=/path/to/llvm-fc83c68-gcc9-x64
export LLVM_SYSPATH="$KURAMA_LLVM_DIR"
MAX_JOBS=24 python3 -m pip install -e . --no-build-isolation --no-deps
```

构建从 `/opt/tops/include`、`/opt/tops/lib` 和
`/opt/tops/extras/TOPSPTI/{include,lib64}` 查找依赖。TOPSPTI 的公开 API
链接 `libtopspti.so`，由 SDK 转发到运行时；不是直接链接内部的 `libtopspti_rt.so`。

FlagTree 一侧也需要配套的构建注册、GCU launcher 和编译器改动；仅更新 FlagPrism 不足以完成联合集成。
SDK 二进制是外部构建依赖，不属于 FlagPrism 源码补丁。

统一验收直接运行本仓库的 Triton 算子，不需要 FlagGems。
若另外使用带自有编译缓存的算子库（例如 FlagGems 的 `LibEntry`），其缓存键应包含
FlagPrism instrumentation mode/config，避免复用带不同隐藏参数 ABI 的 kernel。
这类库的缓存与算子调度修复应由对应仓库独立提供。

## Debugger

```python
import flagtree.debugger as debugger

debugger.activate(auto_collect=True, level=1, addr_level=0,
                  record_capacity=65536, output_dir="debug-results")
# 在这里执行 Triton / FlagGems 算子。
debugger.deactivate()
```

`auto_collect=True` 在 TTIR 中自动添加数值/地址采集区域，因此动态生成的
FlagGems pointwise kernels 也能被覆盖，无需复制或修改 FlagGems 源文件。
默认值仍为 False，保留手工 collect markers 的用法。

GCU 调试编译会展开 block pointers；launcher 追加设备调试缓冲区指针，
运行后同步 GCU、回传和解码记录。设备内存与 pinned host memory 由 TOPS
Runtime 分配/释放，默认流和显式流都走对应的 TOPS API。

调试模式沿用 GCU 编译器的默认优化配置。数值摘要覆盖 load 和计算结果；没有其他可用数值摘要的 kernel 会采集 store 输出，
因此 `eye` 和 `zeros` 等纯输出 kernel 也能导出动态记录，同时避免重复归约。
GCU 的大 tile 采用单独上限，常量填充的摘要归约可在编译期折叠。
布尔值转浮点的摘要用一次真值计数精确推导全部指标，避免重复浮点归约导致寄存器分配失败。
调试耗时不能当作原始 kernel 的性能。其他 GCU 架构尚未真机验证。
GCU300 的地址采集仍有 64 位索引编译限制，统一轻量验收默认关闭地址采集。

GCU300 SDK 在部分标量摘要上会产生缺失的 `fabs(float)` libcall。
仅当确认出现该链接错误时，编译器才链接基于 IEEE 符号位清除的兼容函数，
并在编译元数据中标记 `debug_math_compat=scalar_fabs`。其他编译错误正常报错。
这条路径保持默认优化及完整 L2 范数摘要，不通过省略指标来规避错误。
常规算子数值、动态记录以及正/负/零标量的 L2 范数值已有真机回归。
测试入口检测到设备 context/Sip 异常后会阻止后续任务派发；并发中已经启动的任务仍需结束。
完整覆盖结论以对应运行的 summary.json 为准。

## L2 完整张量采集

L2 插桩新增整数和地址的 64 位存储，即使原算子仅使用 float32，也需要开启
GCU `enable_i64`。后端根据调试配置及插桩后的实际 payload 计划启用此选项；
全局 L1 内的局部 L2 区域同样适用。没有完整 payload 的 L1 和普通执行保持原选项。

当前 GCU300 SDK 的布尔向量直接转 int64 路径存在缺失设备符号，完整插桩叠加设备摘要归约
还会触发寄存器分配失败。Enflame 的 L2 lowering 使用 i32 低字/符号高字打包窄整数、连续
写入 64 位 payload，并使用协议允许的 i32 逐元素偏移计算。导出的 int64 ABI 保持一致。

L2 的计数、均值、最值和 L2 范数摘要从**实际回传的完整设备张量**在主机计算，避免设备重复归约；
没有自身完整 payload 的局部 L1 操作仍保留设备侧摘要。完整 L2 报告以 `summary_source=host_from_device_full_dump` 明确标注来源。
混合级别报告标为 `mixed_device_and_host_from_device_full_dump`。
完整张量仍来自插桩执行及 TOPS 回传，不能用 CPU 参考数据代替。摘要按 float32 计算，
不同归约顺序可能产生浮点舍入差异；采集耗时不是性能基准。

## Profiler

```python
import flagtree.profiler as profiler

session = profiler.start("profile", backend="enflame", hook="triton",
                         mode="runtime_base:runtime_host_timing_fallback=false")
# 在这里执行 Triton / FlagGems 算子。
profiler.finalize(session)
```

也支持 backend 名称 `gcu`/`tops`；不指定时根据 Triton target 自动选择。
TOPSPTI runtime/driver callbacks 建立 correlation ID 与 Triton scope 的映射；
kernel activities 提供设备 start/end（ns）、device ID、stream ID 和 kernel 名称。
活动按 API 启动时的会话归属过滤，暂停期间的 kernel 不计入报告。
同一 vendor 暂不支持重叠会话；拒绝创建不会残留无效会话路径。
flush 会同步当前 GCU 并检查 dropped records；无效时间戳或丢失记录报错。

输出使用公共 Hatchet、timeline、meta、vendor 格式。基础 kernel 时间来自设备，
没有 host timing fallback。此实现没有声明支持硬件性能计数器：必需但未支持的
指标报错，可选指标记录 unsupported 原因。

## 验收

```bash
cd /path/to/FlagPrism
python3 test.py --jobs 8 --devices 0,1,2,3,4,5,6,7
```

同一份算子与输入依次运行 debugger_l1/debugger_l2/profiler；采集失败后补跑普通执行。
普通执行也失败时 WARNING 放行；普通执行成功则 ERROR。最低覆盖数和完整失败语义见
[统一测试说明](TESTING.md)。WARNING 不代表采集成功。

验收使用清单中的轻量输入；不代表所有 shape、dtype 或完整模型均已验证。

摘要 JSON 中有限值保持数字；非有限值使用字符串 `"NaN"`、`"Infinity"`、`"-Infinity"`，并保留类型与 display 字段，避免生成非法 JSON。

历史 FlagGems 清单的测试记录仅代表当时版本；当前自带算子清单与结果见运行生成的
`manifest.json` 和 `summary.json`，不能直接沿用旧清单的通过率。首次 L2 编译默认允许 600 秒。
