# ascendep

昇腾 MoE 专家并行（EP）评测与模型接入项目。在相同模型、逻辑路由和物理拓扑下，对比 EP 方案的训练耗时、吞吐与显存占用，并提供可移植到训练框架的接入示例。

**当前进度**　Ascend 950DT 物理卡 6、7 已完成多种 EP 后端的训练性能实测。场景为 BF16、每卡 4096 tokens、24 个专家、Top-6、EP2，使用均衡路由并关闭负载均衡 planner。各后端的前反向 P50/P95、吞吐、显存与待测原因见[两卡实测报告](reports/ep2-67/benchmark.md)。CPU 首版覆盖独立 SwiGLU 参考层、两进程 Gloo EP、固定副本两步训练、路由生成与回放。

| 本地文档 | 用途 |
|---|---|
| [v0.1 实施与验收计划](docs/ascendep-v0.1-plan.md) | 评审 M1–M4 范围、测试用例、正确性与资源度量口径 |
| [昇腾 EP 开源调研](docs/ascendep-open-source-survey.md) | 查找现有实现、公开性能测点、测试脚本和来源 |
| [项目讨论记录](docs/project-handoff.md) | 回顾需求、设计建议和待决策事项 |
| [项目指令](AGENTS.md) | 在本地 Codex 会话中沿用本项目上下文 |

## 安装与一条命令回归

在项目根目录运行（macOS/Linux，Python 3.9+）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m ascendep regress
```

本次验证环境为 macOS arm64、Python 3.9.6、PyTorch 2.8.0、NumPy 2.0.2、pytest 8.4.2。需要固定核心版本时，安装命令增加 `-c configs/cpu-tested-constraints.txt`。Python 3.9 的 torchrun 兼容依赖由包自动安装。

回归命令执行测试、参考层基准、两进程基准和固定副本基准，生成 `reports/regression/`：

- `results.json` / `results.md`：3 个 CPU 测点与 990 个设备计划配置，明确完成数及未运行原因。
- `training-validation.json`：Top-1/2 共 14 个场景的逐项最大绝对误差，均连续运行两步；包含同一 trace 的固定副本关闭/开启对照。
- `large-route-audit.json`：E2048、Top-16、4096 tokens/rank，六档 EP 的均衡路由流式检查；只代表逻辑配置和计数验证。
- `reference/`、`gloo/`、`replica/`：各自的原始逐迭代/逐 rank 时间、资源信息和完整 `trace.pt`。

本地 launcher 使用真实 torchrun。macOS 默认显式设置回环网络接口与 `USE_LIBUV=0`，规避本机探针中默认 rendezvous 无法启动的问题；不更改系统网络。任务超时会终止该次 launcher 和子进程。

## 单独运行、回放与接入

```bash
# 公共参考层；默认预热 20 次、采样 50 次
python -m ascendep benchmark --backend cpu/reference

# 两进程，固定副本可选
python -m ascendep launch --module ascendep benchmark --backend cpu/gloo
python -m ascendep launch --module ascendep benchmark --backend cpu/gloo --fixed-replica --output reports/replica

# 保存完整 trace，再在不同放置下消费同一份文件
python -m ascendep trace --world 2 --output reports/shared.pt
python -m ascendep launch --module ascendep benchmark --backend cpu/gloo --trace reports/shared.pt --output reports/replay-off
python -m ascendep launch --module ascendep benchmark --backend cpu/gloo --trace reports/shared.pt --fixed-replica --output reports/replay-on

# 模型层接入，包含 latent 投影和两步 SGD
python -m ascendep launch examples/train.py

# 只枚举设备计划，或只做大规模逻辑路由检查
python -m ascendep matrix
python -m ascendep capabilities
python -m ascendep route-audit
```

通过 `--config` 使用与 `configs/cpu-small.json` 相同结构的模型配置；`--target` 指定 expert MaxVio。小 E/K 下的 10、20 档不可达；随机小样本可能无法达到低倾斜目标。trace 仍保存实际值、误差和 `unreachable` / `not_matched` 状态，benchmark 会拒绝未达到目标的随机场景。均衡非整除时允许计数相差 1，标记离散限制。默认四档倾斜的统计测试使用 E256、Top-8、8192 tokens。生成器使用分块抽样，避免分配全局 tokens × E 工作区；完整 trace 仍需保存 O(tokens × hidden + tokens × K) 张量。

`route-audit --targets 0 0.2 1 10 20` 可流式搜索各档倾斜，不保存隐藏状态或完整路由。大 E 和大 token 数的随机搜索计算量仍然很大；默认回归仅执行大规模均衡路由，四档随机倾斜在较小路由测试上验证。流式生成器单独标版本和 chunk size，不承诺与其他版本随机序列一致。

模型入口为 `layer(local_hidden, logical_expert_ids, topk_weights)`；输出恢复本 rank 输入顺序。路由权重可求梯度，默认使用 FP64。单进程 `ReferenceMoE` 直接遍历逻辑专家；`ExpertParallelMoE` 仅保存本 rank 的 owner/副本专家，使用可变 split All-to-All 交换 token 和反向梯度。连续放置要求 E 能被 EP 整除，首版实测范围为 EP2，输入允许不等长甚至为空。所有 rank 必须按同一顺序参与前反向和副本维护；调用方提供一致的配置、trace 及进程组。

专家参数不做 DDP 平均。每 rank 的局部损失按同一全局元素数归一化；副本训练按 `zero_grad → forward → backward → return_replica_gradients → optimizer.step → refresh_replicas` 调用，optimizer 仅持有 `owned_parameters()`。从 checkpoint 加载 owner 后也应刷新一次副本。示例中 dense 投影为复制参数，单独 all-reduce 梯度。当前数值验收使用无动量、无 weight decay 的 SGD；高阶梯度与其他优化器语义未验收。

## 报告口径与未完成范围

每个指标先取该迭代最慢 rank，再分别计算 P50/P95。`fwd_bwd_ms` 独立测量，不由前向/反向分位数相加。`bwd_ms` 含反向通信和副本梯度归还；`train_step_ms` 另含损失、SGD 和副本刷新。trace 构造、校验训练、初始权重初始化在计时外；运行时 dispatcher 与放置选择计入前向。CPU 路径的输入合法性检查也计入耗时，CPU 延迟仅用于本地回归。

内存报告包含各 rank 的精确参数字节数、冗余权重字节数、源输入字节数与进程峰值 RSS；激活/通信缓冲峰值尚未插桩，HBM 使用空值及原因。吞吐按全局源 tokens 计算，不乘 Top-k。固定副本案例明确命名 `fixed_replica_fixture`，不是 MoonEP 或 UltraEP，CPU 时间不用于宣称加速收益。

设备主矩阵覆盖 11 个已列出的后端/变体占位 × 6 EP × 5 路由 × 3 策略 = 990 个配置，当前全部 `not_run`。Megatron、MindSpeed、FSDPTurbo、MC2、MoonEP 的独立变体尚待固定源码版本后盘点，占位项不代表完整原生变体覆盖。无原生实现会回退到 CPU。尚待完成原生策略兼容性、commit/PR SHA 固定、NPU/HCCL、多流事件、完整阶段资源采集、设备曲线和完整训练框架接入，详见实施计划。

## 950DT 单机 EP8

NPU 验证使用独立容器和固定均衡路由。见 [启动与测试说明](ep/README.md) 及 [分后端清单](ep/backends.json)。现有 CPU 回归与设备计划矩阵保留各自的报告口径。

### EP8 profiling

本轮 22 个后端的性能排名、profiling 观察和逐后端压缩包见[八卡实测报告](reports/ep8-profile/benchmark.md)。

在仅映射物理卡 0–7 的验证容器内，从项目根目录运行：

```bash
source /opt/ae-cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m ep.suite --suite ep8_balanced --profile --output results/ep8_run
```

每个通过数值校验的后端先完成常规 20 轮预热和 50 轮计时，然后在相同进程和目标输入上开启 Level1 CPU/NPU profiler。profiler 预热 1 步，记录 3 个连续的完整前反向。每个 rank 的原始数据和解析后的时间线保存在 `profiling/<backend>/rankN/`，阶段标记为 `ep_forward` 和 `ep_backward`。

归档单个已结束的后端，例如：

```bash
python -m ep.archive_profile results/ep8_run alltoall_hccl results/ep8_run/archives
```

压缩包包含原始 profiler 数据、解析结果、benchmark JSON 和运行日志。旁边的 JSON 索引记录压缩包 SHA-256、各 rank 的输入哈希和时间线路径。

EP8 全量运行归档后，以下命令校验压缩包、逐步核对八个 rank 的设备事件并生成报告。若补测，单后端的 suite 输出放在 `results/ep8_run/retry-<backend>`，归档到同一个 archives 目录。

```bash
python -m ep.report_profile results/ep8_run/archives --run results/ep8_run
```
