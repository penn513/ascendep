# Ascend 950DT · EP8 · E24
宿主机工作目录为 /home/g00974569，项目目录为 /home/g00974569/ascendep。先同步 ep、lb_planner 和 scripts 到项目，再执行：
```bash
bash scripts/ep8-container.sh
docker exec -it ascendep-ep8-g00974569 bash
source ascendep/scripts/npu-env.sh
cd ascendep
torchrun --standalone --nproc-per-node=8 scripts/ep8-probe.py
python -m ep.suite --backend alltoall_hccl
python -m ep.suite --suite ep8_balanced
```
每卡 4096 tokens、H7168、FFN3072、Top6、3 个本地专家。权重为 BF16，每个专家接收 8192 条，每对 rank 3072 条；激活采用先限幅再 SwiGLU。小场景用独立 FP32 CPU oracle 验证输出、输入梯度、路由权重梯度、两个专家参数（含 gate/up/down）和一步 SGD。目标形状与 HCCL 逐张量比较相对 L2，阈值 0.035。默认预热 20 次、采样 50 次；完整前反向独立测量。报告保存每个 rank 的原始时间和张量分配器峰值，reserved 可能包含校验阶段留下的缓存。upstream 模式在各目录独立运行；其原生计时不能直接与统一 benchmark 混排。

## Ascend 950DT 双卡复测

在 177.51.27.1 上，独立容器 `ascendep-ep2-g00974569` 挂载物理设备 6、7。容器内从 0 开始编号。进入容器后运行以下命令，suite 会为本轮结果创建新目录。

```bash
source /opt/ae-cann/set_env.sh
cd /home/g00974569/ascendep
export ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m ep.suite --suite ep2_balanced
```

两卡均衡场景有 24 个专家，每卡持有 12 个。每卡输入 4096 个 token，Top-6，每个专家在两卡合计接收 2048 条路由。完整维度为 H7168、FFN3072。两种 DeepEP 使用独立 Python 环境，因为两个发行包都导出 `deep_ep` 模块。Lueelu 使用项目下的 `deps/luenv` 和配套 SHMEM SDK。

本轮运行使用 CANN 9.2.1、torch 2.10.0+cpu、torch_npu 2.10.0.post4。FSDPTurbo 的六个已测调度入口使用 `474a39a4f367fd740723238ebd3e4695f8c3f26c`，针对本版 CANN 和 torch_npu 的 group_list 整数类型与流 API 适配保存在 [补丁](../reports/ep2-67/fsdpturbo-cann921.patch)。独立 MegaMoE 入口来自 `41d479374004696e21f69f9c72094b1d29941987`。完整统计和各入口的运行状态见[双卡报告](../reports/ep2-67/benchmark.md)。
