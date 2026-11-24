# megamoe_fsdpturbo

调用 FSDPTurbo 的 `get_mega_moe_forward_fn` 和原生 `MegaMoeBF16Train` 进行 BF16 路由专家层前反向。两卡均衡场景每卡输入 4096 个 token，24 个专家平均分配，每卡 12 个，每个 token 选择 6 个专家。

从容器内项目根目录开始。将包含 `fsdp_turbo/distributed/expert_parallel/dispatcher_mega_moe.py` 的 FSDPTurbo 源码放在 `deps/fsdp-mega`。本次使用的 dispatcher 文件 SHA256 为 `2dc62570d87d3b88680e3061bb26f60121942deaf4e565df186992c012c917ec`。在已安装 torch 和 torch_npu 的 Ascend 950 环境构建训练扩展：

```bash
source /opt/ae-cann/set_env.sh
git clone -b train https://gitcode.com/jzhoujg/ascend_deepep_hkq.git deps/fsdp-train
git -C deps/fsdp-train checkout b3b141916c068d2098aa437fc502438ab94060bd
git -C deps/fsdp-train submodule update --init --recursive
python -m venv --system-site-packages deps/fmenv
source deps/fmenv/bin/activate
(cd deps/fsdp-train && bash scripts/build.sh -soc_type Ascend950 && bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950)
```

容器仅映射物理卡 6、7 时，使用容器逻辑设备 0、1。从项目根目录启动，输出目录须为新目录：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m ep.suite --backend megamoe_fsdpturbo --output results/manual-fsdp-mega
```

suite 默认设置 `ASCEND_DEEPEP_TOKEN_CHUNK_M=4096`。demo 先验证小形状，释放其 SHMEM 窗口，再为目标形状创建窗口。两个 rank 均通过输出和梯度校验后才采样。
