# MoonEP

此 demo 使用 MoonEP 的 Ascend UDMA dispatch 和 combine。专家归属固定，前向收集路由 ID，并在 CPU 上计算目的行和同 token 重复目的卡的索引。这部分开销计入前向。专家计算、路由权重以及反向通信均参与计时。输入采用项目的均衡 trace，各专家行数相同。

在验证容器的项目根目录构建 PR #40 和它引用的 SHMEM、CATLASS 子模块。

```bash
source /opt/ae-cann/set_env.sh
git clone https://github.com/MoonshotAI/MoonEP deps/moon
git -C deps/moon fetch origin refs/pull/40/head
git -C deps/moon checkout b4127baa1bec0bcccd49b48c2dc2a3be70da9a0f
git -C deps/moon submodule update --init --recursive
python -m venv --system-site-packages deps/moonenv
source deps/moonenv/bin/activate
(cd deps/moon && bash npu_patch/scripts/build_a5.sh /opt/ae-cann/set_env.sh)
export ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m ep.suite --backend moonep
```

物理卡 6、7 映射为本容器内的逻辑设备 0、1。suite 会使用对应的 Python 环境、源码目录和 SHMEM 动态库。
