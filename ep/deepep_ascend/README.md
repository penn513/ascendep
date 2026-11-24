# deepep_ascend

统一 EP8 routed MoE 前反向，须通过数值验收。

在独立 Python 环境安装官方 EPBuffer，核对驱动及 HDK。

```bash
git clone --recursive https://github.com/deepseek-ai/DeepEP-Ascend deps/deepseek
git -C deps/deepseek checkout 4dfccac4fc345519427e1019557301c9e59e9c76
git -C deps/deepseek submodule update --init --recursive
python -m pip install --no-build-isolation ./deps/deepseek
```

从容器内项目根目录执行。

```bash
torchrun --standalone --nproc-per-node=8 ep/deepep_ascend/demo.py --output results/manual
```

训练适配器接收本卡 hidden [4096,7168]、逻辑 ID 和路由权重 [4096,6]；返回同序输出，参数按连续逻辑专家 ID 分片，每卡持有 3 个。通过条件为 8 rank 完成前反向校验且生成同名 JSON。
