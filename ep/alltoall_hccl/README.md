# alltoall_hccl

统一 EP8 routed MoE 前反向，须通过数值验收。

使用容器镜像内 PyTorch 2.10.0 与 torch_npu 2.10.0.post6，加载 scripts/npu-env.sh 指定的 CANN 9.2.0。

从容器内项目根目录执行。

```bash
torchrun --standalone --nproc-per-node=8 ep/alltoall_hccl/demo.py --output results/manual
```

训练适配器接收本卡 hidden [4096,7168]、逻辑 ID 和路由权重 [4096,6]；返回同序输出，参数按连续逻辑专家 ID 分片，每卡持有 3 个。通过条件为 8 rank 完成前反向校验且生成同名 JSON。
