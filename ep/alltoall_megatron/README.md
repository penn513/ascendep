# alltoall_megatron

统一 EP8 routed MoE 前反向，须通过数值验收。

使用服务器源码 /home/g00974569/code/megatron-lm，版本 a845aa7e12b3a117e24c2352b9e3e60bad2e3a17。将该 checkout 复制到项目 deps/megatron，容器中执行 export PYTHONPATH="$PWD/deps/megatron:$PYTHONPATH"。依赖 torch_npu.contrib.transfer_to_npu。

从容器内项目根目录执行。

```bash
torchrun --standalone --nproc-per-node=8 ep/alltoall_megatron/demo.py --output results/manual
```

训练适配器接收本卡 hidden [4096,7168]、逻辑 ID 和路由权重 [4096,6]；返回同序输出，参数按连续逻辑专家 ID 分片，每卡持有 3 个。通过条件为 8 rank 完成前反向校验且生成同名 JSON。
