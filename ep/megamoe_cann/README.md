# megamoe_cann

统一 EP8 routed MoE 前反向，须通过数值验收。

使用 CANN ops-transformer 1c35600887a97f0e90fe1acda428db2e41028aad 和匹配的设备算子包。

```bash
git clone https://gitcode.com/cann/ops-transformer deps/ops-transformer
export CANN_SRC="$PWD/deps/ops-transformer"
git -C "$CANN_SRC" checkout 1c35600887a97f0e90fe1acda428db2e41028aad
```
MegaMoE 的 Python 入口另外依赖该仓 torch_extension 下的 cann_ops_transformer。

从容器内项目根目录执行。

```bash
torchrun --standalone --nproc-per-node=8 ep/megamoe_cann/demo.py --output results/manual
```

训练适配器接收本卡 hidden [4096,7168]、逻辑 ID 和路由权重 [4096,6]；返回同序输出，参数按连续逻辑专家 ID 分片，每卡持有 3 个。通过条件为 8 rank 完成前反向校验且生成同名 JSON。
