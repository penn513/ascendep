# megamoe_lueelu

统一 EP8 routed MoE 前反向，须通过数值验收。

在该后端隔离的 Python 环境中构建固定 PR。MegaMoE 和 ep_gmm_fused 需要构建对应扩展。

```bash
git clone https://github.com/lueelu/DeepEP deps/lueelu
export LUEELU_SRC="$PWD/deps/lueelu"
git -C "$LUEELU_SRC" fetch origin refs/pull/10/head
git -C "$LUEELU_SRC" checkout 52e37b2ac7efae75a48ec45627dbf16af131731d
git -C "$LUEELU_SRC" submodule update --init --recursive
(cd "$LUEELU_SRC" && python -m pip install -r requirements-build.txt && DEEPEP_BUILD_MEGAMOE=ON DEEPEP_BUILD_EP_GMM_FUSED=ON python -m pip install --no-build-isolation --no-deps .)
```
SHMEM、CATLASS 和 CANN 的构建要求见该提交 experiments/megamoe/README.md。

从容器内项目根目录执行。

```bash
torchrun --standalone --nproc-per-node=8 ep/megamoe_lueelu/demo.py --output results/manual
```

训练适配器接收本卡 hidden [4096,7168]、逻辑 ID 和路由权重 [4096,6]；返回同序输出，参数按连续逻辑专家 ID 分片，每卡持有 3 个。通过条件为 8 rank 完成前反向校验且生成同名 JSON。
