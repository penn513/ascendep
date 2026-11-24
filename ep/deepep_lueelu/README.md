# deepep_lueelu

通过 ElasticBuffer 调用展开式 dispatch/combine，反向使用相反的通信操作。

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
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:28674
export ASCEND_DEEPEP_SHMEM_IP_PORT="$DEEPEP_SHMEM_ENDPOINT"
python -m ep.suite --backend deepep_lueelu
```
