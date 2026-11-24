# ep_gmm_fused_lueelu

使用 lueelu/DeepEP PR #10 的 `deep_ep_experimental.ep_gmm_fused`。前向将 token 按全局专家排序，调用 UDMA AlltoAllv/GMM、限幅 SwiGLU 和 GMM/AlltoAllv，再按源 token 的路由权重合并。反向调用相反方向的融合算子传输梯度，并用 grouped matmul 计算专家权重梯度。计时包含布局转换、通信和专家计算。

在双卡容器中使用 CANN 9.2.1、匹配的 torch_npu、固定提交 `52e37b2ac7efae75a48ec45627dbf16af131731d`。先按上游脚本准备 CatCCOS、嵌套 CATLASS 和 SHMEM SDK，再在 Lueelu 的 Python 环境构建可选扩展。项目的 suite 可单独从 `deps/epgmm-site` 加载这个 wheel。

```bash
source /opt/ae-cann/set_env.sh
cd /home/g00974569/ascendep
source deps/luenv/bin/activate
export DEEPEP_NPU_ARCH=Ascend950 DEEPEP_BUILD_EP_GMM_FUSED=ON
export SHMEM_SOURCE_PATH="$PWD/deps/lueelu/third-party/shmem"
export SHMEM_ROOT="$SHMEM_SOURCE_PATH/install/shmem"
bash deps/lueelu/scripts/prepare_ep_gmm_fused_dependencies.sh
export CATCCOS_SOURCE_PATH="${CATCCOS_SOURCE_PATH:-$PWD/deps/lueelu/third-party/catccos}"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:$LD_LIBRARY_PATH"
python -m pip wheel --no-build-isolation --no-deps ./deps/lueelu -w deps/epgmm-wheel
python -m pip install --no-deps --target deps/epgmm-site deps/epgmm-wheel/*.whl
export PYTHONPATH="$PWD:${PYTHONPATH:-}" ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
python -m ep.suite --backend ep_gmm_fused_lueelu
```
