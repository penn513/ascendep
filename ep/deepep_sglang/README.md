# SGLang DeepEP

在 A5 容器中构建固定提交的 BF16 节点内后端。此 wheel 与其他实现共用 Python 包名 deep_ep，suite 为它设置独立导入目录。

```bash
source /opt/ae-cann/set_env.sh
git clone https://github.com/sgl-project/sgl-kernel-npu deps/sgl
git -C deps/sgl checkout 597250fcc1b31f2607bf817d42bd0699ea60b8eb
python -m venv --system-site-packages deps/sglenv
source deps/sglenv/bin/activate
(cd deps/sgl && bash build.sh -a deepep Ascend950)
python -m pip install --no-deps --target deps/sgl-site deps/sgl/output/deep_ep*.whl
export ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m ep.suite --backend deepep_sglang
```

隐藏状态通过上游默认策略的 dispatch/combine 交换。该接口返回每行的源 rank、token、Top-K 槽位。demo 使用这些索引从可微 HCCL AllGather 的路由权重中取值，在专家侧加权，combine 使用单位权重。反向通过同一组节点内原语传递隐藏状态梯度，ReduceScatter 汇总路由权重梯度。以上操作均计入训练时延。

本轮为原生算子单独创建 HCCL 通信组，缓冲区设为 2048 MiB。参考计算、同步和梯度通信使用默认组。小规模正确性检查使用 H=1024，目标形状使用 H=7168。
