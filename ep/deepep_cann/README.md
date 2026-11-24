# CANN DeepEP

使用验证容器内 CANN 9.2.1 的 cann_ops_transformer.ops.mc2.common.elastic_buffer。该包随 CANN 环境提供 Python 和 C++ 扩展，首次加载时通过 Ninja 编译绑定，依赖匹配的 torch_npu 和 CANN 开发工具。

```bash
source /opt/ae-cann/set_env.sh
cd /home/g00974569/ascendep
export ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m ep.suite --backend deepep_cann
```

前向用 ElasticBuffer.dispatch 取得按专家排列的输入与路由权重，专家计算后调用 combine。反向复用原始 handle，分别用 dispatch 分发输出梯度、用 combine 汇总输入和路由权重梯度。底层调用 MoE EP Dispatch、Combine 以及各自的 epilogue。
