# MindSpeed AllGather 通算重叠

使用 MindSpeed 的 MoELayerOverlapAllGather、原生 AllGather dispatcher 和 GroupedMLP overlap 前反向。固定路由直接提供每个 token 的专家 ID 和权重。专家使用限幅 SwiGLU，权重布局转换和路由展开均计入前向。

在验证容器的项目根目录准备源代码。Megatron 版本与本仓的 allgather_megatron 共用，MindSpeed 通过 megatron_adaptor 初始化 Ascend 适配。

```bash
source /opt/ae-cann/set_env.sh
git clone https://github.com/Ascend/MindSpeed deps/minds
git -C deps/minds checkout 01b0c9396774bfdf185a140af5ade4a98be9fc02
export ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m ep.suite --backend allgather_overlap_mindspeed
```

双卡验证的原始记录见 reports/ep2-67/retry-2-allgather_overlap_mindspeed。suite 将 deps/minds 和 deps/mcore 加入该后端的模块搜索路径。
