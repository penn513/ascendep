# MindSpeed hierarchical AllToAllV

单机 EP8 / TP1 不满足上游的多机分层拓扑要求，主表记作 unsupported。
在多机环境安装 core_r0.8.0 对应版本的 Megatron 与 MindSpeed，使用现有训练驱动和数据配置，经 torchrun 启动本目录 demo.py，并传入驱动路径及训练参数。demo 添加分层通信、TP extend EP 和 AllToAll overlap 三个开关。

[上游安装版本、开关及拓扑约束](https://gitee.com/ascend/MindSpeed/blob/c68e201a5733065b3aa59a3d44dc5e2ec700d7c9/docs/features/hierarchical-alltoallv.md)
