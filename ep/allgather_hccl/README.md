# AllGather / ReduceScatter (HCCL)
在 PyTorch + torch_npu 环境加载 CANN。激活、逻辑专家 ID 和路由权重按 EP 组 AllGather。每个 rank 只计算归属于自己的专家，并按源 token 行累加其贡献，然后用 ReduceScatter(SUM) 回到源 rank。反向按互为伴随的 collective 返回激活和路由权重梯度。

两卡测试使用 ASCEND_RT_VISIBLE_DEVICES=6,7，在项目根目录设置 PYTHONPATH 后执行
`torchrun --standalone --nproc-per-node=2 ep/allgather_hccl/demo.py --output results/ag-ep2`。
