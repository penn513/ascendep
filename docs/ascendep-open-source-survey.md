# ascendep 开源调研：昇腾 MoE 专家并行与 benchmark

调研日期为 2026-11-23。以训练为主，同时收录可用于通信微基准、融合前向和推理侧验证的实现。本文依据公开仓库、脚本、PR 和维护者报告，未在 NPU 上独立复测。

## 结论

已有昇腾 EP 通信库、训练框架优化、专家负载均衡和 MoE 层级 benchmark。最值得复用的三份测试是 DeepEP-Ascend 的 950DT 通信测试、SGLang 的 A3 高吞吐/低时延测试，以及 MoonEP 用同一组路由对比两种后端的脚本。训练大融合还有 Mega-MoE-TD 的 A3 前反向案例。在本次核对的公开项目中，尚未发现同时覆盖多后端、固定路由回放、训练前反向、超节点内/跨超拓扑的统一昇腾评测套件。

## 实现和测试入口

| 项目 | 公开实现与入口 | ascendep 可复用的部分与边界 |
|---|---|---|
| DeepSeek DeepEP-Ascend | [官方仓库](https://github.com/deepseek-ai/DeepEP-Ascend)、[tests/ep/test_ep.py](https://github.com/deepseek-ai/DeepEP-Ascend/blob/main/tests/ep/test_ep.py)。EPBuffer 提供 expanded dispatch、BF16 combine、cached handle 和延后 epilogue。| 950DT 通信首选复现点。保留原生计时，加测消费者可读取输出时的完整耗时。公开数据依赖特定 HDK。|
| SGLang sgl-kernel-npu | [维护者数据](https://github.com/sgl-project/sgl-kernel-npu/blob/main/README.md)、[tests/python/deepep/test_intranode.py](https://github.com/sgl-project/sgl-kernel-npu/blob/main/tests/python/deepep/test_intranode.py)。A3 384 SuperPOD 有 normal 与 low-latency 两套结果。| A3 的大 token 吞吐和小 token 时延基线。名称也使用 DeepEP-Ascend，记录来源仓、提交和调用接口。|
| CANN ops-transformer | [DispatchV2](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_distribute_dispatch_v2/op_api)、[CombineV2](https://gitcode.com/cann/ops-transformer/blob/master/mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV2.md)、[AlltoAllvGroupedMatMul](https://gitcode.com/cann/ops-transformer/tree/master/mc2/allto_allv_grouped_mat_mul)、[GroupedMatMulAlltoAllv](https://gitcode.com/cann/ops-transformer/tree/master/mc2/grouped_mat_mul_allto_allv)、[MegaMoE 目录](https://gitcode.com/cann/ops-transformer/tree/master/mc2/mega_moe)。| 分开评测 token 通信、A2Av 与单次 GMM 融合、整层融合前向。按各算子产品、dtype、shape 和版本限制登记。|
| MindSpeed、MindSpeed-LLM/FSDPTurbo | [分层 AlltoAllv](https://gitee.com/ascend/MindSpeed/blob/c68e201a5733065b3aa59a3d44dc5e2ec700d7c9/docs/features/hierarchical-alltoallv.md)、[Balanced MoE](https://github.com/Ascend/MindSpeed/blob/master/docs/zh/features/balanced_moe.md)、[FSDP2 调优说明](https://github.com/Ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/tuning/fsdp2_backend_performance_optimization.md)。 | 训练端到端对照。Balanced MoE 广播热专家、就近计算，依赖 alltoall dispatcher、grouped GEMM 和 FB overlap，与文档列出的分层 AlltoAllv 互斥。FSDP2 有 `--parallel.ep_dispatcher fused` 和 `mc2`，后者依赖 FSDPTurbo 算子及模型适配。|
| MoonEP | [昇腾适配 PR #40](https://github.com/MoonshotAI/MoonEP/pull/40)、[上游横评脚本](https://github.com/MoonshotAI/MoonEP/blob/master/benchmarks/bench_vs_deepep.py)、[910C 社区实现](https://github.com/jianzhnie/MoonEP-Ascend-910C/blob/ascend-910c-adaptation/README_910C.md)。| 昇腾 PR 公开了 SHMEM、MTE 和 950 UDMA 路径及测试入口。910C 社区路径采用主机侧 planner。上游横评在 H20 EP8 上扫 MaxVio，对比规划、权重预取、前反向通信，可移植用例设计。|
| Mega-MoE-TD | [代码与 A3 八卡 benchmark](https://gitcode.com/alan_wangtianqi/Mega-MoE-TD)。有 benchmark、tests、阶段打点和 backward/recompute 路径。| Triton-distributed-ascend 的训练大融合原型，能作为 Kimi 形状下单层前反向对照。README 区分八卡实测与更大 EP 的编译用例，后者仍需上机。|
| vLLM Ascend 与 SGLang | [vLLM Ascend 通信选择](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py)、[EPLB](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md)、[SGLang 昇腾支持矩阵](https://sgl-project.github.io/platforms/ascend/ascend_npu_support_features.html)。| 推理验证覆盖不同 dispatcher、冗余专家和运行时负载均衡。分开测 prefill 与 decode，并核对后端和 EPLB 的兼容组合。|
| HCCL Test | [官方性能测试说明](https://www.hiascend.com/document/detail/en/canncommercial/800/hcclug/hcclug_000025.html)。| 单独测 AllToAllV，判断相同消息规模下网络是否异常。端到端 EP 对照还需本地重排、目标计数和加权归并。|

### 相邻项目与研究线索

- [UltraEP](https://github.com/Dots-Infra/UltraEP)、[LPLB](https://github.com/deepseek-ai/LPLB) 可作为实时复制、专家放置及路由分配的算法对照。当前核对材料不足以将二者列为已验证的昇腾训练后端。
- [ROCm/MAD 的 large-ep-benchmark](https://github.com/ROCm/MAD/blob/develop/scripts/large-ep-benchmark/README.md) 提供多后端 EP 回归测试的工程参考，覆盖 MoRI 和 DeepEP。
- [HyperParallel-MoE](https://arxiv.org/abs/2605.23764) 是面向昇腾训练的多核交错调度研究。复现论文曲线仍需确认对应实现、版本及运行配置。

## 两组可复现的公开测点

下表摘录维护者报告，带宽单位 GB/s，D/C 分别为 dispatch/combine。保留原始输入与时间边界；这些数字用于各自环境的复现校验。

| 来源与配置 | EP8 D / C | EP128 D / C | 口径 |
|---|---|---|---|
| [DeepEP-Ascend 950DT](https://github.com/deepseek-ai/DeepEP-Ascend)，CANN 9.2.0，特定 PoC HDK，netlayer 1，容量 16,384 tokens/rank，H=7168，E=256，Top-6，FP8/BF16 | 373–375 / 345–347 | 313–320 / 272–278 | 各 rank 范围，包含 issue/drain，排除最终 epilogue。|
| [SGLang A3 384 SuperPOD](https://github.com/sgl-project/sgl-kernel-npu/blob/main/README.md)，纯 HCCS，normal，4096 tokens/batch（README 原口径），H=7168，Top-8，INT8/BF16 | 146 / 125 | 57 / 81 | 作者定义的瓶颈带宽。同仓另给 128 tokens/batch 的低时延结果，EP8 为 132 / 126 µs。|

## 建议首版 benchmark

1. **通信**：先用同一份 token、top-k ID、路由权重与 rank 映射驱动 HCCL 参考实现（permute + AllToAllV + unpermute/weighted reduce）和 950DT DeepEP。之后接入支持相同形状与精度的 CANN dispatcher。有的后端直接产出按专家排列的张量，比较终点应设为下游专家可读取输出。
2. **训练单层**：在相同权重和输入上比较普通 EP + GroupedMatMul、MC2、MoonEP 动态冗余专家；接入大融合时增加同精度同范围的 MoE 前反向对照。记录 planner、专家权重预取、梯度归还、前反向墙钟和峰值 HBM。融合流上有重叠时以实测关键路径为准。
3. **训练 step 与推理**：用 MindSpeed/FSDPTurbo 验证单层收益能否转化为 step time。若评推理，另用 SGLang/vLLM 记录 prefill 吞吐和 decode 时延。
4. **负载与拓扑**：先复现原脚本，再固定每 rank 4096 tokens，使用均匀、固定热点、逐步迁移热点和模型路由回放。先测 EP8/64/128，再把 EP256/512 列为扩展目标。能控制 rank 放置时，保持 EP 大小不变，分别在超节点内和跨超运行；记录实际 rank 表、网层、数据类型、token 去重与 padding。
5. **正确性与指标**：先校验路由计数、前向输出、输入梯度和专家权重梯度；量化路径报告误差。每次迭代取各 rank 最大耗时，再跨迭代计算 P50/P95。分别记录 expert MaxVio 和 rank 最大/平均负载比。带宽注明是逻辑载荷、去重后的跨域字节还是硬件计数器实测，以及是单 rank 还是全组聚合。发布硬件、软件版本、commit、同步边界和原始记录。

优先交付一个可回答具体工程问题的实测：在同一 950DT 拓扑和同一批路由上，比较基础 EP、融合通信与动态冗余专家，给出均匀及倾斜负载下的完整 MoE 前反向时间。这样可以判断交换量下降、最慢 rank 改善和额外参数搬运分别如何影响训练收益。
