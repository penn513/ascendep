# Ascend 950DT · EP8 benchmark 与 profiling

22 个后端完成性能采样并归档八个 rank 的 profiling。共 24 个候选。

物理卡 0–7，BF16，均衡路由，planner 关闭。H7168、FFN3072、Top-6、24 个专家，每卡 3 个专家、4096 tokens。吞吐按全组 32768 个源 tokens 计算。

预热 20 轮、采样 50 轮。每轮取八个 rank 的最大耗时，再计算 P50/P95。性能采样结束后开启 Level1 CPU/NPU profiling，预热 1 步，记录 3 步。

| 后端 | 前向 P50/P95 ms | 反向 P50/P95 ms | 前反向 P50/P95 ms | tokens/s | Profiling 归档 |
|---|---:|---:|---:|---:|---|
| megamoe_lueelu | 8.79 / 8.96 | 17.35 / 17.96 | 25.97 / 26.49 | 1261613 | [megamoe_lueelu.tar.gz](megamoe_lueelu.tar.gz) |
| mc2_fsdpturbo | 10.78 / 10.88 | 19.28 / 19.40 | 29.90 / 30.18 | 1095756 | [mc2_fsdpturbo.tar.gz](mc2_fsdpturbo.tar.gz) |
| deepep_cann | 10.42 / 10.48 | 21.61 / 22.73 | 31.75 / 31.84 | 1032063 | [deepep_cann.tar.gz](deepep_cann.tar.gz) |
| alltoallv_gmm_cann | 11.23 / 11.31 | 20.94 / 21.01 | 31.93 / 32.02 | 1026179 | [alltoallv_gmm_cann.tar.gz](alltoallv_gmm_cann.tar.gz) |
| fused_fsdpturbo | 11.88 / 12.02 | 20.73 / 20.77 | 32.41 / 32.55 | 1011194 | [fused_fsdpturbo.tar.gz](fused_fsdpturbo.tar.gz) |
| deepep_ascend | 10.47 / 10.52 | 22.52 / 22.56 | 32.74 / 32.82 | 1000781 | [deepep_ascend.tar.gz](deepep_ascend.tar.gz) |
| ep_gmm_fused_lueelu | 12.14 / 12.31 | 21.01 / 21.07 | 32.93 / 32.99 | 995051 | [ep_gmm_fused_lueelu.tar.gz](ep_gmm_fused_lueelu.tar.gz) |
| allgather_overlap_mindspeed | 12.91 / 13.08 | 20.69 / 20.85 | 33.27 / 33.65 | 984906 | [allgather_overlap_mindspeed.tar.gz](allgather_overlap_mindspeed.tar.gz) |
| megamoe_fsdpturbo | 9.91 / 10.02 | 23.70 / 24.10 | 33.56 / 34.13 | 976464 | [megamoe_fsdpturbo.tar.gz](megamoe_fsdpturbo.tar.gz) |
| domino_fsdpturbo | 11.28 / 11.81 | 23.37 / 23.46 | 34.34 / 34.61 | 954281 | [domino_fsdpturbo.tar.gz](domino_fsdpturbo.tar.gz) |
| chunkmoe_fsdpturbo | 11.33 / 11.40 | 23.26 / 23.32 | 34.43 / 34.53 | 951759 | [chunkmoe_fsdpturbo.tar.gz](chunkmoe_fsdpturbo.tar.gz) |
| deepep_sglang | 12.24 / 12.29 | 22.60 / 22.66 | 34.53 / 34.58 | 949097 | [deepep_sglang.tar.gz](deepep_sglang.tar.gz) |
| allgather_megatron | 12.56 / 13.13 | 22.65 / 22.76 | 34.81 / 35.26 | 941409 | [allgather_megatron.tar.gz](allgather_megatron.tar.gz) |
| alltoall_megatron | 13.04 / 13.16 | 22.36 / 22.47 | 35.15 / 35.42 | 932107 | [alltoall_megatron.tar.gz](alltoall_megatron.tar.gz) |
| allgather_hccl | 12.50 / 12.63 | 23.71 / 23.77 | 35.77 / 36.05 | 916032 | [allgather_hccl.tar.gz](allgather_hccl.tar.gz) |
| alltoall_hccl | 12.79 / 12.91 | 24.41 / 24.51 | 37.05 / 37.66 | 884422 | [alltoall_hccl.tar.gz](alltoall_hccl.tar.gz) |
| alltoall_fsdpturbo | 13.82 / 13.98 | 23.58 / 23.69 | 37.27 / 37.45 | 879249 | [alltoall_fsdpturbo.tar.gz](alltoall_fsdpturbo.tar.gz) |
| allgather_fsdpturbo | 16.44 / 16.95 | 24.59 / 24.67 | 40.59 / 40.74 | 807297 | [allgather_fsdpturbo.tar.gz](allgather_fsdpturbo.tar.gz) |
| mc2_dispatch_cann | 16.61 / 17.48 | 25.01 / 25.46 | 40.89 / 41.84 | 801358 | [mc2_dispatch_cann.tar.gz](mc2_dispatch_cann.tar.gz) |
| megamoe_td | 14.99 / 15.34 | 30.44 / 30.80 | 45.14 / 46.11 | 725913 | [megamoe_td.tar.gz](megamoe_td.tar.gz) |
| moonep | 32.92 / 33.78 | 23.82 / 23.88 | 56.42 / 57.10 | 580825 | [moonep.tar.gz](moonep.tar.gz) |
| deepep_lueelu | 31.39 / 31.74 | 40.12 / 41.29 | 71.67 / 71.95 | 457197 | [deepep_lueelu.tar.gz](deepep_lueelu.tar.gz) |

每个压缩包包含 benchmark JSON、运行日志、各 rank 的原始 profiling 及解析结果。解压后在 MindStudio Insight 中打开对应 rank 的 trace_view.json，搜索 ep_forward、ep_backward 可定位主机阶段，并沿关联事件查看 NPU 执行。

## Profiling 观察

主要算子按 profiler 的累计设备执行时间排序。通信与重叠取 step_trace_time 中各 rank 各步的中位数。设备空闲列为各 rank 三步空闲时长中位数的范围。SHMEM 等融合算子内部通信可能计入算子耗时；可见通信为零时仍可能存在内部传输。

| 后端 | 累计耗时前三算子类型 | 可见未重叠通信 ms | 可见重叠 ms | 设备空闲范围 ms |
|---|---|---:|---:|---:|
| alltoall_hccl | MatMulV3, Mul, IndexPutWithSortV2 | 5.14 | 0.00 | 1.00–1.32 |
| allgather_hccl | MatMulV3, HcclLaunchAicpuKernel, Add | 6.80 | 0.00 | 0.82–1.00 |
| allgather_fsdpturbo | GroupedMatmul, HcclLaunchAicpuKernel, MoeFinalizeRoutingV2Grad | 5.16 | 0.00 | 0.89–1.15 |
| allgather_megatron | MatMulV3, HcclLaunchAicpuKernel, Mul | 7.75 | 0.00 | 1.38–1.50 |
| alltoall_fsdpturbo | MatMulV3, TensorMove, Mul | 5.30 | 0.21 | 0.82–1.32 |
| alltoall_megatron | MatMulV3, HcclLaunchAicpuKernel, ConcatD | 6.50 | 0.00 | 1.44–2.02 |
| chunkmoe_fsdpturbo | GroupedMatmul, Add, HcclLaunchAicpuKernel | 2.99 | 2.91 | 0.91–1.26 |
| deepep_ascend | MatMulV3, Mul, Add | 0.81 | 0.00 | 0.67–0.77 |
| domino_fsdpturbo | GroupedMatmul, Add, HcclLaunchAicpuKernel | 2.58 | 3.37 | 1.17–1.51 |
| fused_fsdpturbo | GroupedMatmul, HcclLaunchAicpuKernel, MoeFinalizeRoutingV2Grad | 5.18 | 0.00 | 0.77–0.85 |
| mc2_fsdpturbo | AlltoAllvGroupedMatMulMc2AicpuKernel, AlltoAllvGroupedMatMul, GroupedMatMulAlltoAllvMc2AicpuKernel | 0.57 | 0.00 | 1.59–2.18 |
| megamoe_fsdpturbo | _Z22mega_moe_bf16_backwardPhS_S_S_S_S_S_S_S_S_S_S_17MegaMoeTilingData, _Z13mega_moe_bf16PhS_S_S_S_S_S_S_S_S_S_17MegaMoeTilingData, ZerosLike | 0.90 | 0.00 | 1.19–1.45 |
| megamoe_lueelu | _Z22mega_moe_bf16_backwardIu6__bf16EvPhS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_21MegaMoeBf16TilingData, _Z21mega_moe_bf16_forwardIu6__bf16EvPhS0_S0_S0_S0_S0_S0_S0_S0_S0_S0_S0_21MegaMoeBf16TilingData, HcclLaunchAicpuKernel | 0.98 | 0.00 | 0.72–0.91 |
| deepep_lueelu | MatMulV3, dispatch_kernel, deepep_instance_sync_kernel | 0.57 | 0.00 | 28.58–37.81 |
| ep_gmm_fused_lueelu | _ZN7Catccos5DGemm6Device13KernelAdapterINS0_6Kernel31Ascend950AllToAllVGroupedMatmulINS0_30AllToAllVAllGatherProblemShapeEN7Catlass4Gemm5Block12BlockMmadTlaINS7_12MmadPingpongINS6_4Arch9Ascend950ELb1ELb0ELj1ELb0ELj2ELj2ELj2ELj2EEEN3tla5tupleIJNSE_1CILi128EEENSG_ILi256EEESI_EEENSF_IJSH_SI_NSG_ILi64EEEEEEu6__bf16u6__bf16u6__bf16vNS7_4Tile17PackedTileCopyTlaISC_u6__bf16NS6_6layout8RowMajorEu6__bf16SP_u6__bf16SP_vLb0ELNSM_16ScaleGranularityE0ENSM_8CopyToGMEEENSM_11TileMmadTlaISC_u6__bf16NSO_2zNEEEEENS_4Comm5Block16CommBlockMteUdmaINSY_9CommBlockINSX_19AtlasCommRemoteCopyISC_Lj2ELb1EEEJNS7_8GemmTypeIu6__bf16SP_LN7AscendC9TPositionE0EEES16_vNSX_4Tile14TileRemoteCopyISC_Lb1ES16_S16_vLNS_6detail10CopyDirectE0ELNS19_13CopyTransportE0EEENS6_8Epilogue4Tile27EpilogueIdentityTileSwizzleEEEENS10_INSX_23AtlasCommUdmaRemoteCopyISC_Lj2EEEJS16_S16_NS17_20TileRemoteCopyUdma2DISC_S16_S16_LS1A_0EEEEEEEENS0_5Block36BlockMmadSchedulerAllToAllVAllGatherINS0_14MoeConstraintsILj1ELj8ELj64EEELj1EEENSY_36BlockCommSchedulerAllToAllVAllGatherIS1Q_EELj2ELb1ELb1ENS10_INSX_18AtlasCommLocalCopyISC_Lj2ELb1EEEJS16_S16_vS1C_S1F_EEEEEEEvNT_6ParamsEPh, _ZN7Catccos5DGemm6Device13KernelAdapterINS0_6Kernel25GroupedMatmulAllToAllVTlaINS0_30AllToAllVAllGatherProblemShapeEN7Catlass4Gemm5Block12BlockMmadTlaINS7_12MmadPingpongINS6_4Arch9Ascend950ELb1ELb0ELj1ELb0ELj2ELj2ELj2ELj2EEEN3tla5tupleIJNSE_1CILi128EEENSG_ILi256EEESI_EEENSF_IJSH_SI_NSG_ILi64EEEEEEu6__bf16u6__bf16u6__bf16vNS7_4Tile17PackedTileCopyTlaISC_u6__bf16NS6_6layout8RowMajorEu6__bf16SP_u6__bf16SP_vLb0ELNSM_16ScaleGranularityE0ENSM_8CopyToGMEEENSM_11TileMmadTlaISC_u6__bf16NSO_2zNEEEEENS_4Comm5Block16CommBlockMteUdmaINSY_9CommBlockINSX_19AtlasCommRemoteCopyISC_Lj2ELb1EEEJNS7_8GemmTypeIu6__bf16SP_LN7AscendC9TPositionE0EEES16_vNSX_4Tile14TileRemoteCopyISC_Lb1ES16_S16_vLNS_6detail10CopyDirectE1ELNS19_13CopyTransportE0EEENS6_8Epilogue4Tile27EpilogueIdentityTileSwizzleEEEENS10_INSX_23AtlasCommUdmaRemoteCopyISC_Lj2EEEJS16_S16_NS17_20TileRemoteCopyUdma2DISC_S16_S16_LS1A_1EEEEEEEENS0_5Block41BlockMmadSchedulerGmmAllToAllVTargetOrderINS0_14MoeConstraintsILj1ELj8ELj32EEEEENS_12CommEpilogue5Block41BlockCommSchedulerGmmAllToAllVTargetOrderIS1Q_Lb1EEELj2ELm419430400ELb1EEEEEvNT_6ParamsEPh, GroupedMatmul | 0.61 | 0.00 | 0.71–0.91 |
| moonep | MatMulV3, moonep_dispatch_udma_kernel, moonep_udma_combine_kernel | 0.46 | 0.00 | 20.83–21.68 |
| alltoallv_gmm_cann | AlltoAllvGroupedMatMulMc2AicpuKernel, AlltoAllvGroupedMatMul, GroupedMatMulAlltoAllv | 0.54 | 0.00 | 0.86–1.13 |
| mc2_dispatch_cann | MatMulV3, MoeDistributeDispatchV3, MoeDistributeCombineV3 | 0.76 | 0.00 | 3.08–4.21 |
| deepep_cann | MatMulV3, Mul, MoeEpCombineEpilogue | 0.86 | 0.00 | 0.66–0.81 |
| allgather_overlap_mindspeed | GroupedMatmul, HcclLaunchAicpuKernel, Mul | 4.43 | 2.20 | 1.17–1.35 |
| megamoe_td | GroupedMatmul, _kernel_dispatch_fc1, _kernel_combine_fc1_bwd_gemm_grid | 1.73 | 0.00 | 6.71–9.57 |
| deepep_sglang | MatMulV3, CamMoeDispatchNormal, CamMoeCombineNormal | 0.33 | 0.00 | 0.97–1.28 |

## 未取得完整结果

- `megamoe_cann` · [rank6]: ValueError: dispatch_quant_mode only supports 4 (MXFP) on Ascend950, got 0 (type: int).
- `hierarchical_alltoallv_mindspeed` · 上游要求多机，本轮使用单机 TP1。
