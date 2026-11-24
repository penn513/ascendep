# mc2_dispatch_cann

本 demo 在 Ascend 950DT 上用 CANN DispatchV2 分发 token，在所属卡完成专家计算，再用 CombineV2 合并结果。反向用 DispatchV2 发送输出梯度，CombineV2 回传输入梯度。各 rank 通过 HCCL 收集路由权重，反向按源 rank 归约对应梯度。

每卡输入仍为 4096 tokens。当前设备上的原生 DispatchV2 单次最多支持 512 tokens，demo 将本地序列连续切成 8 块，计时覆盖所有块的调度、计算和合并。单次直接传入 4096 tokens 的原始错误见双卡报告目录中的 mc2_dispatch_shape_probe.log。小规模验收使用 H1024 以满足算子的隐藏维下限。

使用与驱动匹配的 CANN 9.2.1、PyTorch 2.10.0 及 torch_npu 2.10.0.post4。在项目的双卡容器中执行：

```bash
source /opt/ae-cann/set_env.sh
cd /home/g00974569/ascendep
export ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 HCCL_BUFFSIZE=512
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m ep.suite --backend mc2_dispatch_cann
```
