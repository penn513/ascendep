# alltoallv_gmm_cann

通过 torch_npu 的 `npu_alltoallv_gmm` 和 `npu_gmm_alltoallv` 调用 CANN 通算融合算子。demo 按专家打包 token，交换路由计数，在两次融合算子间执行限幅 SwiGLU，并按原 token 顺序加权合并。反向通过逆向通信和 `npu_grouped_matmul` 计算输入、专家权重及路由权重梯度。FSDPTurbo MC2 也使用这两个 CANN 算子，其框架调度开销单独计量。

本轮使用项目验证容器中的 CANN 9.2.1 和 torch_npu 2.10.0.post4。需要匹配的 Ascend 950 算子包、驱动与 HCCL。容器中已挂载两张物理卡，在项目根目录执行：

```bash
source /opt/ae-cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m ep.suite --backend alltoallv_gmm_cann
```
