source /opt/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_LIBRARY_PATH=/work/host_runtime_urma:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
export PYTHONPATH=/home/g00974569/ascendep:/opt/fsdpturbo:/opt/atk-extra:$PYTHONPATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCENDEP_DEVICES:-0,1,2,3,4,5,6,7}"
