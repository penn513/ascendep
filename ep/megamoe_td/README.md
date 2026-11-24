# Mega-MoE-TD

demo 通过 MegaMoEFunction.apply 调用支持反向的 fused MoE。每个进程先建立 ACLSHMEM 堆，分配反向通信区，再构建 FusedMoEForward。固定专家归属，enable_moonep=False。上游原生 SwiGLU 没有限幅参数，demo 在计时前对当前 trace 全部路由计算投影最大值，只有限幅不生效时才进入验证。

双卡容器的安装顺序如下。使用 CANN 9.2.1、torch_npu、独立 Python 环境，以及 Triton-distributed-ascend 源码指定的 triton-ascend、AscendNPU-IR 和 SHMEM 版本。

```bash
source /opt/ae-cann/set_env.sh
cd /home/g00974569/ascendep
python -m venv --system-site-packages deps/tdenv
source deps/tdenv/bin/activate
git clone https://gitcode.com/alan_wangtianqi/Mega-MoE-TD deps/td
git -C deps/td checkout ce559af3f65adeb1ed268f7bfbc7e26d99f8821b
git clone https://gitcode.com/Ascend/Triton-distributed-ascend deps/triton-dist
git -C deps/triton-dist checkout a7b51aa42f4000f9a47e66e66ea296431517a098
git -C deps/triton-dist submodule update --init --recursive 3rdparty/triton-ascend 3rdparty/shmem
(cd deps/triton-dist/3rdparty/shmem && bash scripts/build.sh -python_extension -soc_type Ascend950)
python -m pip install --no-deps deps/triton-dist/3rdparty/shmem/dist/cann_shmem*.whl
export TRITON_USE_ASCEND=ON TRITON_BUILD_PROTON=OFF TRITON_BUILD_LITTLE_KERNEL=OFF
python -m pip wheel --no-deps --no-build-isolation deps/triton-dist/python -w deps/td-wheels
python -m pip install --no-deps deps/td-wheels/triton*.whl
bash ep/megamoe_td/build_shmem_templates.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m ep.suite --backend megamoe_td
```

本次 CANN 9.2.1 镜像的 C310 模板缺少 ACLSHMEM 的 MLIR 符号。构建脚本用配套 AscendNPU-IR 中的 Shmem.cpp 与 SHMEM 头文件补齐四种 O2 模板，将编译器和模板保存到项目内 deps/td-bishengir。suite 自动选择该目录。

根据所用镜像，编译器还需要上游 README 指定的配套 LLVM 和 AscendNPU-IR 设备工具。suite 为这个后端单独加载 deps/td/src 和上游的 ACLSHMEM 初始化辅助模块。
