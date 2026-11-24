#!/usr/bin/env bash
# Add the pinned ACLSHMEM MLIR wrappers to a project-local copy of the C310 templates.
set -eo pipefail
cd "$(dirname "$0")/../.."
: "${ASCEND_HOME_PATH:?Source the CANN environment before running this script}"
base=$PWD/deps/triton-dist/3rdparty
template=$base/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/lib/Template
compiler=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin
work=$PWD/deps/td-bishengir
mkdir -p "$work"
cp -aL "$ASCEND_HOME_PATH/tools/bishengir/." "$work/"
for mode in aiv aic mix.aiv mix.aic; do
  arch=vec
  flags=()
  [[ $mode == *aic ]] && arch=cube
  [[ $mode == mix.* ]] && flags=(-cce-enable-mix -mllvm -enable-mix=true)
  "$compiler/bisheng" -O2 -x cce "$template/lib/Shmem/Shmem.cpp" -emit-llvm -c --cce-auto-sync=off --cce-aicore-only --cce-generic-addrspace=off -mllvm -disable-llvm-optzns --cce-enable-print "${flags[@]}" "--cce-aicore-arch=dav-c310-$arch" -std=c++17 -I "$template/include" -I "$base/shmem/include" -I "$base/shmem/src" -I "$base/shmem/src/device" -I "$ASCEND_HOME_PATH/aarch64-linux/include" -I "$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw" -I "$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/impl" -I "$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/interface" -o "$work/Shmem.$mode.bc" > "$work/$mode.log" 2>&1 || { grep -m8 'error:' "$work/$mode.log"; exit 1; }
  lib="meta_op.$mode.c310.O2.bc"
  "$compiler/llvm-link" "$ASCEND_HOME_PATH/tools/bishengir/lib/$lib" "$work/Shmem.$mode.bc" -o "$work/lib/$lib"
done
