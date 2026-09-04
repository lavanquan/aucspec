#!/bin/bash
set -x
cd /software/projects/pawsey1257/quanla/aucspec
module load pytorch/2.7.1-rocm6.3.3
export PYTORCH_ROCM_ARCH=gfx90a
export VLLM_TARGET_DEVICE=rocm
export MAX_JOBS=16
mkdir -p vllm_rocm_build
cd vllm_rocm_build
if [ ! -d vllm ]; then
  git clone --depth 1 --branch v0.9.1 https://github.com/vllm-project/vllm.git 2>&1
fi
cd vllm
echo "=== python3/pip sanity check ==="
which python3
python3 -m pip --version
echo "=== pip install rocm requirements ==="
python3 -m pip install --user -r requirements/rocm.txt 2>&1
echo "=== pip install vllm (editable, rocm target) ==="
python3 -m pip install --user --no-build-isolation -e . 2>&1
echo "=== BUILD_DONE exit=$? ==="
