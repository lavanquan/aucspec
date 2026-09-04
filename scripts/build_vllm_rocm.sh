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
python3 -m pip show torch 2>&1 | head -3

# The container's python3 is PEP-668 externally-managed (Debian-based
# image). A --system-site-packages venv on top of it does NOT see the
# container's torch/triton (confirmed by hand: the venv reported "No module
# named 'torch'"), so we don't use a venv. --break-system-packages against
# the container's real site-packages silently falls back to a --user
# install under $HOME/.local, which is quota-limited (confirmed: previous
# run died with "Disk quota exceeded" partway through, meaning
# setuptools_scm and other deps never finished installing either).
# Redirect that --user target to project space, which has plenty of quota.
export PYTHONUSERBASE=/software/projects/pawsey1257/quanla/aucspec/.pythonuserbase-rocm-vllm
mkdir -p "$PYTHONUSERBASE"
export PATH="$PYTHONUSERBASE/bin:$PATH"
echo "PYTHONUSERBASE=$PYTHONUSERBASE"

echo "=== pip install rocm requirements ==="
python3 -m pip install --break-system-packages -r requirements/rocm.txt 2>&1
echo "=== pip install vllm (editable, rocm target) ==="
python3 -m pip install --break-system-packages --no-build-isolation -e . 2>&1
echo "=== BUILD_DONE exit=$? ==="
