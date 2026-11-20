#!/usr/bin/env bash
# Rebuild gpu-a240's environments on another machine, entirely inside the project root (no system changes):
#   env_eval/ = code/envs/gpu-a240_env_eval.txt exactly (CPU only: the official scorer)
#   env/      = code/envs/gpu-a240_env.txt, except the torch stack, which must support the local GPU
#               (merge rule and the list of differences: code/envs/merge_reqs.py -> logs/env_deviations.txt)
#   cuda<NN>/ = CUDA toolkit (via micromamba) used only to compile SegMamba's kernels
# Usage on blackwell (RTX PRO 5000 Blackwell = sm_120, needs CUDA >= 12.8, i.e. torch >= 2.7):
#   bash code/envs/setup_env.sh 2.7.1 0.22.1 cu128 12.8.1
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
TORCH=$1 TV=$2 CU=$3 CUDA_VER=$4
PY=3.11.15                                      # gpu-a240's interpreter
export UV_CACHE_DIR=$P/.uv_cache UV_PYTHON_INSTALL_DIR=$P/python XDG_CACHE_HOME=$P/.cache UV_HTTP_TIMEOUT=600
export MAMBA_ROOT_PREFIX=$P/.micromamba
UV=${UV:-$P/.bootstrap/bin/uv}
E=$P/code/envs; L=$P/logs; mkdir -p $P/tmp $L $P/bin
step() { echo "===== $1 ($(date +%H:%M:%S))"; }

step "env_eval (identical to gpu-a240)"
[ -x $P/env_eval/bin/python ] || $UV venv -q --python $PY $P/env_eval
$UV pip install -q --python $P/env_eval/bin/python -r $E/gpu-a240_env_eval.txt

step "env: torch $TORCH+$CU and torchvision $TV+$CU (PyTorch index, without their dependencies)"
[ -x $P/env/bin/python ] || $UV venv -q --python $PY $P/env
$UV pip install -q --python $P/env/bin/python --no-deps "torch==$TORCH+$CU" "torchvision==$TV+$CU" \
    --index-url https://download.pytorch.org/whl/$CU

step "env: everything else at gpu-a240's versions; torch's own dependencies (NVIDIA libraries) from PyPI"
$P/env/bin/python - > $P/tmp/torch_reqs.txt <<'PY'
import importlib.metadata as md
for pkg in ("torch", "torchvision"):
    for r in md.requires(pkg) or []:
        if "extra ==" not in r:
            print(r)
PY
$P/.bootstrap/bin/python $E/merge_reqs.py $E/gpu-a240_env.txt $P/tmp/torch_reqs.txt $P/tmp/env_requirements.txt $L/env_deviations.txt
$UV pip install -q --python $P/env/bin/python -r $P/tmp/env_requirements.txt
$UV pip freeze --python $P/env/bin/python > $L/env_freeze.txt

step "CUDA toolkit $CUDA_VER (for compiling SegMamba only)"
[ -x $P/bin/micromamba ] || curl -sL -o $P/bin/micromamba \
    https://github.com/mamba-org/micromamba-releases/releases/download/2.0.5-0/micromamba-linux-64
chmod +x $P/bin/micromamba
CUDA_DIR=$P/cuda$(echo $CUDA_VER | cut -d. -f1,2 | tr -d .)
[ -x $CUDA_DIR/bin/nvcc ] || $P/bin/micromamba create -q -y -p $CUDA_DIR -c nvidia/label/cuda-$CUDA_VER -c conda-forge cuda-toolkit
$CUDA_DIR/bin/nvcc --version | tail -2

step "GPU check"
$P/env/bin/python -W ignore $P/code/envs/gpu_check.py
echo "SETUP DONE $(date)"
