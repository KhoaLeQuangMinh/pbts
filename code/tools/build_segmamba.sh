#!/usr/bin/env bash
# Build SegMamba's bundled CUDA extensions (causal_conv1d 1.0.0, mamba_ssm 1.0.1 with bimamba v3) into env/.
#   gpu-a240 (SLURM): srun -p gpu -c 8 --mem=24G -t 02:00:00 bash tools/build_segmamba.sh        (CUDA 12.4, sm_80)
#   blackwell:        CUDA_HOME=<root>/cuda128 TORCH_CUDA_ARCH_LIST=12.0 PBTS_EXTRA_CUDA_ARCHS=120 bash tools/build_segmamba.sh
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)            # project root (holds code/, env/, repos/)
export CUDA_HOME=${CUDA_HOME:-$P/cuda124}
export PATH=$CUDA_HOME/bin:$PATH TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" MAX_JOBS=${MAX_JOBS:-8}
export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE UV_CACHE_DIR=${UV_CACHE_DIR:-$P/.uv_cache}
# conda's cuda-toolkit (12.8 on blackwell) keeps headers/libs under targets/x86_64-linux, which torch's extension
# builder does not search; add them (no effect where they are already in $CUDA_HOME/include)
if [ -d "$CUDA_HOME/targets/x86_64-linux/include" ]; then
  export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
UV=${UV:-uv}
$UV pip install -q --python $P/env/bin/python ninja packaging setuptools wheel
# the unused LM-head import would need `transformers` (see tools/patch_mamba_init.py)
$P/env/bin/python $P/code/tools/patch_mamba_init.py $P/repos/SegMamba/mamba/mamba_ssm/__init__.py
# optional extra GPU architectures (PBTS_EXTRA_CUDA_ARCHS); no change when unset (see tools/patch_mamba_build.py)
$P/env/bin/python $P/code/tools/patch_mamba_build.py $P/repos/SegMamba/causal-conv1d/setup.py $P/repos/SegMamba/mamba/setup.py
cd $P/repos/SegMamba/causal-conv1d && $UV pip install --python $P/env/bin/python --no-build-isolation --no-deps .
cd $P/repos/SegMamba/mamba && $UV pip install --python $P/env/bin/python --no-build-isolation --no-deps .
cd /tmp && $P/env/bin/python -c "import torch, causal_conv1d, mamba_ssm, selective_scan_cuda; print('built OK', torch.__version__, mamba_ssm.__file__)"
