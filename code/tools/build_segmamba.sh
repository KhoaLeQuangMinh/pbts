#!/usr/bin/env bash
# Build SegMamba's bundled CUDA extensions (causal_conv1d 1.0.0, mamba_ssm 1.0.1 with bimamba v3) into env/.
# Run on the server inside a SLURM job:  srun -p gpu -c 8 --mem=24G -t 02:00:00 bash tools/build_segmamba.sh
set -euo pipefail
P=/data/khoalq/pbts
export CUDA_HOME=$P/cuda124 PATH=$P/cuda124/bin:$PATH TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=${MAX_JOBS:-8}
export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE UV_CACHE_DIR=/data/khoalq/.uv_cache
uv pip install -q --python $P/env/bin/python ninja packaging setuptools wheel
cd $P/repos/SegMamba/causal-conv1d && uv pip install --python $P/env/bin/python --no-build-isolation --no-deps .
cd $P/repos/SegMamba/mamba && uv pip install --python $P/env/bin/python --no-build-isolation --no-deps .
$P/env/bin/python -c "import torch, causal_conv1d, mamba_ssm, selective_scan_cuda; print('built OK', torch.__version__)"
