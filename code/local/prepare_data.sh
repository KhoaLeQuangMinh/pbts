#!/bin/bash
# Prepare the study data on another machine exactly as on gpu-a240 (every step is deterministic, same code, same
# package versions), then fingerprint it for the file-by-file comparison with gpu-a240:
#   python tools/data_fingerprint.py --compare <gpu-a240.json> <this machine's .json>
# Needs the Kaggle download in data/kagglehub_cache (download.py). Usage: MACHINE=blackwell bash local/prepare_data.sh
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
cd "$P/code"
PY=$P/env/bin/python; W=${WORKERS:-12}; M=${MACHINE:-$(hostname -s)}
export nnUNet_raw=$P/nnunet/raw nnUNet_preprocessed=$P/nnunet/preprocessed nnUNet_results=$P/nnunet/results nnUNet_compile=false
step() { echo "===== $1 ($(date +%H:%M:%S))"; shift; "$@"; }
step "convert, verified case by case against the manifest" $PY -m pbts.data.convert --workers $W
step "study split (locked test + 5 inner folds)" $PY -m pbts.data.splits
step "head crop + per-channel normalisation" $PY -m pbts.data.preprocess --workers $W
step "nnU-Net dataset fingerprint" $P/env/bin/nnUNetv2_extract_fingerprint -d 502 -np $W
step "nnU-Net ResEnc-L plan" $P/env/bin/nnUNetv2_plan_experiment -d 502 -pl nnUNetPlannerResEncL
step "our whole-head plan + inner-fold splits" $PY -m pbts.data.nnunet_setup
step "nnU-Net preprocessing (3d_wholehead)" $P/env/bin/nnUNetv2_preprocess -d 502 -plans_name pbtsWholeHeadPlans -c 3d_wholehead -np $W
step "fingerprint of all prepared data" $PY -W ignore tools/data_fingerprint.py "$P/logs/data_fingerprint_$M.json" --workers $W
echo "DATA READY $(date)"
