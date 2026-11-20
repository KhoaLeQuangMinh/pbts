#!/bin/bash
# Validate a second machine before it trains (minutes of GPU, nothing retrained):
#   1. its prepared data is identical to gpu-a240's (file-by-file fingerprint)
#   2. a model trained on gpu-a240 (ResEnc fold 0) predicts the same here (tools/cross_machine_check.py)
#   3. smoke run of every model (train 20 steps -> predict + score all validation scans), one more with W&B on
#   4. throughput: 1 job vs 2 jobs at once on the GPU (Swin + ResEnc, 3 x 50 steps each)
# Usage: MACHINE=blackwell bash local/validate_machine.sh      (needs local/prepare_data.sh done + checks/ copied)
set -u
P=$(cd "$(dirname "$0")/../.." && pwd)
cd "$P/code"
PY=$P/env/bin/python; M=${MACHINE:-$(hostname -s)}
export nnUNet_n_proc_DA=${NPROC_DA:-10}
step() { echo "===== STEP: $1 ($(date +%H:%M:%S))"; shift; if "$@"; then echo "===== PASS"; else echo "===== FAIL (exit $?)"; fi; }
train() { $PY -u -W ignore -m pbts.train.run --fold 0 "$@"; }
step "1. data identical to gpu-a240" $PY tools/data_fingerprint.py --compare "$P/logs/data_fingerprint_gpu-a240.json" "$P/logs/data_fingerprint_$M.json"
step "2. cross-machine prediction check (ResEnc fold-0 weights trained on gpu-a240)" $PY -W ignore tools/cross_machine_check.py \
     "$P/checks/resenc_f0_gpu-a240" "$P/checks/resenc_f0_gpu-a240/reference_predictions" "$P/checks/resenc_f0_gpu-a240/$M"
for m in nnunet_resenc_l primus_v2b segmamba swinunetr; do
  step "3. smoke run $m" train --model $m --tag smoke --smoke --no_wandb
done
step "3. smoke run with W&B (ResEnc)" train --model nnunet_resenc_l --tag smoke_wandb --smoke
T="--smoke --smoke_iters 50 --smoke_epochs 3 --no_wandb"
step "4. throughput: Swin alone" train --model swinunetr --tag tput_alone $T
step "4. throughput: ResEnc alone" train --model nnunet_resenc_l --tag tput_alone $T
pair() { train --model swinunetr --tag tput_pair $T > "$P/logs/tput_pair_swin.log" 2>&1 & a=$!
         train --model nnunet_resenc_l --tag tput_pair $T > "$P/logs/tput_pair_resenc.log" 2>&1 & b=$!
         wait $a; ra=$?; wait $b; rb=$?; return $(( ra || rb )); }
step "4. throughput: Swin + ResEnc at the same time" pair
$PY - "$P/runs" <<'EOF'
import json, sys
from pathlib import Path
R = Path(sys.argv[1])
def per_step(model, tag):
    ep = [json.loads(l) for l in open(R / model / tag / "fold_0/epochs.jsonl")][1:]     # epoch 0 = warm-up
    it = 50
    return sum(e["epoch_s"] for e in ep) / (len(ep) * it), sum(e["data_wait_s"] for e in ep) / (len(ep) * it)
rel = 0
for model in ("swinunetr", "nnunet_resenc_l"):
    (a, wa), (p, wp) = per_step(model, "tput_alone"), per_step(model, "tput_pair")
    rel += a / p
    print(f"{model:16s} alone {a:.2f} s/step (data wait {wa:.2f}) | paired {p:.2f} s/step (data wait {wp:.2f}) | speed {a / p:.0%} of alone")
print(f"THROUGHPUT with 2 jobs = {rel:.2f} x one job at a time  ->  {'2 jobs pay off' if rel >= 1.3 else 'keep 1 job at a time'}")
EOF
echo "VALIDATION DONE $(date)"
