#!/bin/bash
# Validation, part B (full A100): the SegMamba and Swin UNETR steps that failed in job 2740.
# Swin ran out of memory on the 20 GB slice and SegMamba could not import (unused LM-head import, now patched).
# Measures each model's true training peak on the 40 GB GPU (for the queue's GPU choice), picks SegMamba's
# batch size (its original 2 if it fits the full A100 with the queue's margin, else 1), then runs both smoke runs
# (train + whole-frame TTA prediction + scoring).
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=38G
#SBATCH -t 03:00:00
#SBATCH -J validate_b
#SBATCH -o /data/khoalq/pbts/logs/slurm-validate_b-%j.out
cd /data/khoalq/pbts/code
P=../env/bin/python
export nnUNet_raw=/data/khoalq/pbts/nnunet/raw nnUNet_preprocessed=/data/khoalq/pbts/nnunet/preprocessed nnUNet_results=/tmp/nnres_$SLURM_JOB_ID nnUNet_n_proc_DA=12
B=../logs/bench_a100_$SLURM_JOB_ID.jsonl
step() { echo "===== STEP: $1 ($(date +%H:%M:%S))"; shift; if "$@"; then echo "===== PASS"; else echo "===== FAIL (exit $?)"; fi; }
for b in "swinunetr 1" "segmamba 1" "segmamba 2"; do
  step "memory $b (full A100)" bash -c "$P -W ignore tools/bench_a4.py $b | tee -a $B"
done
SEG_BS=$($P - "$B" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
r = [x for x in rows if x["model"] == "segmamba" and x["bs"] == 2]
ok = r and isinstance(r[0]["peak_GB"], (int, float)) and r[0]["peak_GB"] * 1.15 + 1.0 <= 39.4
print(2 if ok else 1)
EOF
)
echo "SegMamba batch size chosen: $SEG_BS (original recipe 2 if it fits the full A100 with margin)"
step "Swin UNETR smoke run" $P -u -m pbts.train.run --model swinunetr --fold 0 --tag smoke --smoke --no_wandb
step "SegMamba smoke run (bs $SEG_BS)" $P -u -m pbts.train.run --model segmamba --fold 0 --tag smoke --smoke --no_wandb --batch_size $SEG_BS
echo "ALL STEPS DONE $(date)"
