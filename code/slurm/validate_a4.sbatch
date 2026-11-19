#!/bin/bash
# One-off validation of the remaining code (SegMamba build, external pipeline, report, dashboard, A4 smoke runs).
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH -c 12
#SBATCH --mem=34G
#SBATCH -t 06:00:00
#SBATCH -J validate_a4
#SBATCH -o /data/khoalq/pbts/logs/slurm-validate_a4-%j.out
cd /data/khoalq/pbts/code
P=../env/bin/python
export nnUNet_raw=/data/khoalq/pbts/nnunet/raw nnUNet_preprocessed=/data/khoalq/pbts/nnunet/preprocessed nnUNet_results=/tmp/nnres_$SLURM_JOB_ID nnUNet_n_proc_DA=10 MAX_JOBS=10
step() { echo "===== STEP: $1 ($(date +%H:%M:%S))"; shift; if "$@"; then echo "===== PASS"; else echo "===== FAIL (exit $?)"; fi; }
step "build SegMamba CUDA extensions" bash tools/build_segmamba.sh
step "tests: external preprocessing + stats" $P -m pytest -q -p no:cacheprovider -W ignore::DeprecationWarning tests/test_external.py tests/test_analysis.py
step "run report on smoke run" $P -m pbts.report.run_report ../runs/nnunet_resenc_l/resume_test/fold_0
step "comparison dashboard on smoke runs" $P -m pbts.analysis.compare --models nnunet_resenc_l primus_v2b --tag nnunet_resenc_l=resume_test primus_v2b=smoke --allow_partial --out ../reports/compare_smoke
for b in "swinunetr 1" "segmamba 1" "segmamba 2"; do step "memory $b (20GB slice)" $P tools/bench_a4.py $b; done
step "Swin UNETR smoke run" $P -u -m pbts.train.run --model swinunetr --fold 0 --tag smoke --smoke --no_wandb
step "SegMamba smoke run" $P -u -m pbts.train.run --model segmamba --fold 0 --tag smoke --smoke --no_wandb --batch_size 1
echo "ALL STEPS DONE $(date)"
