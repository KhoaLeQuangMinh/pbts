#!/bin/bash
# Usage:  sbatch [--gres=gpu:a100_3g.20gb:1] -J <name> slurm/train.sbatch --model M --fold K --tag T --samples N
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=38G
#SBATCH -t 3-00:00:00
#SBATCH --signal=USR1@900
#SBATCH --requeue
#SBATCH -o /data/khoalq/pbts/logs/slurm-%x-%j.out
set -euo pipefail
cd /data/khoalq/pbts/code
export nnUNet_n_proc_DA=${NPROC_DA:-12}
echo "job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname) gres=$SLURM_JOB_GPUS args: $*"
srun ../env/bin/python -u -m pbts.train.run "$@"
