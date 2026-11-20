#!/bin/bash
# Machines without SLURM (blackwell): the local equivalent of slurm/train.sbatch. Started by the queue
# (pbts.final.queue --machine blackwell) as its own session, so it survives the queue being restarted.
#   bash local/run_job.sh --model M --fold K --tag T --samples N [...]
# Pause a run cleanly: kill -USR1 <pid>  -> checkpoint at the end of the epoch, exit 0; starting it again resumes.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
cd "$P/code"
export nnUNet_n_proc_DA=${NPROC_DA:-12}
echo "local job pid $$ on $(hostname) $(date -u +%FT%T) args: $*"
exec "$P/env/bin/python" -u -m pbts.train.run "$@"
