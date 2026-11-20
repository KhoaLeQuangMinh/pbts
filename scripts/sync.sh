#!/usr/bin/env bash
# Sync code to the servers, recording the exact code version (commit + uncommitted diff) that runs will log.
#   scripts/sync.sh            -> every server
#   scripts/sync.sh blackwell  -> one server
# Each server keeps the project in its own folder; code/ goes to <root>/code/ (config.py finds the root from there).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
root_of() { case $1 in gpu-a240) echo /data/khoalq/pbts;; blackwell) echo /home/khoalq/pbts;; *) echo "unknown host $1" >&2; exit 2;; esac; }
if [ $# -gt 0 ]; then TARGETS=("$@"); else TARGETS=(gpu-a240 blackwell); fi
COMMIT=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain code | wc -l | tr -d ' ')
git diff HEAD -- code > code/VERSION.diff
python3 - "$COMMIT" "$DIRTY" <<'PY'
import json, sys, time, hashlib
diff = open("code/VERSION.diff","rb").read()
json.dump({"commit": sys.argv[1], "uncommitted_files": int(sys.argv[2]),
           "diff_sha256": hashlib.sha256(diff).hexdigest(), "synced": time.strftime("%Y-%m-%dT%H:%M:%S")},
          open("code/VERSION.json","w"), indent=1)
PY
fail=0
for host in "${TARGETS[@]}"; do
  root=$(root_of "$host")
  if rsync -a --delete --exclude __pycache__ --exclude .pytest_cache code/ "$host:$root/code/"; then
    echo "synced $COMMIT to $host:$root (uncommitted files: $DIRTY)"
  else
    echo "WARNING: sync to $host failed" >&2; fail=1
  fi
done
exit $fail
