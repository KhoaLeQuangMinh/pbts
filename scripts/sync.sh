#!/usr/bin/env bash
# Sync code to the server, recording the exact code version (commit + uncommitted diff) that runs will log.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
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
rsync -a --delete --exclude __pycache__ --exclude .pytest_cache code/ gpu-a240:/data/khoalq/pbts/code/
echo "synced $COMMIT (uncommitted files: $DIRTY)"
