#!/bin/sh
# Every 10 minutes: register new result files in HISTORY.jsonl, regenerate SUMMARY.md, commit and push.
# Each commit is one reproducibility checkpoint (results + ledger + summary at that time).
cd "$(dirname "$0")/../.." || exit 1
PY=/home/user/venv-scout/bin/python
while true; do
  if [ -n "$(git status --porcelain experiments/q1_osfno)" ]; then
    (cd experiments/q1_osfno && $PY history.py update) >> /home/user/data_raw/autosave.log 2>&1
    git add -A experiments/q1_osfno
    N=$(git diff --cached --name-only | grep -c "results/.*_seed")
    git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit -q -m "autosave: $N result file(s), ledger and summary updated

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NHHw7jaJ85LG8uAREdFU4p" >> /home/user/data_raw/autosave.log 2>&1
    git push -q origin exp/q1-audit-osfno >> /home/user/data_raw/autosave.log 2>&1
    echo "$(date -u +%FT%TZ) checkpoint $(git rev-parse --short HEAD) ($N result files)" >> /home/user/data_raw/autosave.log
  fi
  sleep 600
done
