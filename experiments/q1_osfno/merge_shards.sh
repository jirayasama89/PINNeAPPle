#!/bin/sh
# Bring the result files of the remote shard branches into exp/q1-audit-osfno (results and shard logs only),
# register them in HISTORY.jsonl, commit and push. Files that already exist locally are never overwritten.
cd "$(dirname "$0")/../.." || exit 1
for b in $(git ls-remote --heads origin 'exp/q1-shard-*' | awk '{print $2}' | sed 's#refs/heads/##'); do
  git fetch -q origin "$b"
  for f in $(git ls-tree -r --name-only "origin/$b" -- experiments/q1_osfno/results experiments/q1_osfno/locks "experiments/q1_osfno/shard_*.log" "experiments/q1_osfno/p3_*.log" "experiments/q1_osfno/p7_*.log" "experiments/q1_osfno/p8_*.log"); do
    if [ ! -e "$f" ] || echo "$f" | grep -qE "(shard|p3|p7|p8)_.*\.log"; then
      mkdir -p "$(dirname "$f")"; git show "origin/$b:$f" > "$f"
    fi
  done
done
(cd experiments/q1_osfno && /home/user/venv-scout/bin/python history.py update)
git add -A experiments/q1_osfno
git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit -q -m "merge remote shard results into the main experimental branch

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NHHw7jaJ85LG8uAREdFU4p" && git push -q origin exp/q1-audit-osfno
