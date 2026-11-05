# Remote worker instructions: Phase 3 (nested HP selection) and Phase 7 (compound faults)

1. Repository root, branch `exp/q1-audit-osfno` (fork cifer-sudo/PINNeAPPle). Do NOT set Q1_SMOKE.
2. `python3 -m venv /home/user/venv-scout && /home/user/venv-scout/bin/pip install -q numpy scipy scikit-learn xgboost torch matplotlib`
3. `cd experiments/q1_osfno`, then for a Phase 3 shard `<ds> <task> <family>`:
   `Q1_SLIM=1 /home/user/venv-scout/bin/python -u phase3.py select <ds> <task> <family> 2>&1 | tee p3_<task>_<family>.log`
   then (only after `locks/phase3_<task>_<family>.json` exists)
   `Q1_SLIM=1 /home/user/venv-scout/bin/python -u phase3.py confirm <ds> <task> <family> 2>&1 | tee -a p3_<task>_<family>.log`
   For a Phase 7 shard: `Q1_SLIM=1 /home/user/venv-scout/bin/python -u phase7.py run <method> 2>&1 | tee p7_<method>.log`
4. Push the lock file AS SOON AS selection finishes (before confirmation), then every ~30 min and at the end:
   `git checkout -B exp/q1-shard-<NAME>`, `git add experiments/q1_osfno/locks experiments/q1_osfno/results experiments/q1_osfno/*.log`,
   commit, `git push -u origin exp/q1-shard-<NAME>`. Never modify code or other files; never push to exp/q1-audit-osfno.
5. Rules: do not change grids, epochs, seeds or partitions; do not delete any result; the test tercile is never used for
   selection (the script enforces it). If a job fails, record the error in the log, push, and stop.

## Phase 8 robustness shard (NAME=P8-<task>)
`for m in rf mlp xgb cnn-spec osfno-A-nospeed osfno-B-head osfno-E-phys; do Q1_SLIM=1 /home/user/venv-scout/bin/python -u phase8_robust.py <ds> <task> $m 2>&1 | tee -a p8_<task>.log; done`
(existing result files are skipped). Push `experiments/q1_osfno/results experiments/q1_osfno/p8_<task>.log` to
`exp/q1-shard-P8-<task>` after each model and at the end, as in step 4.
