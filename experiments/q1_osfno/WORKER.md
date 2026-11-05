# Remote worker instructions (one cloud session per shard)

1. `cd` to the repository root, branch `exp/q1-audit-osfno` (fork cifer-sudo/PINNeAPPle).
2. `python3 -m venv /home/user/venv-scout && /home/user/venv-scout/bin/pip install -q numpy scipy scikit-learn xgboost torch matplotlib`
3. `cd experiments/q1_osfno && Q1_SLIM=1 /home/user/venv-scout/bin/python -u jobqueue.py <SHARD> 2>&1 | tee shard_<SHARD>.log`
   (the slim cache `data_slim/mafaulda_slim.npz` is committed; results go to `results/phase1/<task>/<model>/`).
4. Every ~30 min and at the end: `git checkout -B exp/q1-shard-<SHARD>`, `git add experiments/q1_osfno/results experiments/q1_osfno/shard_<SHARD>.log`,
   commit, `git push -u origin exp/q1-shard-<SHARD>`. Never modify other files, never push to exp/q1-audit-osfno.
5. Rules: do not change models, hyper-parameters, seeds or partitions; do not delete any result; if a job fails,
   record the error in the log and continue with the next job.
