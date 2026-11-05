"""Shared machinery of the Q1 programme: data loading from the per-run caches, locked partitions, inner (speed-block)
validation, model registry and a runner that stores per-run predictions, partitions, configuration and metrics.

Rules enforced here (see PROTOCOL.md):
* a run (5 s recording) is one sample; partitions are lists of run paths, saved with every result;
* the outer speed-tercile partitions are those of the manuscript (``classify.splits(protocol="speed")``) and are the
  same for every model;
* hyper-parameters are chosen ONLY on inner splits of the training runs (leave-one-speed-block-out inside the two
  training terciles), never on the held-out tercile;
* every result file contains the predictions and class probabilities of every test run.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pinneapple_analysis.inspection.rotating import classify as C  # noqa: E402

MAF_CACHE = os.environ.get("MAF_CACHE", "/home/user/data_raw/maf_cache")
COM_CACHE = os.environ.get("COM_CACHE", "/home/user/data_raw/com_cache")
RESULTS = ROOT / "experiments" / "q1_osfno" / "results"
BANDS = ("low", "mid", "high")


# ------------------------------------------------------------------ data
def load_rows(dataset: str = "mafaulda") -> list[dict[str, Any]]:
    """Per-run caches if present (full extraction), else the committed slim cache data_slim/<dataset>_slim.npz
    (identical spectra 0-8X and features; verified equal inputs and partitions). Q1_SLIM=1 forces the slim cache."""
    cache = MAF_CACHE if dataset == "mafaulda" else COM_CACHE
    slim = Path(__file__).resolve().parent / "data_slim" / f"{dataset}_slim.npz"
    if os.environ.get("Q1_SLIM") == "1" or not glob.glob(os.path.join(cache, "*.npz")):
        return list(np.load(slim, allow_pickle=True)["rows"])
    rows = []
    for f in sorted(glob.glob(os.path.join(cache, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        rows += list(z["row"]) if "row" in z.files else list(z["rows"])
    return rows


def task_data(rows, task):
    sel, y = C.select(rows, task)
    paths = np.array([r["path"] for r in sel])
    sp = np.array([r["speed_hz"] for r in sel], float)
    return sel, y, paths, sp


# ------------------------------------------------------------------ partitions
def outer_splits(y, sp, protocol="speed", seed=0):
    """Exactly the manuscript's partitions: ``classify.splits``."""
    return [(n, tr, te) for n, tr, te in C.splits(y, sp, protocol, seed=seed)]


def band_of(name: str) -> str:
    for b in BANDS:
        if f"_{b}_" in name:
            return b
    return name


def inner_splits(sp_train: np.ndarray, n_blocks: int = 4):
    """Leave-one-speed-block-out inside the training runs: contiguous speed blocks by quantile of the training
    speeds. Used only for model / hyper-parameter selection."""
    edges = np.quantile(sp_train, np.linspace(0, 1, n_blocks + 1))
    blk = np.clip(np.searchsorted(edges, sp_train, side="right") - 1, 0, n_blocks - 1)
    return [(f"inner{b}", np.flatnonzero(blk != b), np.flatnonzero(blk == b)) for b in range(n_blocks)]


def partition_record(paths, splits):
    return {n: {"train": paths[tr].tolist(), "test": paths[te].tolist()} for n, tr, te in splits}


def partition_hash(paths, splits) -> str:
    h = hashlib.sha1()
    for n, tr, te in splits:
        h.update(n.encode()); h.update(",".join(sorted(paths[te])).encode())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ metrics
def metrics(y, pred, n_cls):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    return {"macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(n_cls)), zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(np.mean(y == pred)),
            "confusion": confusion_matrix(y, pred, labels=list(range(n_cls))).tolist(),
            "per_class_f1": f1_score(y, pred, average=None, labels=list(range(n_cls)), zero_division=0).tolist()}


# ------------------------------------------------------------------ runner
def run(task, model_name, fit_predict, rows, seed, exp, protocol="speed", config=None, dataset="mafaulda",
        overwrite=False, log=print):
    """Fit/predict on every outer split, save one JSON: results/<exp>/<task>/<model>/<protocol>_seed<k>.json."""
    out_f = RESULTS / exp / task / model_name / f"{protocol}_seed{seed}.json"
    if out_f.exists() and not overwrite:
        return json.load(open(out_f))
    sel, y, paths, sp = task_data(rows, task)
    n_cls = len(C.TASKS[task])
    splits = outer_splits(y, sp, protocol, seed)
    pred = np.full(len(y), -1)
    proba = np.full((len(y), n_cls), np.nan)
    per, t_fit = {}, 0.0
    extra = {}
    for name, tr, te in splits:
        t0 = time.perf_counter()
        p, pr, info = fit_predict(sel, y, sp, tr, te, seed)
        t_fit += time.perf_counter() - t0
        pred[te] = p
        if pr is not None:
            proba[te] = pr
        per[name] = metrics(y[te], p, n_cls)
        extra[name] = info or {}
    import platform
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "experiments/q1_osfno/q1lib.py",
                                              "experiments/q1_osfno/models.py", "experiments/q1_osfno/models2.py",
                                              "pinneapple_analysis/inspection/rotating"], cwd=ROOT, text=True).strip())
    except Exception:  # noqa: BLE001
        commit, dirty = "unknown", None
    import sklearn
    import torch
    provenance = {"code_commit": commit, "code_dirty": dirty, "python": platform.python_version(),
                  "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
                  "utc_finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    res = {"provenance": provenance, "task": task, "model": model_name, "seed": seed, "protocol": protocol, "dataset": dataset,
           "classes": C.TASKS[task], "n": int(len(y)), "config": config or {},
           "partition_hash": partition_hash(paths, splits), "partitions": partition_record(paths, splits),
           "pooled": metrics(y, pred, n_cls), "per_split": per, "split_info": extra, "seconds_total": t_fit,
           "runs": [{"path": str(paths[i]), "y": int(y[i]), "pred": int(pred[i]), "speed_hz": float(sp[i]),
                     "proba": None if np.isnan(proba[i]).any() else [round(float(v), 5) for v in proba[i]]}
                    for i in range(len(y))]}
    out_f.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_f, "w"))
    log(f"{exp} {task:12s} {model_name:22s} s{seed} {protocol} macro-F1 {res['pooled']['macro_f1']:.3f} "
        f"({', '.join(f'{band_of(k)} {v['macro_f1']:.2f}' for k, v in per.items())}) {t_fit:.0f}s")
    return res
