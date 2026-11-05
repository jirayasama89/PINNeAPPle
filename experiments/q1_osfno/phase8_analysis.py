"""Phase 8 (part 1, no training): calibration, per-class F1, confusion, speed-resolved accuracy and cost, computed
from the stored per-run probabilities of the Phase 1 confirmations (same partitions, 5 seeds).

Calibration: expected calibration error (15 equal-width bins on the max probability), Brier score, NLL; per seed,
then mean ± 95 % t-interval. Speed-resolved: accuracy of the seed-vote prediction in 10 equal-count speed bins.
Cost: fit seconds per outer fold and number of trainable parameters when stored.
Writes results/phase8_calibration.json and PHASE8_CALIBRATION.md.
usage: python phase8_analysis.py
"""
import json

import numpy as np

from q1lib import RESULTS
from stats_q1 import ci, load

TASKS = ["misalignment", "unbalance", "bearings", "all", "comfaulda7"]
MODELS = ["rf", "mlp", "kan", "svm", "xgb", "cnn-spec", "fno-trainonly", "osfno-A-nospeed", "osfno-B-head", "osfno-C-film",
          "osfno-E-phys", "osfno-B-dro", "osfno-B-supcon", "fuse-rf+cnn-spec"]


def calib(y, P, n_bins=15):
    P = np.clip(np.asarray(P, float), 1e-12, 1)
    P = P / P.sum(1, keepdims=True)
    conf, pred = P.max(1), P.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    Y = np.eye(P.shape[1])[y]
    return {"ece": float(ece), "brier": float(np.mean(np.sum((P - Y) ** 2, 1))),
            "nll": float(-np.mean(np.log(P[np.arange(len(y)), y]))), "mean_conf": float(conf.mean()), "acc": float(acc.mean())}


def main():
    out, lines = {}, ["# Phase 8: calibration, per-class F1 and cost (from stored Phase 1 probabilities)", "",
                      "KAN and the train-only FNO (repository classifiers) store no probabilities and are omitted. ECE (15 bins), Brier and NLL pooled over the three held-out terciles; mean ± 95 % t-interval over "
                      "5 seeds. Lower is better. Speed bins: accuracy of the 5-seed vote in 10 equal-count speed bins.", ""]
    for t in TASKS:
        lines += [f"## {t}", "", "| model | macro-F1 | ECE | Brier | NLL | mean conf. | acc. | fit s / fold | params |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            R = load("phase1", m, t)
            if len(R) < 5 or not isinstance(R[0]["runs"][0].get("proba"), list):
                continue                        # kan, fno-trainonly: repo classifiers store no probabilities
            cs = [calib(np.array([r["y"] for r in d["runs"]]), [r["proba"] for r in d["runs"]]) for d in R]
            f1 = np.array([d["pooled"]["macro_f1"] for d in R])
            agg = {k: (float(np.mean([c[k] for c in cs])), ci(np.array([c[k] for c in cs]))[1]) for k in cs[0]}
            fit = [v.get("fit_seconds") for d in R for v in (d.get("split_info") or {}).values() if isinstance(v, dict)]
            fit = [x for x in fit if x is not None]
            npar = [v.get("n_params") for d in R for v in (d.get("split_info") or {}).values() if isinstance(v, dict)]
            npar = [x for x in npar if x]
            # per-class F1 (mean over seeds) and seed-vote speed curve
            pcf = np.mean([d["pooled"]["per_class_f1"] for d in R], 0)
            runs0 = R[0]["runs"]
            y = np.array([r["y"] for r in runs0])
            sp = np.array([r["speed_hz"] for r in runs0])
            votes = np.array([[r["pred"] for r in d["runs"]] for d in R])
            vote = np.array([np.bincount(votes[:, i], minlength=len(R[0]["classes"])).argmax() for i in range(len(y))])
            order = np.argsort(sp)
            bins = np.array_split(order, 10)
            curve = [{"speed_lo": float(sp[b].min()), "speed_hi": float(sp[b].max()), "acc": float(np.mean(vote[b] == y[b]))}
                     for b in bins]
            out[f"{t}/{m}"] = {"macro_f1": [float(f1.mean()), ci(f1)[1]], **{k: list(v) for k, v in agg.items()},
                               "per_class_f1": dict(zip(R[0]["classes"], map(float, pcf))), "speed_curve": curve,
                               "fit_seconds_per_fold": float(np.mean(fit)) if fit else None,
                               "n_params": int(np.median(npar)) if npar else None,
                               "confusion_seed_sum": np.sum([d["pooled"]["confusion"] for d in R], 0).tolist()}
            lines.append(f"| {m} | {f1.mean():.3f} | {agg['ece'][0]:.3f} ± {agg['ece'][1]:.3f} | {agg['brier'][0]:.3f} | "
                         f"{agg['nll'][0]:.3f} | {agg['mean_conf'][0]:.3f} | {agg['acc'][0]:.3f} | "
                         f"{np.mean(fit):.0f} | {int(np.median(npar)) if npar else '-'} |" if fit else
                         f"| {m} | {f1.mean():.3f} | {agg['ece'][0]:.3f} ± {agg['ece'][1]:.3f} | {agg['brier'][0]:.3f} | "
                         f"{agg['nll'][0]:.3f} | {agg['mean_conf'][0]:.3f} | {agg['acc'][0]:.3f} | - | - |")
        lines.append("")
        # per-class table for the best feature model, the CNN and OS-FNO-A
        cls = None
        rows = []
        for m in ["rf", "mlp", "cnn-spec", "osfno-A-nospeed", "fuse-rf+cnn-spec"]:
            if f"{t}/{m}" in out:
                pc = out[f"{t}/{m}"]["per_class_f1"]
                cls = list(pc)
                rows.append(f"| {m} | " + " | ".join(f"{pc[c]:.2f}" for c in cls) + " |")
        if cls:
            lines += ["Per-class F1 (mean of 5 seeds):", "", "| model | " + " | ".join(cls) + " |",
                      "|---" * (len(cls) + 1) + "|", *rows, ""]
    json.dump(out, open(RESULTS / "phase8_calibration.json", "w"), indent=1)
    open(RESULTS.parent / "PHASE8_CALIBRATION.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
