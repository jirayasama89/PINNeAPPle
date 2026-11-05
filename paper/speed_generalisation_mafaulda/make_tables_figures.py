"""Builds every table (tables/*.tex), figure (figures/*.pdf) and number quoted in the manuscript (stats.json) from
the committed result files in docs/research/results -- no number of the paper is typed by hand.

Run from the repository root:  python paper/speed_generalisation_mafaulda/make_tables_figures.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "docs" / "research" / "results"
OUT = Path(__file__).resolve().parent
TASKS = ("misalignment", "unbalance", "bearings", "all")
TNAME = {"misalignment": "T1 misalignment", "unbalance": "T2 unbalance", "bearings": "T3 bearings",
         "all": "T4 all ten states", "comfaulda7": "C1 COMFAULDA"}
MNAME = {"rf": "RF", "mlp": "MLP", "kan": "KAN", "fno": "FNO", "ocfno": "OC-FNO", "hnn_logreg": "HNN+LR",
         "hnn_rf": "HNN+RF"}


def load(name):
    return json.loads((RES / name).read_text())


def ci(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan")), float("nan"), float("nan")
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))), float(
        v.std(ddof=1))


def collect():
    seeds = load("mafaulda_classification_seeds.json")
    oc = {**load("ocfno_seeds_0_2.json"), **load("ocfno_seeds_3_4.json")}
    cb = load("comfaulda_seeds_baselines.json")
    c0 = load("comfaulda_results.json")
    vals: dict = {}
    bands: dict = {}

    def put(t, p, m, s, r):
        vals.setdefault((t, p, m), {})[int(s)] = r["macro_f1"]
        if p == "speed" and "per_split_macro_f1" in r:
            bands.setdefault((t, m), {})[int(s)] = list(r["per_split_macro_f1"].values())

    for s, tasks in seeds.items():
        for t, v in tasks.items():
            for key, r in v["results"].items():
                put(t, *key.split("/"), s, r)
    for s, tasks in oc.items():
        for t, v in tasks.items():
            if "speed/ocfno" in v["results"]:
                put(t, "speed", "ocfno", s, v["results"]["speed/ocfno"])
    for s, v in cb.items():
        for key, r in v["results"].items():
            put("comfaulda7", *key.split("/"), s, r)
    for key, r in c0["comfaulda7"]["results"].items():
        p, m = key.split("/")
        if 0 not in vals.get(("comfaulda7", p, m), {}):
            put("comfaulda7", p, m, 0, r)
    for t, v in load("mafaulda_hnn_baseline.json").items():
        for key, r in v["results"].items():
            p, m = key.split("/")
            vals.setdefault((t, p, m), {})[0] = r["macro_f1"]
    return vals, bands, c0, load("ocfno_ablation.json")


def fmt(vs, digits=3):
    if not vs:
        return "--"
    m, h, _ = ci(list(vs.values()))
    return f"{m:.{digits}f}" if len(vs) < 2 else f"{m:.{digits}f}\\,$\\pm$\\,{h:.{digits}f}"


def write(name, tex):
    (OUT / "tables" / name).write_text(tex)


def table_protocol(vals, proto, models, label, caption):
    rows = [f"{TNAME[t]} & " + " & ".join(fmt(vals.get((t, proto, m))) for m in models) + r" \\"
            for t in TASKS + ("comfaulda7",)]
    write(label.split(":")[1] + ".tex",
          r"\begin{table*}[t]" "\n" r"\centering" "\n" rf"\caption{{{caption}}}" "\n" rf"\label{{{label}}}" "\n"
          r"\resizebox{\textwidth}{!}{%" "\n" rf"\begin{{tabular}}{{l{'c' * len(models)}}}" "\n" r"\toprule" "\n"
          + "Task & " + " & ".join(MNAME[m] for m in models) + r" \\" "\n" r"\midrule" "\n" + "\n".join(rows)
          + "\n" r"\bottomrule" "\n" r"\end{tabular}}" "\n" r"\end{table*}" "\n")


def table_gain(vals, abl, st):
    rows = []
    for t in TASKS + ("comfaulda7",):
        oc, rf = vals.get((t, "speed", "ocfno"), {}), vals.get((t, "speed", "rf"), {})
        d = [oc[s] - rf[s] for s in sorted(set(oc) & set(rf))]
        m, h, _ = ci(d)
        pos = sum(x > 0 for x in d)
        st.setdefault("ocfno_gain", {})[t] = {"mean": m, "ci95": h, "n": len(d), "positive": pos}
        rows.append(f"{TNAME[t]} & {fmt(oc)} & {fmt(rf)} & ${m:+.3f}\\pm{h:.3f}$ & {pos}/{len(d)}" + r" \\")
    write("gain.tex",
          r"\begin{table}[t]" "\n" r"\centering" "\n"
          r"\caption{Order-conditioned FNO against the random forest at unseen speeds (Pb): macro-F1 (mean $\pm$ "
          r"95\% interval over seeds) and the paired same-seed difference.}" "\n" r"\label{tab:gain}" "\n"
          r"\resizebox{\columnwidth}{!}{%" "\n" r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
          r"Task & OC-FNO & RF & OC-FNO $-$ RF & Seeds $>0$ \\" "\n" r"\midrule" "\n" + "\n".join(rows) + "\n"
          r"\bottomrule" "\n" r"\end{tabular}}" "\n" r"\end{table}" "\n")
    rows = []
    for name, key in (("Full OC-FNO", None), ("without FiLM speed conditioning", "ocfno-nofilm"),
                      ("without order channels", "ocfno-noprior"), ("without gain augmentation", "ocfno-noaug")):
        cells = []
        for t in ("all", "comfaulda7"):
            v = (vals.get((t, "speed", "ocfno"), {}).get(0) if key is None
                 else abl.get("0", {}).get(t, {}).get("results", {}).get(f"speed/{key}", {}).get("macro_f1"))
            st.setdefault("ablation", {}).setdefault(t, {})[key or "full"] = v
            cells.append("--" if v is None else f"{v:.3f}")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    write("ablation.tex",
          r"\begin{table}[t]" "\n" r"\centering" "\n"
          r"\caption{Ablation of the order-conditioned FNO at unseen speeds (macro-F1, seed 0; -- not run).}" "\n"
          r"\label{tab:ablation}" "\n" r"\footnotesize" "\n" r"\begin{tabular}{lcc}" "\n" r"\toprule" "\n"
          r"Variant & T4 MAFAULDA & C1 COMFAULDA \\" "\n" r"\midrule" "\n" + "\n".join(rows) + "\n"
          r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")


def table_composition(c0, st):
    comp = c0["composition"]
    rows = []
    for m, v in comp["results"].items():
        f = v["per_label_f1"]
        rows.append(f"{MNAME[m]} & {v['exact_match']:.3f} & {f['unbalance']:.2f} & {f['horizontal']:.2f} & "
                    f"{f['vertical']:.2f}" + r" \\")
    st["composition"] = comp
    write("composition.tex",
          r"\begin{table}[t]" "\n" r"\centering" "\n"
          rf"\caption{{Combined faults recognised from single-fault training on COMFAULDA: one detector per fault "
          rf"type trained on {comp['n_train']} normal and single-fault runs, tested on {comp['n_test']} "
          r"combined-fault runs never seen. Exact match: all three labels right.}" "\n" r"\label{tab:composition}"
          "\n" r"\resizebox{\columnwidth}{!}{%" "\n" r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
          r"Model & Exact match & F1 unbalance & F1 horizontal & F1 vertical \\" "\n" r"\midrule" "\n"
          + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}}" "\n" r"\end{table}" "\n")


def figures(vals, bands):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    models = ("rf", "mlp", "kan", "fno")
    tasks = TASKS + ("comfaulda7",)
    fig, axs = plt.subplots(1, len(tasks), figsize=(7.2, 2.1), sharey=True)
    for ax, t in zip(axs, tasks, strict=True):
        x = np.arange(len(models))
        for k, (p, col, lab) in enumerate((("folds", "0.6", "seen speeds (Pa)"), ("speed", "#c0392b",
                                                                                  "unseen speeds (Pb)"))):
            mh = [ci(list(vals.get((t, p, m), {}).values())) for m in models]
            ax.bar(x + (k - 0.5) * 0.38, [a[0] for a in mh], 0.38, color=col, label=lab, capsize=1.5,
                   yerr=[0 if np.isnan(a[1]) else a[1] for a in mh], error_kw={"lw": 0.6})
        ax.set_xticks(x, [MNAME[m] for m in models], rotation=45, ha="right")
        ax.set_title(TNAME[t], fontsize=7)
        ax.set_ylim(0.5, 1.0)
    axs[0].set_ylabel("macro-F1")
    axs[0].legend(fontsize=5.5, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_protocols.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True)
    for ax, t in zip(axs, ("unbalance", "all", "comfaulda7"), strict=True):
        for k, m in enumerate(("rf", "fno", "ocfno")):
            b = bands.get((t, m))
            if not b:
                continue
            B = np.array(list(b.values()))
            ax.errorbar(np.arange(3) + (k - 1) * 0.08, B.mean(0), yerr=B.std(0, ddof=1) if len(B) > 1 else None,
                        marker="o", ms=3, capsize=2, lw=0.8, label=f"{MNAME[m]} ({len(B)} seeds)")
        ax.set_xticks(range(3), ["low", "mid", "high"])
        ax.set_xlabel("held-out speed tercile")
        ax.set_title(TNAME[t], fontsize=7)
        ax.legend(fontsize=5.5, frameon=False)
    axs[0].set_ylabel("macro-F1")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_speed_bands.pdf")
    plt.close(fig)
    src = ROOT / "examples" / "mafaulda_unbalance_pilot" / "_out" / "level_unbalance.png"
    if src.exists():
        shutil.copy(src, OUT / "figures" / "fig_level_unbalance.png")


def main():
    vals, bands, c0, abl = collect()
    st: dict = {"protocol": {f"{t}/{p}/{m}": dict(zip(("mean", "ci95", "sd"), ci(list(v.values())), strict=True),
                                                  n_seeds=len(v)) for (t, p, m), v in sorted(vals.items())}}
    table_protocol(vals, "folds", ("rf", "mlp", "kan", "fno", "hnn_rf"), "tab:seen",
                   r"Macro-F1 when the test speeds are seen in training (Pa: stratified 5-fold by run). Mean $\pm$ "
                   r"95\% interval over seeds (RF, MLP, KAN: 5; FNO: 3 on MAFAULDA); single values: one seed.")
    table_protocol(vals, "speed", ("rf", "mlp", "kan", "fno", "hnn_rf", "ocfno"), "tab:unseen",
                   r"Macro-F1 at unseen rotation speeds (Pb: each speed tercile held out in turn). Mean $\pm$ "
                   r"95\% interval over seeds; single values: one seed.")
    table_gain(vals, abl, st)
    table_composition(c0, st)
    figures(vals, bands)
    (OUT / "stats.json").write_text(json.dumps(st, indent=1, default=float))
    print("tables:", sorted(p.name for p in (OUT / "tables").iterdir()))
    print("figures:", sorted(p.name for p in (OUT / "figures").iterdir()))


if __name__ == "__main__":
    main()
