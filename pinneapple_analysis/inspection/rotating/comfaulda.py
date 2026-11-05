"""COMFAULDA (Composed Fault Dataset; Zenodo 8208454, IEEE DataPort 10.21227/89ye-ap56, CC BY 4.0): read the runs
straight from the nested archives and compute phase-free order features.

What the files are (read from the archive, not only from its readme): 48 inner archives, one per scenario and level
(``Normal_Condition/Normal_k.zip``, ``Single Faults/Unbalance/Unbalance_6_g.zip``, ``Combined Faults/...``), one CSV
per speed named by its rotation frequency in Hz; nine columns (time, then eight); 51.2 kHz; normal runs separated by
``;``, fault runs by ``, ``. No column is a pulse tachometer and column 8 is near zero, so the speed comes from the
file name, refined by the largest 1X line within +-5 % on the six accelerometers (columns 2-7), and the angle is
taken at that constant speed; with no phase reference only phase-free quantities are computed.
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .mafaulda import ENVELOPE_BAND
from .orders import ShaftAngle, order_spectrum

ACCEL_COLUMNS = (2, 3, 4, 5, 6, 7)        # capacitive X Z Y, piezoelectric X Z Y (readme order)
FAULTS = ("unbalance", "horizontal", "vertical")


def labels_of(inner: str) -> dict[str, Any]:
    """State and levels of an inner archive name."""
    n = inner.lower()
    has = {"unbalance": "unbalance" in n, "horizontal": "hor" in n, "vertical": "ver" in n}
    if "normal" in n:
        has = dict.fromkeys(FAULTS, False)
    kinds = [k for k in FAULTS if has[k]]
    state = "normal" if not kinds else "+".join(kinds)
    num = [float(x.replace("_", ".")) for x in re.findall(r"(\d+(?:[._]\d+)?)\s*_?(?:g|mm)", n)]
    return {"state": state, "labels": has, "combined": len(kinds) > 1, "levels": num, "inner": inner}


def read_csv_bytes(b: bytes) -> np.ndarray:
    import pandas as pd
    sep = ";" if ";" in b[:400].decode("latin-1") else ","
    return pd.read_csv(io.BytesIO(b), header=None, sep=sep, skipinitialspace=True, engine="c").to_numpy(float)


def constant_speed_angle(n: int, fs: float, speed_hz: float) -> ShaftAngle:
    revs = int(np.floor(n / fs * speed_hz)) - 1
    last = int(np.floor((revs + 1) / speed_hz * fs))
    t = np.arange(last) / fs
    return ShaftAngle(theta=2 * np.pi * speed_hz * t, first=0, last=last, pulse_times_s=np.arange(revs + 1) / speed_hz,
                      speed_hz=float(speed_hz), speed_cv=0.0, revolutions=revs)


def refine_speed(X: np.ndarray, fs: float, nominal: float, tol: float = 0.05) -> float:
    x = X - X.mean(0)
    S = np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)).sum(1)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    band = (f >= nominal * (1 - tol)) & (f <= nominal * (1 + tol))
    k = np.flatnonzero(band)[np.argmax(S[band])]
    if 0 < k < len(S) - 1:                                      # parabolic refinement of the peak
        a, b, c = np.log(S[k - 1:k + 2] + 1e-30)
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return float(k * (f[1] - f[0]))


def features(b: bytes, name: str, meta: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import kurtosis, skew
    D = read_csv_bytes(b)
    fs = 1.0 / np.median(np.diff(D[:200, 0]))
    X = D[:, list(ACCEL_COLUMNS)]
    nominal = float(re.match(r"\s*(\d+)[.,](\d+)", name).expand(r"\1.\2"))   # '36.40..csv', '58,80.csv' exist
    sp = refine_speed(X, fs, nominal)
    ang = constant_speed_angle(len(X), fs, sp)
    x = X - X.mean(0)
    peak = np.abs(x).max(0)
    _, S = order_spectrum(X, ang, fs)
    _, E = order_spectrum(X, ang, fs, envelope_band=ENVELOPE_BAND)
    return {**meta, "path": f"{meta['inner']}/{name}", "nominal_hz": nominal, "speed_hz": sp, "fs": fs,
            "tach_ok": True, "rms": x.std(0), "kurtosis": kurtosis(x, axis=0, fisher=False), "skewness": skew(x, axis=0),
            "peak": peak, "crest": peak / np.maximum(x.std(0), 1e-12), "order_spectrum": S.astype(np.float32),
            "envelope_spectrum": E.astype(np.float32)}


def _one(args):
    b, name, meta = args
    try:
        return features(b, name, meta)
    except Exception as e:                                       # one broken file does not stop the table
        return {**meta, "path": f"{meta['inner']}/{name}", "error": f"{type(e).__name__}: {e}"}


def feature_table(archive: str | Path, cache_dir: str | Path, workers: int = 4, budget_s: float | None = None,
                  log=print) -> list[dict[str, Any]]:
    """Features of every run, one cache file per inner archive (``cache_dir``), so an interrupted pass resumes;
    ``budget_s`` stops before opening a new inner archive once the time is spent."""
    import multiprocessing as mp
    import time
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(archive)
    inners = sorted(i.filename for i in z.infolist() if i.filename.endswith(".zip"))
    rows, t0 = [], time.perf_counter()
    for inner in inners:
        cf = cache_dir / (re.sub(r"[^A-Za-z0-9]+", "_", inner) + ".npz")
        if cf.exists():
            rows += np.load(cf, allow_pickle=True)["rows"].tolist()
            continue
        if budget_s is not None and time.perf_counter() - t0 > budget_s:
            log("budget spent")
            break
        zi = zipfile.ZipFile(io.BytesIO(z.read(inner)))
        meta = labels_of(inner)
        jobs, corrupt = [], []
        for n in sorted(zi.namelist()):
            if not n.endswith(".csv"):
                continue
            try:
                jobs.append((zi.read(n), os.path.basename(n), meta))
            except zipfile.BadZipFile:              # a CRC error inside the published archive: reported, skipped
                corrupt.append(n)
        with mp.get_context("fork").Pool(workers) as pool:
            new = pool.map(_one, jobs)
        new += [{**meta, "path": f"{inner}/{n}", "error": "BadZipFile: CRC error in the published archive"}
                for n in corrupt]
        np.savez(cf, rows=np.array(new, dtype=object))
        rows += new
        log(f"{inner}: {len(new)} runs, {sum('error' in r for r in new)} errors ({time.perf_counter() - t0:.0f} s)")
    return rows
