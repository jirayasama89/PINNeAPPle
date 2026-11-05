"""MAFAULDA (Machinery Fault Database, SMT / COPPE / UFRJ): file index and order features.

The database (https://www02.smt.ufrj.br/~offshore/mfs/page_01.html) holds 5 s records at 50 kHz from a SpectraQuest
Machinery Fault Simulator: one CSV per run, eight columns -- tachometer; underhang bearing accelerometer (axial,
radial, tangential); overhang bearing accelerometer (axial, radial, tangential); microphone. Folders name the state
and its level (``normal/``, ``imbalance/6g/``, ``horizontal-misalignment/0.5mm/``, ``underhang/outer_race/20g/``,
...) and each file name is the nominal rotation frequency in Hz. The bench data published with it (bearing orders
BPFO 2.998, BPFI 5.002, BSF 1.871, FTF 0.375 per revolution) are in :data:`BEARING_ORDERS`.

:func:`index` lists the files of an extracted folder; :func:`features` reads one run and returns its speed and the
order phasors of the seven vibration and sound channels; :func:`feature_table` does it for many runs in parallel and
caches the result (the raw CSVs can then be deleted).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .orders import order_phasors, order_resolution, order_spectrum, shaft_angle

FS = 50_000.0
COLUMNS = ("tachometer", "ub_axial", "ub_radial", "ub_tangential", "ob_axial", "ob_radial", "ob_tangential",
           "microphone")
CHANNELS = COLUMNS[1:]
#: published by the database page (per revolution of the shaft)
BEARING_ORDERS = {"FTF": 0.3750, "BPFO": 2.9980, "BPFI": 5.0020, "BSF": 1.8710}
ORDERS = (1, 2, 3)


@dataclass(frozen=True)
class Run:
    path: str
    state: str            # normal, imbalance, horizontal-misalignment, ..., underhang/outer_race, ...
    level: str            # "" (normal), "6g", "0.5mm", ...
    level_value: float    # the number in ``level`` (0 for normal)
    level_unit: str       # "g", "mm" or ""
    nominal_hz: float     # from the file name


_LEVEL = re.compile(r"^([0-9.]+)\s*(g|mm)$")


def index(root: str | Path) -> list[Run]:
    """Every CSV under ``root``, with its state, level and nominal speed (from the folder and file names)."""
    root = Path(root)
    out = []
    for p in sorted(root.rglob("*.csv")):
        parts = p.relative_to(root).parts[:-1]
        level, value, unit = "", 0.0, ""
        if parts and _LEVEL.match(parts[-1]):
            level = parts[-1]
            m = _LEVEL.match(level)
            value, unit = float(m.group(1)), m.group(2)
            parts = parts[:-1]
        out.append(Run(str(p), "/".join(parts), level, value, unit, float(p.stem)))
    return out


def read(path: str | Path) -> np.ndarray:
    """One run as an array (250 000, 8), columns in COLUMNS order."""
    import pandas as pd
    return pd.read_csv(path, header=None, dtype=np.float64, engine="c").to_numpy()


#: band of the envelope spectra (Hz): above the shaft harmonics, inside the accelerometers' flat range
ENVELOPE_BAND = (1000.0, 10000.0)


def snippet(X: np.ndarray, factor: int = 10, seconds: float = 1.0) -> np.ndarray:
    """The six accelerometers, low-passed and decimated by ``factor`` (5 kHz from 50 kHz), first ``seconds`` (float32):
    enough for models that need the waveform (an HNN on the measured trajectories) once the raw CSVs are deleted."""
    from scipy.signal import decimate
    y = decimate(X[:, 1:7], factor, axis=0, zero_phase=True)
    return y[: int(seconds * FS / factor)].astype(np.float32)


def features(run: Run, orders=ORDERS, spectra: bool = False, waveform: bool = False) -> dict[str, Any]:
    """Speed from the tachometer and the order phasors (orders x channels, complex) of one run, with the record's
    order resolution and the RMS of every channel; ``spectra``: also the amplitude and envelope order spectra of the
    seven channels on a grid of 0.01 orders up to 12 (float32)."""
    X = read(run.path)
    ang = shaft_angle(X[:, 0], FS)
    out = {"path": run.path, "state": run.state, "level": run.level, "level_value": run.level_value,
           "level_unit": run.level_unit, "nominal_hz": run.nominal_hz, "speed_hz": ang.speed_hz,
           "speed_cv": ang.speed_cv, "tach_ok": ang.reliable and 0.9 < ang.speed_hz / run.nominal_hz < 1.05,
           "revolutions": ang.revolutions, "order_resolution": order_resolution(ang), "phasors": order_phasors(X[:, 1:], ang, orders), "rms": X[:, 1:].std(0)}
    from scipy.stats import kurtosis, skew
    x = X[:, 1:] - X[:, 1:].mean(0)
    peak = np.abs(x).max(0)
    out.update(kurtosis=kurtosis(x, axis=0, fisher=False), skewness=skew(x, axis=0), peak=peak,
               crest=peak / np.maximum(x.std(0), 1e-12))
    if waveform:
        out["snippet"] = snippet(X)
    if spectra:
        _, out["order_spectrum"] = order_spectrum(X[:, 1:], ang, FS)
        _, out["envelope_spectrum"] = order_spectrum(X[:, 1:], ang, FS, envelope_band=ENVELOPE_BAND)
        out["order_spectrum"] = out["order_spectrum"].astype(np.float32)
        out["envelope_spectrum"] = out["envelope_spectrum"].astype(np.float32)
    return out


def _one(args):
    run, orders, spectra, waveform = args
    try:
        return features(run, orders, spectra, waveform)
    except Exception as e:                       # a broken file is reported, not fatal to the table
        return {"path": run.path, "error": f"{type(e).__name__}: {e}"}


def feature_table(runs: list[Run], cache: str | Path | None = None, orders=ORDERS, workers: int = 4,
                  spectra: bool = False, waveform: bool = False) -> list[dict[str, Any]]:
    """:func:`features` of every run (``workers`` processes), cached in ``cache`` (.npz) when given; runs already in
    the cache are not read again."""
    done: dict[str, dict] = {}
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        done = {r["path"]: r for r in z["rows"].tolist()}
    todo = [r for r in runs if r.path not in done]
    if todo:
        if workers > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers) as pool:
                new = pool.map(_one, [(r, orders, spectra, waveform) for r in todo])
        else:
            new = [_one((r, orders, spectra, waveform)) for r in todo]
        done.update({r["path"]: r for r in new})
        if cache:
            np.savez(cache, rows=np.array(list(done.values()), dtype=object))
    return [done[r.path] for r in runs]
