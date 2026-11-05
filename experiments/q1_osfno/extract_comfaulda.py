"""Stream COMFAULDA (Zenodo 8208454, 23 GB) inner archive by inner archive (HTTP range on the outer zip, one inner
zip of a few hundred MB on disk at a time, deleted after use), caching per run the repository's phase-free features
(``rotating.comfaulda.features``) plus extras: multi-band envelope order spectra, angle-domain signals at constant
refined speed (64 / rev), first second of the six accelerometers (float16), SHA-1 of the CSV bytes.

usage: python extract_comfaulda.py <cache_dir> [workers]
"""
from __future__ import annotations

import hashlib
import importlib
import io
import os
import re
import sys
import types
import zipfile
from multiprocessing import get_context
from pathlib import Path

import numpy as np

URL = "https://zenodo.org/api/records/8208454/files/COMFAULDA.zip/content"
ROOT = Path(__file__).resolve().parents[2]
ROT = ROOT / "pinneapple_analysis" / "inspection" / "rotating"
sys.path.insert(0, "/home/user/tools")
BANDS = ((500.0, 2000.0), (2000.0, 5000.0), (5000.0, 10000.0), (10000.0, 20000.0))


def rotating():
    if "rot" not in sys.modules:
        pkg = types.ModuleType("rot")
        pkg.__path__ = [str(ROT)]
        sys.modules["rot"] = pkg
    return importlib.import_module("rot.comfaulda"), importlib.import_module("rot.orders")


def one(args):
    b, name, meta = args
    com, orders = rotating()
    try:
        f = com.features(b, name, meta)
        D = com.read_csv_bytes(b)
        X = D[:, list(com.ACCEL_COLUMNS)]
        fs = f["fs"]
        ang = com.constant_speed_angle(len(X), fs, f["speed_hz"])
        f["env_bands"] = np.stack([orders.order_spectrum(X, ang, fs, envelope_band=bd)[1].astype(np.float16) for bd in BANDS])
        f["angle_sig"] = orders.angle_resample(X, ang, fs, 64).astype(np.float16)
        f["wave_1s"] = X[: int(fs)].astype(np.float16)
        f["sha1"] = hashlib.sha1(b).hexdigest()
        return f
    except Exception as e:  # noqa: BLE001
        return {**meta, "path": f"{meta['inner']}/{name}", "error": f"{type(e).__name__}: {e}"}


def main():
    cache = Path(sys.argv[1])
    cache.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    from remote_zip import HttpRangeFile
    com, _ = rotating()
    outer = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(URL), buffer_size=1 << 22))
    inners = sorted(i.filename for i in outer.infolist() if i.filename.endswith(".zip"))
    print(len(inners), "inner archives", flush=True)
    for inner in inners:
        cf = cache / (re.sub(r"[^A-Za-z0-9]+", "_", inner) + ".npz")
        if cf.exists():
            continue
        for attempt in range(5):
            try:
                data = outer.read(inner)
                break
            except Exception as e:  # noqa: BLE001
                print("retry", inner, e, flush=True)
                outer = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(URL), buffer_size=1 << 22))
        zi = zipfile.ZipFile(io.BytesIO(data))
        meta = com.labels_of(inner)
        jobs, corrupt = [], []
        for n in sorted(zi.namelist()):
            if n.endswith(".csv"):
                try:
                    jobs.append((zi.read(n), os.path.basename(n), meta))
                except zipfile.BadZipFile:
                    corrupt.append(n)
        with get_context("fork").Pool(workers) as pool:
            rows = pool.map(one, jobs)
        rows += [{**meta, "path": f"{inner}/{n}", "error": "BadZipFile: CRC error in the published archive"} for n in corrupt]
        np.savez_compressed(cf, rows=np.array(rows, dtype=object))
        del data, zi, jobs
        print(inner, len(rows), "runs", sum("error" in r for r in rows), "errors", flush=True)


if __name__ == "__main__":
    main()
