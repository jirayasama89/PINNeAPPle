"""Stream MAFAULDA from the published full.zip (HTTP range requests, no full download) and cache, per run:

* the repository's features (``rotating.mafaulda.features`` with spectra=True, orders 1-4): speed, tach quality,
  order phasors (tach-referenced), RMS/kurtosis/skewness/crest/peak, amplitude and envelope order spectra 0-12X;
* extra representations needed by the Q1 programme (computed from the same record, nothing from labels):
  - ``env_bands``: envelope order spectra (0-12X) in four carrier bands 0.5-2, 2-5, 5-10, 10-20 kHz;
  - ``angle_sig``: the six accelerometers order-tracked at 64 samples / revolution (float16), for angle-domain CNNs;
  - ``wave_1s``: the first second of the six accelerometers at 50 kHz (float16), for impulse / wavelet / cyclostationary
    branches;
  - ``sha1``: hash of the raw CSV bytes (duplicate audit).

One output file per run (``<cache>/<folder path with _>__<speed>.npz``), so an interrupted pass resumes.
The raw CSV is deleted as soon as its features are written.

usage: SSL_CERT_FILE=/root/.ccr/ca-bundle.crt python extract_mafaulda.py <cache_dir> [workers]
"""
from __future__ import annotations

import hashlib
import importlib
import io
import os
import sys
import types
import zipfile
from multiprocessing import get_context
from pathlib import Path

import numpy as np

URL = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/full.zip"
ROOT = Path(__file__).resolve().parents[2]
ROT = ROOT / "pinneapple_analysis" / "inspection" / "rotating"
sys.path.insert(0, "/home/user/tools")
BANDS = ((500.0, 2000.0), (2000.0, 5000.0), (5000.0, 10000.0), (10000.0, 20000.0))


def rotating():
    """Import the rotating sub-package without the heavy pinneapple_analysis/__init__ (which needs torch)."""
    if "rot" not in sys.modules:
        pkg = types.ModuleType("rot")
        pkg.__path__ = [str(ROT)]
        sys.modules["rot"] = pkg
    return importlib.import_module("rot.mafaulda"), importlib.import_module("rot.orders")


def out_name(member: str) -> str:
    p = Path(member)
    return "__".join(["_".join(p.parts[:-1]).replace(".", "p"), p.stem.replace(".", "p")]) + ".npz"


_ZF = None


def _zip():
    """One open archive (central directory read once) per worker process."""
    global _ZF
    if _ZF is None:
        from remote_zip import HttpRangeFile
        _ZF = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(URL), buffer_size=1 << 20))
    return _ZF


def work(args):
    member, cache = args
    out = Path(cache) / out_name(member)
    if out.exists():
        return member, "cached"
    global _ZF
    maf, orders = rotating()
    raw, last = None, None
    for attempt in range(5):                      # network resets through the proxy are retried, never cached
        try:
            raw = _zip().read(member)
            break
        except Exception as e:  # noqa: BLE001
            _ZF = None
            import time
            time.sleep(3 * (attempt + 1))
            last = e
    if raw is None:
        return member, f"network error, not cached: {type(last).__name__}"
    try:
        tmp = Path(cache) / ("tmp_" + out_name(member).replace(".npz", ".csv"))
        tmp.write_bytes(raw)
        parts = Path(member).parts
        level = parts[-2] if len(parts) > 2 else ""
        m = maf._LEVEL.match(level) if level else None
        state = "/".join(parts[:-2] if m else parts[:-1])
        run = maf.Run(str(tmp), state, level if m else "", float(m.group(1)) if m else 0.0, m.group(2) if m else "",
                      float(Path(member).stem))
        f = maf.features(run, orders=(1, 2, 3, 4), spectra=True)
        X = maf.read(tmp)
        ang = orders.shaft_angle(X[:, 0], maf.FS)
        env = [orders.order_spectrum(X[:, 1:], ang, maf.FS, envelope_band=b)[1].astype(np.float16) for b in BANDS]
        f["env_bands"] = np.stack(env)
        f["angle_sig"] = orders.angle_resample(X[:, 1:7], ang, maf.FS, 64).astype(np.float16)
        f["wave_1s"] = X[: int(maf.FS), 1:7].astype(np.float16)
        f["sha1"] = hashlib.sha1(raw).hexdigest()
        f["member"] = member
        f["path"] = member
        tmp.unlink()
        np.savez_compressed(out, row=np.array([f], dtype=object))
        return member, "ok"
    except Exception as e:  # reported, not fatal
        try:
            tmp.unlink()
        except Exception:
            pass
        np.savez_compressed(out, row=np.array([{"member": member, "path": member, "error": f"{type(e).__name__}: {e}"}],
                                              dtype=object))
        return member, f"error {type(e).__name__}: {e}"


def main():
    cache = Path(sys.argv[1])
    cache.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    from remote_zip import HttpRangeFile
    zf = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(URL), buffer_size=1 << 20))
    members = sorted(i.filename for i in zf.infolist() if i.filename.endswith(".csv"))
    todo = [(m, str(cache)) for m in members if not (cache / out_name(m)).exists()]
    print(f"{len(members)} runs, {len(todo)} to do", flush=True)
    with get_context("fork").Pool(workers) as pool:
        for i, (m, st) in enumerate(pool.imap_unordered(work, todo, chunksize=1)):
            if st != "ok" or i % 50 == 0:
                print(i, m, st, flush=True)


if __name__ == "__main__":
    main()
