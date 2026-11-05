"""Run a queue of (dataset, task, model, seeds) jobs sequentially, skipping finished result files.

usage: python jobqueue.py <queue_name>
"""
import sys
import time

import models as M
import models2 as M2
import q1lib as L

M.REGISTRY.update(M2.EXTRA)
MAF = ["misalignment", "unbalance", "bearings", "all"]
SEEDS = [0, 1, 2, 3, 4]
QUEUES = {
    # fast feature baselines first (per-run predictions on the locked partitions)
    "features_maf": [("mafaulda", t, m, SEEDS) for m in ("rf", "mlp", "kan", "svm", "xgb") for t in MAF],
    "osfno_core_maf": [("mafaulda", t, m, SEEDS) for m in ("osfno-B-head", "osfno-C-film", "osfno-A-nospeed") for t in MAF],
    # remote worker shards (cloud sessions, slim cache)
    "R1": [("mafaulda", t, "osfno-C-film", SEEDS) for t in MAF],
    "R2": [("mafaulda", t, "osfno-A-nospeed", SEEDS) for t in MAF],
    "R3": [("mafaulda", t, "osfno-E-phys", SEEDS) for t in MAF]
          + [("mafaulda", "all", m, SEEDS) for m in ("osfno-B-noaug", "osfno-B-noprior", "osfno-B-commonaug")],
    "R4": [("mafaulda", t, m, SEEDS) for m in ("cnn-spec", "fno-trainonly") for t in MAF],
    "R5": [("mafaulda", t, m, SEEDS) for m in ("osfno-B-dro", "osfno-B-supcon") for t in MAF],
    "R6": [("mafaulda", t, "osfno-B-head", SEEDS) for t in ("bearings", "all")],
    "R7": [("comfaulda", "comfaulda7", m, SEEDS) for m in ("rf", "mlp", "kan", "svm", "xgb", "osfno-B-head",
                                                          "osfno-C-film")],
    "R8": [("comfaulda", "comfaulda7", m, SEEDS) for m in ("osfno-A-nospeed", "osfno-E-phys", "cnn-spec")],
    "local_B_rest": [("mafaulda", t, "osfno-B-head", SEEDS) for t in ("bearings", "all")],
    "features": [("mafaulda", t, m, SEEDS) for m in ("rf", "mlp", "kan", "svm", "xgb") for t in MAF]
                + [("comfaulda", "comfaulda7", m, SEEDS) for m in ("rf", "mlp", "kan", "svm", "xgb")],
    # Phase 1/3 core: speed handling in the spectral operator
    "osfno_core": [(d, t, m, SEEDS) for m in ("osfno-B-head", "osfno-C-film", "osfno-A-nospeed")
                   for d, t in [("mafaulda", x) for x in MAF] + [("comfaulda", "comfaulda7")]],
    "osfno_more": [(d, t, m, SEEDS) for m in ("osfno-E-phys", "osfno-B-noaug", "osfno-B-noprior", "osfno-B-commonaug")
                   for d, t in [("mafaulda", "all"), ("comfaulda", "comfaulda7")]],
    "deep_baselines": [(d, t, m, SEEDS) for m in ("cnn-spec", "fno-trainonly", "cnn-angle")
                       for d, t in [("mafaulda", x) for x in MAF] + [("comfaulda", "comfaulda7")]],
    "dg": [(d, t, m, SEEDS) for m in ("osfno-B-dro", "osfno-B-supcon", "fusion-rf-osfno")
           for d, t in [("mafaulda", x) for x in MAF] + [("comfaulda", "comfaulda7")]],
}

if __name__ == "__main__":
    q = QUEUES[sys.argv[1]]
    cache = {}
    for ds, task, model, seeds in q:
        if ds not in cache:
            cache = {ds: L.load_rows(ds)}
        for s in seeds:
            t0 = time.time()
            L.run(task, model, M.REGISTRY[model], cache[ds], s, exp="phase1", dataset=ds, config={"model": model})
            sys.stdout.flush()
