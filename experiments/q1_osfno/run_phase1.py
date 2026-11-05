"""Phase 1: confirmation runs on the locked partitions, 5 seeds, predictions saved.

usage: python run_phase1.py <dataset> <task> <model> [seeds]
  dataset: mafaulda | comfaulda ; task: misalignment | unbalance | bearings | all | comfaulda7
  model: a key of models.REGISTRY
"""
import sys

import models as M
import q1lib as L

if __name__ == "__main__":
    dataset, task, model = sys.argv[1:4]
    seeds = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2, 3, 4]
    rows = L.load_rows(dataset)
    for s in seeds:
        L.run(task, model, M.REGISTRY[model], rows, s, exp="phase1", dataset=dataset, config={"model": model})
