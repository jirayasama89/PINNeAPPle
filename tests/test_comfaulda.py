"""COMFAULDA reader: labels from the inner archive names, file names as published (including the malformed ones),
and the speed refinement on a synthetic run."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.inspection.rotating.comfaulda import labels_of, refine_speed


@pytest.mark.parametrize("name, state, combined", [
    ("Normal_Condition/Normal_1.zip", "normal", False),
    ("Single Faults/Unbalance/Unbalance_6_g.zip", "unbalance", False),
    ("Single Faults/Horizontal Misalignment/Horizontal_misalignment_0_5mm.zip", "horizontal", False),
    ("Combined Faults/Unbalance+Vertical_Misalignment/Unbalance_20_g+Ver. Misalignment_0.51_mm.zip",
     "unbalance+vertical", True),
    ("Combined Faults/Vertical_misalignment+Horizontal_misalignment/Ver. Misalignment_1.91_mm+Hor. Misalignment_1_mm..zip",
     "horizontal+vertical", True),
])
def test_labels(name, state, combined):
    lab = labels_of(name)
    assert (lab["state"], lab["combined"]) == (state, combined)


def test_speed_refinement_finds_the_1x_line():
    fs, f = 51200.0, 33.14 * 0.989
    t = np.arange(int(4.88 * fs)) / fs
    X = np.stack([np.sin(2 * np.pi * f * t + k) + 0.3 * np.sin(4 * np.pi * f * t) for k in range(6)], 1)
    assert refine_speed(X, fs, 33.14) == pytest.approx(f, abs=0.02)
