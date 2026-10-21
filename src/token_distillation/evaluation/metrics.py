"""Comparable three-class direction metrics with an explicit failure denominator."""
from __future__ import annotations

import numpy as np


def direction_metrics(probabilities, targets, valid_mask=None) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or targets.shape != probabilities.shape[:1]:
        raise ValueError("Expected probabilities [samples,3] and targets [samples]")
    mask = np.ones(len(targets), dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    if mask.dtype != np.bool_ or mask.shape != targets.shape:
        raise ValueError("valid_mask must be boolean [samples]")
    p, y = probabilities[mask], targets[mask]
    if y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y > 2):
        raise ValueError("Valid targets must be integer class IDs 0..2")
    if (not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-6)):
        raise ValueError("Valid probabilities must be finite normalized three-class distributions")
    result = {"requested": len(mask), "valid": int(mask.sum()), "excluded": int((~mask).sum()),
              "valid_fraction": float(mask.mean()) if len(mask) else None,
              "log_loss": None, "brier": None,
              "brier_definition": "mean_sum_squared_error_over_three_classes", "qualified": False}
    if len(y):
        truth = np.eye(3)[y]
        result.update(log_loss=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.)).mean()),
                      brier=float(np.square(p - truth).sum(axis=1).mean()))
    return result
