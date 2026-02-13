#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    median: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> BootstrapResult:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return BootstrapResult(np.nan, np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots[i] = float(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    return BootstrapResult(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        ci_low=float(np.quantile(boots, alpha)),
        ci_high=float(np.quantile(boots, 1.0 - alpha)),
        n=int(arr.size),
    )


def fisher_exact(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Return odds ratio and two-sided p-value for a 2x2 table."""
    try:
        from scipy.stats import fisher_exact as scipy_fisher_exact

        odds_ratio, pvalue = scipy_fisher_exact([[a, b], [c, d]], alternative="two-sided")
        return float(odds_ratio), float(pvalue)
    except Exception:
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        pvalue = 1.0
        return float(odds_ratio), float(pvalue)


def multiple_testing_correction(pvalues: Iterable[float], method: str = "bh") -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    if method.lower() in {"bh", "fdr_bh", "benjamini-hochberg"}:
        q = ranked * n / (np.arange(n) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
    elif method.lower() == "holm":
        q = (n - np.arange(n)) * ranked
        q = np.maximum.accumulate(q)
    else:
        raise ValueError(f"Unknown correction method: {method}")
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out
