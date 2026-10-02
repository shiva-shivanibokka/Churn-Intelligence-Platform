"""Shared metric helpers for the SOP evaluation (eval_sop/).

Everything here is deterministic given a seed. Bootstrap CIs are percentile
intervals over resampled evaluation rows (rows of the held-out set), so they
capture test-set sampling noise only -- not training variance, which is what
the multi-seed runs are for.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

N_BOOT = 1000


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins on [0, 1], weighted by bin size."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def classification_metrics(y, p, base_rate: float) -> dict:
    """AUC, PR-AUC, Brier, Brier skill score vs a constant base-rate forecast, ECE."""
    y = np.asarray(y)
    p = np.asarray(p)
    brier = brier_score_loss(y, p)
    brier_ref = brier_score_loss(y, np.full_like(p, base_rate, dtype=float))
    return {
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier,
        "brier_ref_base_rate": brier_ref,
        "bss": 1.0 - brier / brier_ref,
        "ece": ece(y, p),
    }


def bootstrap_ci(stat_fn, n: int, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of stat_fn(idx) over n_boot resamples of range(n)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(stat_fn(idx))
        except ValueError:
            continue
    vals = np.asarray(vals, float)
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def metrics_with_ci(y, p, base_rate, seed=0, n_boot=N_BOOT) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    point = classification_metrics(y, p, base_rate)
    out = {}
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        m = classification_metrics(y[idx], p[idx], base_rate)
        for k, v in m.items():
            boots[k].append(v)
    for k, v in point.items():
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        out[k] = float(v)
        out[k + "_lo"] = float(lo)
        out[k + "_hi"] = float(hi)
    return out


# ── Uplift evaluation ─────────────────────────────────────────────────────────

def qini_curve(y, t, score):
    """Qini curve Q(k) = Y_t(k) - Y_c(k) * N_t(k)/N_c(k), customers sorted by score desc.

    Returns (fraction_targeted, Q) including the origin.
    """
    order = np.argsort(-np.asarray(score), kind="mergesort")
    y = np.asarray(y, float)[order]
    t = np.asarray(t, int)[order]
    nt = np.cumsum(t)
    nc = np.cumsum(1 - t)
    yt = np.cumsum(y * t)
    yc = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = yt - np.where(nc > 0, yc * nt / nc, 0.0)
    frac = np.arange(1, len(y) + 1) / len(y)
    return np.concatenate([[0.0], frac]), np.concatenate([[0.0], q])


def qini_coefficient(y, t, score) -> float:
    """Area between the Qini curve and the random-targeting line, divided by n.

    Units: incremental positive outcomes per customer in the evaluation set,
    averaged over all targeting depths. 0 = random targeting.
    """
    x, q = qini_curve(y, t, score)
    n = len(y)
    random_line = x * q[-1]
    return float(np.trapezoid(q - random_line, x) / n)


def uplift_curve_auuc(y, t, score) -> float:
    """Area under the uplift curve U(k) = (ybar_t(k) - ybar_c(k)) * k/n, minus random.

    Units: average treatment-effect-weighted share; 0 = random targeting.
    """
    order = np.argsort(-np.asarray(score), kind="mergesort")
    y = np.asarray(y, float)[order]
    t = np.asarray(t, int)[order]
    nt = np.cumsum(t)
    nc = np.cumsum(1 - t)
    yt = np.cumsum(y * t)
    yc = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where((nt > 0) & (nc > 0), yt / np.maximum(nt, 1) - yc / np.maximum(nc, 1), 0.0)
    frac = np.arange(1, len(y) + 1) / len(y)
    curve = u * frac
    rand = frac * curve[-1]
    return float(np.trapezoid(np.concatenate([[0], curve - rand]), np.concatenate([[0], frac])))


def uplift_at_k(y, t, score, k: float) -> float:
    """Observed ATE (treated mean - control mean) inside the top-k fraction by score."""
    order = np.argsort(-np.asarray(score), kind="mergesort")
    m = order[: max(1, int(round(k * len(order))))]
    y = np.asarray(y, float)[m]
    t = np.asarray(t, int)[m]
    return float(y[t == 1].mean() - y[t == 0].mean())


def uplift_by_decile(y, t, score, n_bins=10):
    order = np.argsort(-np.asarray(score), kind="mergesort")
    y = np.asarray(y, float)[order]
    t = np.asarray(t, int)[order]
    s = np.asarray(score, float)[order]
    rows = []
    for d, chunk in enumerate(np.array_split(np.arange(len(y)), n_bins)):
        yy, tt = y[chunk], t[chunk]
        rows.append({
            "decile": d + 1,
            "n": int(len(chunk)),
            "n_treated": int(tt.sum()),
            "n_control": int((1 - tt).sum()),
            "mean_pred_uplift": float(s[chunk].mean()),
            "treated_rate": float(yy[tt == 1].mean()),
            "control_rate": float(yy[tt == 0].mean()),
            "observed_uplift": float(yy[tt == 1].mean() - yy[tt == 0].mean()),
        })
    return rows


def fmt_ci(d, k, nd=3):
    return f"{d[k]:.{nd}f} [{d[k + '_lo']:.{nd}f}, {d[k + '_hi']:.{nd}f}]"
