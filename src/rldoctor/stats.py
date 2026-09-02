"""Robust trend statistics used by the detectors.

RL training curves are noisy, heavy-tailed and often have a handful of wild
outliers (a single batch of degenerate rollouts). Ordinary least squares and
plain means over-react to those, which is exactly how a diagnostic tool earns a
reputation for crying wolf.  Everything here is therefore rank- or median-based.

No SciPy dependency: the only distributional result needed is the standard
normal CDF, which :func:`math.erf` gives exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_RNG = np.random.default_rng(0)


def fmt_p(p: float) -> str:
    """Format a p-value honestly.

    Returns the relational operator too, so call sites read ``f"p{fmt_p(x)}"``.
    The normal approximation underflows to exactly 0.0 for strong trends, and
    printing "p=0" in a report aimed at researchers is a fast way to lose them.
    """
    if p <= 0.0:
        return "<1e-16"
    if p < 1e-4:
        return f"={p:.1e}"
    return f"={p:.3g}"


def normal_cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class Trend:
    """Result of a robust trend test over a series."""

    slope: float  #: Theil-Sen slope, units of y per unit of x
    intercept: float
    p_value: float  #: Mann-Kendall two-sided p-value
    n: int
    tau: float  #: Kendall's tau-b, a scale-free effect size in [-1, 1]

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05 and self.n >= 8

    @property
    def direction(self) -> str:
        if not self.significant:
            return "flat"
        return "up" if self.slope > 0 else "down"

    def predict(self, x: float) -> float:
        return self.intercept + self.slope * x


def theil_sen(x: np.ndarray, y: np.ndarray, max_pairs: int = 40_000) -> Tuple[float, float]:
    """Median-of-pairwise-slopes regression.

    Returns ``(slope, intercept)``.  Breakdown point ~29%, so a handful of
    catastrophic steps cannot flip the reported direction of a trend.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        return 0.0, float(y[0]) if n else 0.0

    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        # Subsample pairs; the median slope estimate is stable well below the
        # full pair set and this keeps long runs interactive.
        i = _RNG.integers(0, n, size=max_pairs)
        j = _RNG.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]

    dx = x[j] - x[i]
    valid = dx != 0
    if not np.any(valid):
        return 0.0, float(np.median(y))
    slopes = (y[j][valid] - y[i][valid]) / dx[valid]
    slope = float(np.median(slopes))
    intercept = float(np.median(y) - slope * np.median(x))
    return slope, intercept


def mann_kendall(y: np.ndarray) -> Tuple[float, float]:
    """Mann-Kendall trend test with tie correction.

    Returns ``(p_value, tau_b)``.  Non-parametric, so it makes no assumption
    that reward curves are normal or homoscedastic -- both of which are false.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 4:
        return 1.0, 0.0

    # S = sum of sign(y_j - y_i) for i < j, computed pairwise.
    diff = np.sign(y[None, :] - y[:, None])
    s = float(np.sum(np.triu(diff, k=1)))

    # Variance with correction for tied groups.
    _, counts = np.unique(y, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return 1.0, 0.0

    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * (1.0 - normal_cdf(abs(z)))

    # tau-b denominator accounts for ties on both sides.
    n0 = n * (n - 1) / 2.0
    n1 = float(np.sum(counts * (counts - 1) / 2.0))
    denom = math.sqrt(max(n0 - n1, 1e-12) * n0)
    tau = s / denom if denom > 0 else 0.0
    return float(min(max(p, 0.0), 1.0)), float(max(min(tau, 1.0), -1.0))


def trend(x: np.ndarray, y: np.ndarray) -> Trend:
    """Combined Theil-Sen slope + Mann-Kendall significance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return Trend(0.0, float(y[0]) if y.size else 0.0, 1.0, int(x.size), 0.0)
    slope, intercept = theil_sen(x, y)
    p, tau = mann_kendall(y)
    return Trend(slope, intercept, p, int(x.size), tau)


def mad(y: np.ndarray) -> float:
    """Median absolute deviation, scaled to be a consistent sigma estimator."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(y - np.median(y))))


def robust_z(y: np.ndarray) -> np.ndarray:
    """Per-point robust z-scores using median/MAD."""
    y = np.asarray(y, dtype=float)
    scale = mad(y)
    if scale <= 0:
        return np.zeros_like(y)
    return (y - np.median(y)) / scale


def ewma(y: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Exponentially weighted moving average, nan-tolerant."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    acc: Optional[float] = None
    for idx, value in enumerate(y):
        if not np.isfinite(value):
            out[idx] = acc if acc is not None else np.nan
            continue
        acc = value if acc is None else alpha * value + (1 - alpha) * acc
        out[idx] = acc
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, robust to the monotone-but-nonlinear case."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return 0.0
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float(np.dot(ra, ra) * np.dot(rb, rb)))
    if denom == 0:
        return 0.0
    return float(np.dot(ra, rb) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (equivalent to scipy.stats.rankdata)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    # Average the ranks inside each tied group.
    sorted_vals = values[order]
    start = 0
    for idx in range(1, values.size + 1):
        if idx == values.size or sorted_vals[idx] != sorted_vals[start]:
            if idx - start > 1:
                ranks[order[start:idx]] = ranks[order[start:idx]].mean()
            start = idx
    return ranks


@dataclass
class ChangePoint:
    """A detected level shift in a series."""

    index: int
    step: float
    before: float  #: raw median before the shift, for display
    after: float  #: raw median after the shift, for display
    magnitude: float  #: shift size in robust sigmas, measured on the detrended series
    shift: float = 0.0  #: signed shift on the detrended series

    @property
    def is_drop(self) -> bool:
        """Direction of the *discontinuity*, which on a trending series can
        differ from the direction of the raw levels."""
        return self.shift < 0


def diff_sigma(y: np.ndarray) -> float:
    """Noise scale estimated from successive differences.

    ``mad(y)`` is the wrong scale whenever the series has structure: a step or a
    trend inflates it, and the feature then hides inside its own noise estimate.
    Differencing removes any level and most of a slow trend, and for i.i.d.
    noise ``std(diff) = sqrt(2) * sigma``.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return 0.0
    return float(mad(np.diff(y)) / math.sqrt(2.0))


def cusum_changepoint(
    x: np.ndarray,
    y: np.ndarray,
    min_sigma: float = 3.0,
    edge_frac: float = 0.1,
    detrend: bool = True,
) -> Optional[ChangePoint]:
    """Locate the single most likely *level shift* via a CUSUM scan.

    Two corrections make this usable on training curves:

    * **Detrend first.** A CUSUM on a steadily rising series always finds a
      "shift" at the middle, because a ramp really does have a different mean
      before and after any split point. Removing a robust linear fit means we
      detect discontinuities -- a resumed checkpoint, a changed reward function
      -- rather than rediscovering the trend.
    * **Scale by differenced noise.** See :func:`diff_sigma`.

    ``edge_frac`` refuses candidates in the first/last slice of the run, where a
    one-sided window makes the "before" or "after" mean meaningless.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = y.size
    if n < 12:
        return None

    if detrend:
        slope, intercept = theil_sen(x, y)
        residual = y - (intercept + slope * x)
    else:
        residual = y - float(np.median(y))

    scale = diff_sigma(residual) or mad(residual)
    if scale <= 0:
        return None

    cumulative = np.cumsum(residual - residual.mean())
    guard = max(int(n * edge_frac), 3)
    interior = np.arange(guard, n - guard)
    if interior.size == 0:
        return None

    idx = int(interior[np.argmax(np.abs(cumulative[interior]))])
    shift = float(np.median(residual[idx:]) - np.median(residual[:idx]))
    magnitude = abs(shift) / scale
    if magnitude < min_sigma:
        return None

    # Report the raw levels -- that is what a reader wants to see -- while
    # judging significance and direction on the detrended residual.
    return ChangePoint(
        index=idx,
        step=float(x[idx]),
        before=float(np.median(y[:idx])),
        after=float(np.median(y[idx:])),
        magnitude=float(magnitude),
        shift=shift,
    )


def project_crossing(tr: Trend, current_x: float, current_y: float, threshold: float) -> Optional[float]:
    """Extrapolate a linear trend to the x where it crosses ``threshold``.

    Returns ``None`` when the trend moves away from the threshold or is flat.
    This is deliberately a naive linear extrapolation -- it is presented to the
    user as an order-of-magnitude "you have about this long", not a forecast.
    """
    if tr.slope == 0 or not tr.significant:
        return None
    delta = threshold - current_y
    if delta == 0:
        return current_x
    if (delta > 0) != (tr.slope > 0):
        return None
    return current_x + delta / tr.slope


def project_threshold(
    x: np.ndarray, y: np.ndarray, threshold: float
) -> Tuple[Optional[float], str]:
    """Project when a series will cross ``threshold``, choosing the better model.

    Quantities like entropy and gradient norm decay *multiplicatively*, so a
    straight-line extrapolation of a decaying exponential badly overstates how
    soon it will arrive -- it reads the current steep slope as if it continued
    forever. Getting "15 steps" when the true answer is 150 is the kind of error
    that costs a tool its credibility the first time someone checks.

    So we fit both a linear and a log-linear model, keep whichever has the
    smaller residual spread in the original units, and report which one was
    used. Returns ``(crossing_x, model)`` where model is "linear", "exponential"
    or "none".
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 8:
        return None, "none"

    linear = trend(x, y)
    best = ("linear", linear, mad(y - (linear.intercept + linear.slope * x)))

    if np.all(y > 0) and threshold > 0:
        log_fit = trend(x, np.log(y))
        residual = mad(y - np.exp(log_fit.intercept + log_fit.slope * x))
        if residual < best[2]:
            best = ("exponential", log_fit, residual)

    model, fit, _ = best
    if not fit.significant or fit.slope == 0:
        return None, "none"

    current_x = float(x[-1])
    if model == "exponential":
        current_y = float(np.exp(fit.intercept + fit.slope * current_x))
        target, value = math.log(threshold), math.log(max(current_y, 1e-300))
    else:
        current_y = fit.intercept + fit.slope * current_x
        target, value = threshold, current_y

    delta = target - value
    if delta == 0:
        return current_x, model
    if (delta > 0) != (fit.slope > 0):
        return None, "none"  # moving away from the threshold
    return current_x + delta / fit.slope, model


def head(arr: np.ndarray, frac: float = 0.10, minimum: int = 5, maximum: int = 25) -> np.ndarray:
    """First slice of a series -- 'where this run started'.

    Capped at ``maximum`` points on purpose. A fraction-only window drifts as a
    run grows, so in live monitoring the "initial" value would keep being
    recomputed over later and later steps -- and a baseline that follows the
    series it is supposed to anchor cannot detect a decline against it.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr
    k = min(max(minimum, int(round(arr.size * frac))), maximum, arr.size)
    return arr[:k]


def tail(arr: np.ndarray, frac: float = 0.25, minimum: int = 5) -> np.ndarray:
    """Last ``frac`` of a series -- 'what is happening right now'."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr
    k = max(minimum, int(round(arr.size * frac)))
    return arr[-min(k, arr.size) :]


def nanmedian(arr: np.ndarray, default: float = float("nan")) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.median(arr))
