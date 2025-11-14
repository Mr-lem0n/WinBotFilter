# scorer/good_entry_nonparam.py
# -*- coding: utf-8 -*-
"""
Distribution-free entry scoring using bootstrap-based local trend detection
and ensemble forecasts from KF/EnKF.

This module is a **non-parametric alternative** to the jump–diffusion DDM
approach. It does *not* assume Gaussian increments or any specific jump law.

Core idea
---------
1. Work with log prices:
       X_t = log S_t,  r_k = X_k - X_{k-1}.
2. Use bootstrap (optionally moving-block) to approximate the empirical
   distribution of the **average drift** over the window:
       m = mean(r_k).
3. Classify each bootstrap sample's mean drift as:
       - "up"   if m > +drift_eps,
       - "down" if m < -drift_eps,
       - "flat" otherwise.
   The relative frequencies of those cases give:
       P_up, P_flat, P_down  (distribution-free).
4. Use the forecast ensemble (from KF/EnKF) to compute expected **relative
   profit** for:
       - a long:   E_rel_profit_long (%)
       - a short:  E_rel_profit_short (%)
5. Apply thresholds:
       - only enter long if:
              E_rel_profit_long >= profit_threshold_rel (e.g. 0.2%)
              AND P_up >= prob_threshold (e.g. 0.6)
       - only enter short if:
              E_rel_profit_short >= profit_threshold_rel
              AND P_down >= prob_threshold
       - otherwise: 'not-enter'.

To integrate into your pipeline
-------------------------------
In MarketDataProcessor.run_model, after you computed:

    smooth_Data = self.smooth_time_series(array_to_analyse, len(array_to_analyse))
    horizon = 20
    if self.ensemble > 0:
        X = self.enkf.forecast(smooth_Data, horizon, self.ensemble)
    else:
        X = self.kf.forecast(smooth_Data, horizon)[None, :]

and after you have snr from your RMT cleaning, replace the old logistic scoring with:

    from scorer.good_entry_nonparam import GoodEntryNonParam

    entry_res = GoodEntryNonParam(
        prices_window=smooth_Data,
        forecast_paths=X,
        snr=snr,
        profit_threshold_rel=0.2,  # 0.2% min expected relative profit
        prob_threshold=0.6,
        n_bootstrap=500,
        block_size=None,           # or e.g. 5–20 for moving-block bootstrap
        random_state=None,         # or an int for reproducibility
    )
    self.signal = entry_res["decision"]

    print("Entry scoring result (non-parametric):")
    print(f"  P_up   = {entry_res['P_up']:.3f}")
    print(f"  P_flat = {entry_res['P_flat']:.3f}")
    print(f"  P_down = {entry_res['P_down']:.3f}")
    print(f"  E_rel_profit_long  = {entry_res['E_rel_profit_long']:.3f}%")
    print(f"  E_rel_profit_short = {entry_res['E_rel_profit_short']:.3f}%")
    print(f"  decision = {self.signal}")

Everything downstream (exit logic, CSV logging) remains unchanged, since
`self.signal` still takes the same values: 'enter-long' / 'enter-short' / 'not-enter'.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# 1. Bootstrap helpers (distribution-free drift estimation)
# ---------------------------------------------------------------------------

def _bootstrap_means(
    returns: np.ndarray,
    n_bootstrap: int = 500,
    block_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate bootstrap samples of the mean return, using either simple
    i.i.d. bootstrap or moving-block bootstrap (if block_size is provided).

    Parameters
    ----------
    returns : np.ndarray
        1D array of log-returns over the window.
    n_bootstrap : int
        Number of bootstrap replications.
    block_size : int or None
        If None or <= 1 or >= len(returns):
            -> standard i.i.d. resampling (simple bootstrap).
        Else:
            -> moving-block bootstrap with the specified block size.
    rng : numpy.random.Generator or None
        Random number generator; if None, a new default_rng() is created.

    Returns
    -------
    np.ndarray
        1D array of bootstrap sample means (length n_bootstrap).
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or len(returns) == 0:
        raise ValueError("_bootstrap_means: returns must be 1D, length > 0")

    if rng is None:
        rng = np.random.default_rng()

    n = len(returns)
    if n_bootstrap <= 0:
        raise ValueError("_bootstrap_means: n_bootstrap must be >= 1")

    # Simple i.i.d. bootstrap
    if block_size is None or block_size <= 1 or block_size >= n:
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        samples = returns[idx]
        means = samples.mean(axis=1)
        return means

    # Moving-block bootstrap (for serially correlated returns)
    block_size = int(block_size)
    n_blocks = int(np.ceil(n / block_size))

    means = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        sample = np.empty(n, dtype=float)
        pos = 0
        for _ in range(n_blocks):
            start = rng.integers(0, n - block_size + 1)
            block = returns[start:start + block_size]
            end = min(pos + block_size, n)
            sample[pos:end] = block[: (end - pos)]
            pos = end
            if pos >= n:
                break
        means[b] = sample.mean()

    return means


def _distribution_free_trend_probs(
    returns: np.ndarray,
    n_bootstrap: int = 500,
    drift_eps: Optional[float] = None,
    block_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Estimate P_up, P_flat, P_down in a distribution-free way using bootstrap.

    Strategy
    --------
    - Use bootstrap to approximate the sampling distribution of the average
      return (drift) over the window.
    - Define a small threshold drift_eps > 0.
    - For each bootstrap sample, compute the average drift m_b and classify:
          m_b > +drift_eps   -> "up"
          m_b < -drift_eps   -> "down"
          otherwise          -> "flat"
    - Relative frequencies yield empirical probabilities for up/flat/down.

    Parameters
    ----------
    returns : np.ndarray
        1D array of log-returns over the window.
    n_bootstrap : int
        Number of bootstrap replications.
    drift_eps : float or None
        Minimal drift magnitude considered "significant" (per step).
        If None, we set it to:
            drift_eps = c * std(returns) / sqrt(n),
        with c ≈ 0.25 by default.
    block_size : int or None
        Block size for moving-block bootstrap. If None, use i.i.d. bootstrap.
    rng : numpy.random.Generator or None
        Random number generator; if None, a new default_rng() is created.

    Returns
    -------
    dict
        Dictionary with:
            - "P_up"
            - "P_flat"
            - "P_down"
            - "drift_eps" (the actual threshold used)
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or len(returns) == 0:
        raise ValueError("_distribution_free_trend_probs: returns must be 1D, length > 0")

    n = len(returns)
    std = float(np.std(returns)) if np.std(returns) > 0 else 1e-12

    if drift_eps is None:
        # Default threshold: fraction of the standard error of the mean
        drift_eps = 0.25 * std / np.sqrt(n)

    if rng is None:
        rng = np.random.default_rng()

    means = _bootstrap_means(
        returns=returns,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        rng=rng,
    )

    up = float(np.mean(means > drift_eps))
    down = float(np.mean(means < -drift_eps))
    flat = 1.0 - up - down

    # Numerical safeguard if flat < 0 due to rounding
    if flat < 0.0:
        flat = 0.0
        s = up + down
        if s > 0:
            up /= s
            down /= s
        else:
            up = down = 0.5

    return {
        "P_up": up,
        "P_flat": flat,
        "P_down": down,
        "drift_eps": drift_eps,
    }


# ---------------------------------------------------------------------------
# 2. Main non-parametric entry scorer
# ---------------------------------------------------------------------------

def GoodEntryNonParam(
    prices_window: np.ndarray,
    forecast_paths: np.ndarray,
    snr: float = 1.0,
    profit_threshold_rel: float = 0.2,
    prob_threshold: float = 0.6,
    n_bootstrap: int = 500,
    drift_eps: Optional[float] = None,
    block_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Distribution-free entry scoring using bootstrap + ensemble forecasts.

    This function is a non-parametric alternative to the jump–diffusion DDM
    version of GoodEntry. It keeps the same **external behaviour**:

        decision in {"enter-long", "enter-short", "not-enter"}

    but computes the regime probabilities P_up, P_flat, P_down via bootstrap,
    without assuming a parametric distribution for returns.

    Conceptual pipeline
    -------------------
    1. Input:
         - prices_window: historical price window (1D, length >= 3).
         - forecast_paths: KF/EnKF forecasts (N_scenarios x horizon).
         - snr: signal-to-noise ratio from RMT cleaning.
    2. Compute log-returns from prices_window.
    3. Use bootstrap (optionally moving-block) to estimate the empirical
       distribution of the average drift and obtain:
           raw_P_up, raw_P_flat, raw_P_down.
    4. Use snr to **shrink** those probabilities towards "flat" when snr is
       low (we distrust all trend signals), and trust them more when snr is
       high. This yields final P_up, P_flat, P_down.
    5. Use the forecast ensemble to compute expected **relative profit**
       (in %) for:
           - long:   E_rel_profit_long
           - short:  E_rel_profit_short
    6. Apply thresholds:
           - long only if:
                 E_rel_profit_long >= profit_threshold_rel
                 AND P_up >= prob_threshold
           - short only if:
                 E_rel_profit_short >= profit_threshold_rel
                 AND P_down >= prob_threshold
           - else: not-enter.

    Parameters
    ----------
    prices_window : array-like
        Historical price window (e.g. smoothed offer prices for ~240s).
        1D array of length >= 3.
    forecast_paths : array-like
        Forecasted price paths from KF/EnKF.
        Shapes:
            - (N_scenarios, horizon)
            - or (horizon,) for a single path (promoted to (1, horizon)).
        Only the last column (end of horizon) is used for profit estimation.
    snr : float, optional
        Signal-to-noise ratio. Used only to shrink probabilities towards
        "flat" when snr is low:
            shrink = exp(-alpha_snr * snr).
    profit_threshold_rel : float, optional
        Minimum expected **relative** profit (%) required to enter a trade.
        Consistent with your CSV definition:
            rel_profit = (abs_profit * 100) / entry_price.
        So 0.2 means "at least 0.2% expected gain".
    prob_threshold : float, optional
        Minimum P_up (for long) or P_down (for short) to commit.
    n_bootstrap : int, optional
        Number of bootstrap replications for drift estimation.
    drift_eps : float or None, optional
        Minimal per-step drift magnitude considered "significant".
        If None, uses c * std(returns) / sqrt(n), with c ≈ 0.25.
    block_size : int or None, optional
        Block size for moving-block bootstrap. If None, use simple bootstrap.
    random_state : int or None, optional
        Seed for reproducible bootstrap. If None, bootstrap is random.

    Returns
    -------
    dict
        Dictionary with:
            - "decision"           : 'enter-long' / 'enter-short' / 'not-enter'
            - "P_up"               : probability of up-trend (after SNR shrink)
            - "P_flat"             : probability of flat/random trend
            - "P_down"             : probability of down-trend
            - "E_rel_profit_long"  : expected relative profit (%) of a long
            - "E_rel_profit_short" : expected relative profit (%) of a short
            - "S0"                 : last price in window (entry price)
            - "raw_P_up"           : raw bootstrap P_up (before shrinkage)
            - "raw_P_flat"         : raw bootstrap P_flat
            - "raw_P_down"         : raw bootstrap P_down
            - "drift_eps"          : drift threshold actually used
            - "snr"                : SNR used for shrinkage
    """
    # ---------- Input normalisation ----------
    prices_window = np.asarray(prices_window, dtype=float)
    if prices_window.ndim != 1 or len(prices_window) < 3:
        raise ValueError("GoodEntryNonParam: prices_window must be 1D with length >= 3.")

    X = np.asarray(forecast_paths, dtype=float)
    if X.ndim == 1:
        X = X[None, :]  # promote (horizon,) -> (1, horizon)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("GoodEntryNonParam: forecast_paths must have shape (N_scenarios, horizon).")

    rng = np.random.default_rng(random_state)

    # ---------- Step 1: log-returns ----------
    log_p = np.log(prices_window)
    returns = np.diff(log_p)

    # ---------- Step 2: bootstrap P_up / P_flat / P_down ----------
    boot_res = _distribution_free_trend_probs(
        returns=returns,
        n_bootstrap=n_bootstrap,
        drift_eps=drift_eps,
        block_size=block_size,
        rng=rng,
    )
    raw_P_up = boot_res["P_up"]
    raw_P_flat = boot_res["P_flat"]
    raw_P_down = boot_res["P_down"]
    drift_eps_used = boot_res["drift_eps"]

    # ---------- Step 3: SNR-based shrinkage towards flat ----------
    # shrink ∈ (0,1):
    #   - snr ≈ 0   -> shrink ≈ 1 (strong shrinkage, push mass to flat)
    #   - snr large -> shrink ≈ 0 (trust bootstrap)
    alpha_snr = 0.7
    shrink = float(np.exp(-alpha_snr * snr))

    P_flat = raw_P_flat + shrink * (raw_P_up + raw_P_down) / 2.0
    P_up = raw_P_up * (1.0 - shrink)
    P_down = raw_P_down * (1.0 - shrink)

    # Renormalize to ensure probabilities sum to 1
    s = P_up + P_flat + P_down
    if s <= 0:
        P_up = P_down = 0.0
        P_flat = 1.0
    else:
        P_up /= s
        P_flat /= s
        P_down /= s

    # ---------- Step 4: expected relative profit from forecasts ----------
    S0 = float(prices_window[-1])
    future_prices = X[:, -1]  # end-of-horizon price per scenario

    rel_long_s = (future_prices - S0) / S0 * 100.0   # long: up is profit
    rel_short_s = (S0 - future_prices) / S0 * 100.0  # short: down is profit

    E_rel_long = float(np.mean(rel_long_s))
    E_rel_short = float(np.mean(rel_short_s))

    # ---------- Step 5: decision ----------
    decision = "not-enter"

    long_ok = (E_rel_long >= profit_threshold_rel) and (P_up >= prob_threshold)
    short_ok = (E_rel_short >= profit_threshold_rel) and (P_down >= prob_threshold)

    if long_ok and not short_ok:
        decision = "enter-long"
    elif short_ok and not long_ok:
        decision = "enter-short"
    elif long_ok and short_ok:
        # If both pass, choose the side with higher expected profit
        decision = "enter-long" if E_rel_long >= E_rel_short else "enter-short"
    else:
        decision = "not-enter"

    return {
        "decision": decision,
        "P_up": P_up,
        "P_flat": P_flat,
        "P_down": P_down,
        "E_rel_profit_long": E_rel_long,
        "E_rel_profit_short": E_rel_short,
        "S0": S0,
        "raw_P_up": raw_P_up,
        "raw_P_flat": raw_P_flat,
        "raw_P_down": raw_P_down,
        "drift_eps": drift_eps_used,
        "snr": snr,
    }


# ---------------------------------------------------------------------------
# 3. Minimal smoke test (safe to delete in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple monotone increasing price path + single forecast
    prices = np.linspace(100.0, 110.0, 240)  # window
    horizon = 20
    forecast = np.linspace(prices[-1], prices[-1] + 1.0, horizon)  # ends higher

    res = GoodEntryNonParam(
        prices_window=prices,
        forecast_paths=forecast,
        snr=2.0,
        profit_threshold_rel=0.2,
        prob_threshold=0.6,
        n_bootstrap=200,
        block_size=None,
        random_state=42,
    )

    print("Non-parametric smoke test result:")
    for k, v in res.items():
        print(f"{k}: {v}")
