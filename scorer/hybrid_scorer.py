# scorer/hybrid_scorer.py
# -*- coding: utf-8 -*-
"""
Hybrid, distribution-free entry scoring & timing on top of ensemble forecasts.

This module is designed to plug directly into your existing pipeline
(KF / EnKF forecasts, RMT-based SNR, etc.).

It provides two main components:

1) GoodEntryHybrid
   ----------------
   A *directional* entry scorer that combines:
     - distribution-free, bootstrap-based trend probabilities
       (no Gaussian / jump assumptions),
     - a DDM-style evidence accumulation path built from bootstrap
       log-odds, with symmetric boundaries,
     - ensemble forecasts from KF/EnKF to compute expected relative
       profit for long vs short.

   Output:
     - decision in {"enter-long", "enter-short", "not-enter"},
     - P_up / P_flat / P_down,
     - evidence path & DDM signal,
     - expected relative profit for long/short.

2) analyze_entry_timing
   ---------------------
   A *timing* layer that, given:
     - direction ('enter-long' / 'enter-short'),
     - ensemble forecast paths (N_scenarios x horizon),
     - current price S0,
   evaluates, for candidate entry offsets (0..max_entry_delay):

     - probability that the trend does NOT reverse over the next
       window_len steps,
     - probability that a given profit threshold is hit BEFORE
       a trend break within that window,
     - expected relative profit at the end of the window.

   It returns the best entry offset according to a simple objective:
     score_k = P(hit profit before break at offset k) * max(E[rel profit end]_k, 0).

Usage sketch inside MarketDataProcessor.run_model
-------------------------------------------------
Assuming you already have:

    smooth_Data = self.smooth_time_series(array_to_analyse, len(array_to_analyse))
    horizon = 20
    if self.ensemble > 0:
        X = self.enkf.forecast(smooth_Data, horizon, self.ensemble)
    else:
        X = self.kf.forecast(smooth_Data, horizon)[None, :]

    # ... compute cov, snr via RMT or simple covariance ...

You can do:

    from scorer.hybrid_scorer import GoodEntryHybrid, analyze_entry_timing

    # 1) Directional decision
    entry_res = GoodEntryHybrid(
        prices_window=smooth_Data,
        forecast_paths=X,
        snr=snr,
        profit_threshold_rel=0.2,  # 0.2% expected profit threshold
        prob_threshold=0.6,
        n_bootstrap_final=500,
        n_bootstrap_evidence=200,
        block_size=None,
        evidence_boundary=1.5,
    )
    self.signal = entry_res["decision"]

    # 2) Optional timing analysis if we have a direction
    if self.signal in ("enter-long", "enter-short"):
        timing_res = analyze_entry_timing(
            forecast_paths=X,
            direction=self.signal,
            S0=entry_res["S0"],
            profit_threshold_rel=0.2,
            window_len=5,
            max_entry_delay=5,
        )
        best_offset = timing_res["best_offset"]
        # -> You can decide whether to enter now (offset 0)
        #    or delay entry by 'best_offset' steps according to your design.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

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
    - For each bootstrap sample's mean drift m_b, classify:
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


def _bootstrap_evidence_path(
    returns: np.ndarray,
    n_bootstrap: int = 200,
    drift_eps: Optional[float] = None,
    block_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Build a non-parametric DDM-like evidence path using bootstrap.

    For each prefix of returns r[:t], t = 1..n, we:
      1) estimate P_up_t and P_down_t via bootstrap, using
         _distribution_free_trend_probs on r[:t],
      2) compute a "log-odds" surrogate:
             log_odds_t = log( (P_up_t + eps) / (P_down_t + eps) ),
      3) define an evidence process E_t as the cumulative change in log-odds:
             E_t = E_{t-1} + (log_odds_t - log_odds_{t-1}),
         which is algebraically equivalent to E_t = log_odds_t, but makes
         explicit the "incremental update" like in a DDM.

    Parameters
    ----------
    returns : np.ndarray
        1D array of log-returns over the window.
    n_bootstrap : int
        Number of bootstrap replications at each step. Smaller than the
        global n_bootstrap used for final probabilities to keep runtime
        reasonable (e.g. 100-200).
    drift_eps : float or None
        Drift threshold passed to _distribution_free_trend_probs. If None,
        that function will choose its default (fraction of std / sqrt(n)).
    block_size : int or None
        Block size for moving-block bootstrap. If None, use simple bootstrap.
    rng : numpy.random.Generator or None
        RNG. If None, a new default_rng() is created.

    Returns
    -------
    dict with keys:
        - "evidence"    : np.ndarray of shape (n,), evidence path E_t
        - "log_odds"    : np.ndarray of shape (n,), log_odds_t
        - "P_up_path"   : np.ndarray of shape (n,), P_up_t per step
        - "P_down_path" : np.ndarray of shape (n,), P_down_t per step
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or len(returns) == 0:
        raise ValueError("_bootstrap_evidence_path: returns must be 1D, length > 0")

    n = len(returns)
    if rng is None:
        rng = np.random.default_rng()

    evidence = np.zeros(n, dtype=float)
    log_odds = np.zeros(n, dtype=float)
    P_up_path = np.zeros(n, dtype=float)
    P_down_path = np.zeros(n, dtype=float)

    eps = 1e-6
    prev_log_odds = 0.0
    prev_E = 0.0

    for t in range(1, n + 1):
        prefix = returns[:t]
        boot_res = _distribution_free_trend_probs(
            returns=prefix,
            n_bootstrap=n_bootstrap,
            drift_eps=drift_eps,
            block_size=block_size,
            rng=rng,
        )
        P_up_t = boot_res["P_up"]
        P_down_t = boot_res["P_down"]

        # Log-odds surrogate; eps guards against log(0)
        lo_t = np.log((P_up_t + eps) / (P_down_t + eps))

        # Incremental update to make the "accumulation" explicit
        delta_lo = lo_t - prev_log_odds
        E_t = prev_E + delta_lo

        idx = t - 1
        evidence[idx] = E_t
        log_odds[idx] = lo_t
        P_up_path[idx] = P_up_t
        P_down_path[idx] = P_down_t

        prev_log_odds = lo_t
        prev_E = E_t

    return {
        "evidence": evidence,
        "log_odds": log_odds,
        "P_up_path": P_up_path,
        "P_down_path": P_down_path,
    }


# ---------------------------------------------------------------------------
# 2. Hybrid entry scorer: distribution-free DDM + ensemble profit
# ---------------------------------------------------------------------------

def GoodEntryHybrid(
    prices_window: np.ndarray,
    forecast_paths: np.ndarray,
    snr: float = 1.0,
    profit_threshold_rel: float = 0.2,
    prob_threshold: float = 0.6,
    n_bootstrap_final: int = 500,
    n_bootstrap_evidence: int = 200,
    drift_eps: Optional[float] = None,
    block_size: Optional[int] = None,
    evidence_boundary: float = 1.5,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Hybrid entry scoring:
    - distribution-free bootstrap for directional *probabilities*,
    - distribution-free DDM-like evidence *accumulation*,
    - ensemble forecasts for expected relative profit.

    This combines two ingredients:
      1) Non-parametric, bootstrap-based trend detection (no explicit
         likelihood form).
      2) A DDM-style evidence process that "accumulates" information and
         uses thresholds.

    Decision logic
    --------------
    1. Historical window:
        - prices_window -> log-returns r.
        - Use _distribution_free_trend_probs on the *full* r with
          n_bootstrap_final -> raw_P_up / raw_P_down / raw_P_flat.
        - Use snr to shrink these raw probabilities towards "flat"
          when snr is small (we distrust all trend signals), and trust
          them more when snr is high.
    2. Evidence path:
        - Use _bootstrap_evidence_path on r with n_bootstrap_evidence to
          build an evidence trajectory E_t and log-odds_t.
        - DDM-style signal:
              if max(E_t) >= evidence_boundary:    ddm_signal = 'enter-long'
              elif min(E_t) <= -evidence_boundary: ddm_signal = 'enter-short'
              else:                                ddm_signal = 'not-enter'
    3. Forecast ensemble:
        - forecast_paths (N_scenarios x horizon) -> end-of-horizon prices.
        - Expected relative profit (%), long vs short:
              E_rel_profit_long  = mean( (P_T - S0) / S0 * 100 )
              E_rel_profit_short = mean( (S0 - P_T) / S0 * 100 )
    4. Final decision:
        - candidate_long  if:
              ddm_signal == 'enter-long'
              AND P_up >= prob_threshold
              AND E_rel_profit_long  >= profit_threshold_rel
        - candidate_short if:
              ddm_signal == 'enter-short'
              AND P_down >= prob_threshold
              AND E_rel_profit_short >= profit_threshold_rel
        - pick the better of candidates; else 'not-enter'.

    Parameters
    ----------
    prices_window : array-like
        Historical price window (e.g., smoothed offer prices).
        1D array, length >= 3.
    forecast_paths : array-like
        KF/EnKF forecast paths:
            - shape (N_scenarios, horizon)
            - or (horizon,) for a single path (promoted to 2D).
    snr : float
        Signal-to-noise ratio from RMT cleaning. Used to shrink raw
        bootstrap probabilities towards "flat".
    profit_threshold_rel : float
        Minimum expected relative profit (%) required to trade (e.g., 0.2).
    prob_threshold : float
        Minimum P_up / P_down required for long/short.
    n_bootstrap_final : int
        Number of bootstrap replications for the *global* P_up/P_down/P_flat.
    n_bootstrap_evidence : int
        Number of bootstrap replications per step for the evidence path.
        Typically smaller than n_bootstrap_final (e.g., 100-200).
    drift_eps : float or None
        Drift threshold passed to bootstrap routines; if None, they choose
        a default based on std(returns) / sqrt(n).
    block_size : int or None
        Block size for moving-block bootstrap; if None, use simple bootstrap.
    evidence_boundary : float
        Symmetric DDM-like boundary for the evidence path. Higher values
        require stronger aggregated evidence to trigger long/short.
    random_state : int or None
        Seed for reproducible bootstrap. If None, bootstrap is random.

    Returns
    -------
    dict
        - "decision"           : 'enter-long' / 'enter-short' / 'not-enter'
        - "P_up"               : shrunk P_up
        - "P_flat"             : shrunk P_flat
        - "P_down"             : shrunk P_down
        - "E_rel_profit_long"  : expected relative profit (%) long
        - "E_rel_profit_short" : expected relative profit (%) short
        - "S0"                 : last price in window
        - "raw_P_up"           : raw bootstrap P_up (full window)
        - "raw_P_flat"         : raw bootstrap P_flat
        - "raw_P_down"         : raw bootstrap P_down
        - "drift_eps"          : drift threshold used
        - "snr"                : snr used for shrinkage
        - "evidence_path"      : np.ndarray, E_t
        - "log_odds_path"      : np.ndarray, log-odds_t
        - "ddm_signal"         : 'enter-long' / 'enter-short' / 'not-enter'
    """
    prices_window = np.asarray(prices_window, dtype=float)
    if prices_window.ndim != 1 or len(prices_window) < 3:
        raise ValueError("GoodEntryHybrid: prices_window must be 1D with length >= 3.")

    X = np.asarray(forecast_paths, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("GoodEntryHybrid: forecast_paths must have shape (N_scenarios, horizon).")

    rng = np.random.default_rng(random_state)

    # Step 1: log-returns
    log_p = np.log(prices_window)
    returns = np.diff(log_p)

    # Step 2: global bootstrap probabilities on full window
    boot_res_full = _distribution_free_trend_probs(
        returns=returns,
        n_bootstrap=n_bootstrap_final,
        drift_eps=drift_eps,
        block_size=block_size,
        rng=rng,
    )
    raw_P_up = boot_res_full["P_up"]
    raw_P_flat = boot_res_full["P_flat"]
    raw_P_down = boot_res_full["P_down"]
    drift_eps_used = boot_res_full["drift_eps"]

    # Step 3: SNR-based shrinkage towards flat
    alpha_snr = 0.7
    shrink = float(np.exp(-alpha_snr * snr))

    P_flat = raw_P_flat + shrink * (raw_P_up + raw_P_down) / 2.0
    P_up = raw_P_up * (1.0 - shrink)
    P_down = raw_P_down * (1.0 - shrink)

    s = P_up + P_flat + P_down
    if s <= 0:
        P_up = P_down = 0.0
        P_flat = 1.0
    else:
        P_up /= s
        P_flat /= s
        P_down /= s

    # Step 4: evidence path via bootstrap log-odds on prefixes
    ev_res = _bootstrap_evidence_path(
        returns=returns,
        n_bootstrap=n_bootstrap_evidence,
        drift_eps=drift_eps_used,
        block_size=block_size,
        rng=rng,
    )
    evidence_path = ev_res["evidence"]
    log_odds_path = ev_res["log_odds"]

    max_E = float(np.max(evidence_path))
    min_E = float(np.min(evidence_path))
    if max_E >= evidence_boundary:
        ddm_signal = "enter-long"
    elif min_E <= -evidence_boundary:
        ddm_signal = "enter-short"
    else:
        ddm_signal = "not-enter"

    # Step 5: expected relative profit from ensemble forecasts
    S0 = float(prices_window[-1])
    future_prices = X[:, -1]

    rel_long_s = (future_prices - S0) / S0 * 100.0
    rel_short_s = (S0 - future_prices) / S0 * 100.0

    E_rel_long = float(np.mean(rel_long_s))
    E_rel_short = float(np.mean(rel_short_s))

    # Step 6: final decision combining DDM + probabilities + profit
    decision = "not-enter"

    long_ok = (
        ddm_signal == "enter-long"
        and P_up >= prob_threshold
        and E_rel_long >= profit_threshold_rel
    )
    short_ok = (
        ddm_signal == "enter-short"
        and P_down >= prob_threshold
        and E_rel_short >= profit_threshold_rel
    )

    if long_ok and not short_ok:
        decision = "enter-long"
    elif short_ok and not long_ok:
        decision = "enter-short"
    elif long_ok and short_ok:
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
        "evidence_path": evidence_path,
        "log_odds_path": log_odds_path,
        "ddm_signal": ddm_signal,
    }


# ---------------------------------------------------------------------------
# 3. Entry timing analysis on top of the hybrid scorer
# ---------------------------------------------------------------------------

def analyze_entry_timing(
    forecast_paths: np.ndarray,
    direction: str,
    S0: float,
    profit_threshold_rel: float = 0.2,
    window_len: int = 5,
    max_entry_delay: int = 5,
) -> Dict[str, Any]:
    """
    Analyze entry timing given an ensemble of forecasts and trade direction.

    Parameters
    ----------
    forecast_paths : array-like
        Ensemble of future price paths from KF/EnKF.
        Shape:
            - (N_scenarios, horizon)
        Each row is one scenario; each column is one future time step
        (same dt as your aggregation, e.g. 1 second).
    direction : {'enter-long', 'enter-short'}
        Trade direction you *intend* to take, as decided by the hybrid scorer.
        Any other value will raise.
    S0 : float
        Current price at time t=0 (last observed price in your window).
        This is the true spot now, not a forecast.
    profit_threshold_rel : float, optional
        Profit threshold in **relative** terms (percent) used as the
        "take-profit" level for timing analysis. For example:
            0.2  => +0.2% relative profit from the entry price.
        We estimate, for each candidate entry time k:
            P( reach +profit_threshold_rel% BEFORE local trend reversal
               within the next window_len steps | entry at k ).
    window_len : int, optional
        Length of the "timing window" in forecast steps (e.g. 5 seconds).
        For each candidate entry offset k, we look up to k+window_len
        (clipped by the horizon).
    max_entry_delay : int, optional
        Maximum number of forecast steps ahead at which we consider entering.
        Example:
            max_entry_delay = 5
        means we consider entry at:
            k = 0 (now), 1, 2, 3, 4, 5
        as long as the forecast horizon is long enough.

    Returns
    -------
    dict
        - "best_offset": int
              The entry offset (0..max_entry_delay) that maximises the
              chosen objective. 0 means "enter now".

        - "score_per_offset": np.ndarray of shape (K,)
              Objective used to select best_offset (higher is better).

        - "p_trend_survival": np.ndarray of shape (K,)
              For each offset k: estimated probability that the trend does
              NOT reverse during the next window_len steps after k.

        - "p_profit_before_break": np.ndarray of shape (K,)
              For each offset k: probability that the profit threshold is
              reached before the first trend break (and within window_len).

        - "E_rel_profit_end": np.ndarray of shape (K,)
              For each offset k: expected relative profit (%) at the end
              of the window (k + window_len, clipped).

        - "break_time_distribution_now": np.ndarray
              For entry at offset 0 (now), distribution of "time to first
              break" in steps across scenarios. This only counts scenarios
              where a break occurs within the horizon.

        - "direction": str
              Echoes the input direction.

        - "analysis_params": dict
              Echoes parameters used: profit_threshold_rel, window_len,
              max_entry_delay.

    Notes
    -----
    1) "Trend reversal" is defined locally on the forecast as a change in the
       sign of the **increment** against the intended direction:

           increment_j = X[s, j] - X[s, j-1]
           direction_sign = +1 for long, -1 for short

       A break occurs at j if direction_sign * increment_j < 0.

    2) "Profit threshold hit" for a given scenario s and offset k means there
       exists some j in [k, k+window_len] (clipped) such that:

           direction_sign * (X[s, j] - S_entry_s) >= tau_abs

       where S_entry_s is the entry price (S0 at offset 0, otherwise the
       forecasted price X[s, k] at offset k), and tau_abs is the absolute
       profit corresponding to profit_threshold_rel:

           tau_abs = profit_threshold_rel / 100 * S_entry_s

    3) Objective for each offset k (for selection) is:

           score_k = p_profit_before_break[k] * max(E_rel_profit_end[k], 0)

       i.e. probability of "good event" times expected upside at window end.
    """
    X = np.asarray(forecast_paths, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("analyze_entry_timing: forecast_paths must have shape (N_scenarios, horizon>=2).")

    if direction not in ("enter-long", "enter-short"):
        raise ValueError("analyze_entry_timing: direction must be 'enter-long' or 'enter-short'.")

    direction_sign = 1.0 if direction == "enter-long" else -1.0

    N, H = X.shape
    max_delay = min(max_entry_delay, H - 2)  # need at least 1 step after entry
    K = max_delay + 1

    p_trend_survival = np.zeros(K, dtype=float)
    p_profit_before_break = np.zeros(K, dtype=float)
    E_rel_profit_end = np.zeros(K, dtype=float)
    score_per_offset = np.zeros(K, dtype=float)

    break_times_now = []

    for k in range(K):
        j_end = min(k + window_len, H - 1)

        survive_flags = np.zeros(N, dtype=bool)
        good_event_flags = np.zeros(N, dtype=bool)
        rel_profit_end_s = np.zeros(N, dtype=float)

        for s in range(N):
            if k == 0:
                S_entry = float(S0)
            else:
                S_entry = float(X[s, k])

            tau_abs = profit_threshold_rel / 100.0 * S_entry

            # 1) first trend break after entry
            break_j = None
            for j in range(k + 1, j_end + 1):
                increment = X[s, j] - X[s, j - 1]
                if direction_sign * increment < 0:
                    break_j = j
                    break

            # 2) first time we hit profit threshold
            hit_j = None
            for j in range(k, j_end + 1):
                move = X[s, j] - S_entry
                if direction_sign * move >= tau_abs:
                    hit_j = j
                    break

            survive = break_j is None
            survive_flags[s] = survive

            if hit_j is not None and (break_j is None or hit_j <= break_j):
                good_event_flags[s] = True
            else:
                good_event_flags[s] = False

            final_price = float(X[s, j_end])
            rel_profit_end = direction_sign * (final_price - S_entry) / S_entry * 100.0
            rel_profit_end_s[s] = rel_profit_end

            if k == 0 and break_j is not None:
                break_times_now.append(break_j)

        p_trend_survival[k] = float(np.mean(survive_flags))
        p_profit_before_break[k] = float(np.mean(good_event_flags))
        E_rel_profit_end[k] = float(np.mean(rel_profit_end_s))
        score_per_offset[k] = p_profit_before_break[k] * max(E_rel_profit_end[k], 0.0)

    best_offset = int(np.argmax(score_per_offset))
    break_time_distribution_now = np.array(break_times_now, dtype=float) if break_times_now else np.array([])

    return {
        "best_offset": best_offset,
        "score_per_offset": score_per_offset,
        "p_trend_survival": p_trend_survival,
        "p_profit_before_break": p_profit_before_break,
        "E_rel_profit_end": E_rel_profit_end,
        "break_time_distribution_now": break_time_distribution_now,
        "direction": direction,
        "analysis_params": {
            "profit_threshold_rel": profit_threshold_rel,
            "window_len": window_len,
            "max_entry_delay": max_entry_delay,
        },
    }


# ---------------------------------------------------------------------------
# 4. Minimal smoke tests (can be removed in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test for GoodEntryHybrid
    prices = np.linspace(100.0, 110.0, 240)
    horizon = 20
    # Toy ensemble: upward trend + small noise
    base = np.linspace(prices[-1], prices[-1] + 2.0, horizon)
    noise = np.random.normal(scale=0.2, size=(50, horizon))
    X = base + noise

    snr_example = 2.0
    entry_res = GoodEntryHybrid(
        prices_window=prices,
        forecast_paths=X,
        snr=snr_example,
        profit_threshold_rel=0.2,
        prob_threshold=0.6,
        n_bootstrap_final=200,
        n_bootstrap_evidence=100,
        block_size=None,
        evidence_boundary=1.5,
        random_state=42,
    )

    print("Hybrid entry smoke test:")
    for k, v in entry_res.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape}")
        else:
            print(f"{k}: {v}")

    if entry_res["decision"] in ("enter-long", "enter-short"):
        timing_res = analyze_entry_timing(
            forecast_paths=X,
            direction=entry_res["decision"],
            S0=entry_res["S0"],
            profit_threshold_rel=0.2,
            window_len=5,
            max_entry_delay=5,
        )
        print("\nTiming analysis smoke test:")
        for k, v in timing_res.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: shape={v.shape}, values={v}")
            else:
                print(f"{k}: {v}")
