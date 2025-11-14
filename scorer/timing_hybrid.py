# scorer/timing_hybrid.py
# -*- coding: utf-8 -*-
"""
Entry timing analysis on top of hybrid ensemble-based scoring.

This module assumes:
  - You already used your hybrid DDM / bootstrap / ensemble scorer to decide
    the DIRECTION of the trade ('enter-long' or 'enter-short').
  - You have an ensemble forecast of future prices X with shape
        X.shape = (N_scenarios, horizon)
    coming from KF / EnKF.

Goal
----
Given:
  - the current price S0,
  - the forecast ensemble X,
  - a trade direction ('enter-long' or 'enter-short'),

we want to:
  1. Estimate the probability that, over the **next time window** of length
     `window_len` (e.g. 5 steps), the local trend **does NOT reverse**.
  2. Estimate the probability that a **profit threshold** (e.g. 0.2% or more)
     is hit BEFORE a local trend reversal.
  3. Scan possible entry times in the next `max_entry_delay` steps (including
     "now" = offset 0), and choose an "optimal" entry offset that maximises
     a simple objective combining:
         - probability of threshold hit before break
         - expected relative profit at the end of the window.

The analysis is purely ensemble-based:
  - No distributional form is assumed for the predictive distribution.
  - All probabilities are Monte Carlo estimates from the forecast scenarios.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np


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
        Trade direction you *intend* to take, as decided by your hybrid scorer.
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
        A dictionary with the following keys:

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
        X = X[None, :]  # promote (horizon,) -> (1, horizon)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("analyze_entry_timing: forecast_paths must have shape (N_scenarios, horizon>=2).")

    if direction not in ("enter-long", "enter-short"):
        raise ValueError("analyze_entry_timing: direction must be 'enter-long' or 'enter-short'.")

    direction_sign = 1.0 if direction == "enter-long" else -1.0

    N, H = X.shape
    # Ensure we don't look beyond the forecast horizon
    max_delay = min(max_entry_delay, H - 2)  # need at least 1 step after entry
    K = max_delay + 1  # number of candidate offsets (0..max_delay)

    p_trend_survival = np.zeros(K, dtype=float)
    p_profit_before_break = np.zeros(K, dtype=float)
    E_rel_profit_end = np.zeros(K, dtype=float)
    score_per_offset = np.zeros(K, dtype=float)

    # For distribution of break times at "now" (offset 0)
    break_times_now = []

    # Loop over candidate entry offsets
    for k in range(K):
        # j_end: last index we observe in the timing window
        j_end = min(k + window_len, H - 1)

        survive_flags = np.zeros(N, dtype=bool)
        good_event_flags = np.zeros(N, dtype=bool)
        rel_profit_end_s = np.zeros(N, dtype=float)

        for s in range(N):
            # Entry price for this scenario and offset:
            # - offset 0: use the true current price S0;
            # - offsets >0: use the forecasted price at step k.
            if k == 0:
                S_entry = float(S0)
            else:
                S_entry = float(X[s, k])

            # Threshold in absolute terms for this scenario
            tau_abs = profit_threshold_rel / 100.0 * S_entry

            # 1) Detect first trend break after entry
            break_j = None
            for j in range(k + 1, j_end + 1):
                increment = X[s, j] - X[s, j - 1]
                if direction_sign * increment < 0:
                    break_j = j
                    break

            # 2) Detect first time we hit the profit threshold
            hit_j = None
            for j in range(k, j_end + 1):
                move = X[s, j] - S_entry
                if direction_sign * move >= tau_abs:
                    hit_j = j
                    break

            # 3) Trend survival over the whole window
            #    -> no break between k+1 and j_end
            survive = break_j is None
            survive_flags[s] = survive

            # 4) Profit threshold hit before break (and within window)
            if hit_j is not None and (break_j is None or hit_j <= break_j):
                good_event_flags[s] = True
            else:
                good_event_flags[s] = False

            # 5) Relative profit at the end of the window
            final_price = float(X[s, j_end])
            rel_profit_end = direction_sign * (final_price - S_entry) / S_entry * 100.0
            rel_profit_end_s[s] = rel_profit_end

            # 6) For offset 0, record time to break if any (for diagnostics)
            if k == 0 and break_j is not None:
                break_times_now.append(break_j)  # steps ahead from now

        # Ensemble probabilities and expectation for this offset
        p_trend_survival[k] = float(np.mean(survive_flags))
        p_profit_before_break[k] = float(np.mean(good_event_flags))
        E_rel_profit_end[k] = float(np.mean(rel_profit_end_s))

        # Objective: probability of good event * expected upside
        score_per_offset[k] = p_profit_before_break[k] * max(E_rel_profit_end[k], 0.0)

    # Choose best offset according to the objective
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
# Minimal smoke test (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: simple upward-trending forecasts
    N = 100  # scenarios
    H = 20   # horizon steps

    # Build a toy ensemble: linear uptrend + some noise
    base = np.linspace(100, 105, H)
    noise = np.random.normal(scale=0.2, size=(N, H))
    X = base + noise

    S0 = 100.0

    res = analyze_entry_timing(
        forecast_paths=X,
        direction="enter-long",
        S0=S0,
        profit_threshold_rel=0.2,
        window_len=5,
        max_entry_delay=5,
    )

    print("Best entry offset:", res["best_offset"])
    print("Score per offset:", res["score_per_offset"])
    print("P(trend survives) per offset:", res["p_trend_survival"])
    print("P(hit profit before break) per offset:", res["p_profit_before_break"])
    print("E[rel profit end] per offset:", res["E_rel_profit_end"])
