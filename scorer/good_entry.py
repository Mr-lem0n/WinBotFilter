# scorer/good_entry.py
# -*- coding: utf-8 -*-
"""
Jump–diffusion based entry scoring for EnKF/KF forecast ensembles.

This module is designed to plug directly into the existing pipeline:

    from scorer.good_entry import GoodEntry, JDScenario

The main entry point is the function:

    GoodEntry(prices_window, forecast_paths, snr, ...)

which returns a dict containing:
    - decision  : 'enter-long', 'enter-short', or 'not-enter'
    - P_up      : posterior probability that the local trend is up
    - P_flat    : posterior probability that the local trend is flat/random
    - P_down    : posterior probability that the local trend is down
    - E_rel_profit_long  : expected relative profit (%) for a long entry
    - E_rel_profit_short : expected relative profit (%) for a short entry
    - S0        : last price in the window (entry price reference)
    - trend_strength_prior: how strongly priors are tilted away from "flat"
                             towards "trend" based on SNR

USAGE IN PIPELINE (simplified)
------------------------------

Inside MarketDataProcessor.run_model, when you currently have:

    smooth_Data = self.smooth_time_series(array_to_analyse, len(array_to_analyse))
    horizon = 20
    if self.ensemble > 0:
        X = self.enkf.forecast(smooth_Data, horizon, self.ensemble)
    else:
        X = self.kf.forecast(smooth_Data, horizon)[None, :]

    # ... RMT cleaning, compute cov and snr ...

Replace the old logistic "prob" scoring block with:

    entry_res = GoodEntry(
        prices_window=smooth_Data,
        forecast_paths=X,
        snr=snr,
        profit_threshold_rel=0.2,  # 0.2% min expected relative profit
        prob_threshold=0.6,
    )
    self.signal = entry_res["decision"]

    print("Entry scoring result:")
    print(f"  P_up   = {entry_res['P_up']:.3f}")
    print(f"  P_flat = {entry_res['P_flat']:.3f}")
    print(f"  P_down = {entry_res['P_down']:.3f}")
    print(f"  E_rel_profit_long  = {entry_res['E_rel_profit_long']:.3f}%")
    print(f"  E_rel_profit_short = {entry_res['E_rel_profit_short']:.3f}%")
    print(f"  decision = {self.signal}")

Everything else in the pipeline (exit logic, CSV logging) can stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. Scenario specification for the jump–diffusion model
# ---------------------------------------------------------------------------

@dataclass
class JDScenario:
    """
    Parameters of one jump–diffusion scenario.

    We model the log price X_t via:

        dX_t = mu dt + sigma dW_t + J dN_t

    where:
        - mu        : drift under the given regime (up / down / flat)
        - sigma     : diffusion volatility
        - N_t       : Poisson process with intensity lambda_jump
        - J         : jump size (i.i.d.) with mean jump_mean, var jump_var

    The scenario contains:
        - mu_up     : drift parameter under an "up" regime (positive)
        - mu_down   : drift parameter under a "down" regime (negative)
        - mu_flat   : drift parameter under a "flat/random" regime (~0)
        - sigma     : diffusion volatility
        - lambda_jump : jump intensity per unit time
        - jump_mean : mean jump size in log-price space
        - jump_var  : variance of jump size in log-price space
        - weight    : weight of this scenario in the ensemble
                      (used to compute ensemble averages over scenarios)
    """
    mu_up: float
    mu_down: float
    mu_flat: float
    sigma: float
    lambda_jump: float
    jump_mean: float
    jump_var: float
    weight: float = 1.0


# ---------------------------------------------------------------------------
# 2. Jump–diffusion DDM ensemble for local trend classification
# ---------------------------------------------------------------------------

class JumpDDMEnsemble:
    """
    Jump–diffusion based Drift–Diffusion Model (DDM) on a price window.

    Given a vector of historical prices and an ensemble of JDScenario objects,
    this class infers the posterior probabilities that the *local* trend is:
        - up
        - flat/random
        - down

    The core idea:
        - Work in log-price space: X_t = log S_t.
        - Compute log-returns r_k = X_{k} - X_{k-1}.
        - Assume that under each regime h ∈ {up, flat, down} and scenario s,
          the increments are approximately Gaussian with:
              r_k ~ N(mu_eff_h^{(s)} * dt, sigma_eff_sq^{(s)} * dt),

          where mu_eff and sigma_eff_sq incorporate the jump component.
        - Compute the log-likelihood of the observed returns under each
          regime, combine with regime priors, and normalise to obtain
          posterior probabilities.

    Attributes
    ----------
    prices : np.ndarray
        Historical price series (window).
    scenarios : List[JDScenario]
        Jump–diffusion scenarios.
    dt : float
        Time step between aggregated prices (in seconds, or consistent units).
    window_size : Optional[int]
        Number of returns used; if None, use all.
    prior_regimes : Dict[str, float]
        Prior probabilities for regimes {'up','flat','down'}.
    """

    def __init__(
        self,
        prices: np.ndarray,
        scenarios: List[JDScenario],
        dt: float = 1.0,
        window_size: Optional[int] = None,
        prior_regimes: Optional[Dict[str, float]] = None,
    ):
        self.prices = np.asarray(prices, dtype=float)
        if self.prices.ndim != 1 or len(self.prices) < 3:
            raise ValueError("JumpDDMEnsemble: prices must be a 1D array with length >= 3.")

        self.scenarios = scenarios
        if len(self.scenarios) == 0:
            raise ValueError("JumpDDMEnsemble: at least one JDScenario must be provided.")

        self.dt = float(dt)
        self.window_size = window_size or (len(self.prices) - 1)

        if prior_regimes is None:
            # Default: uniform prior over up, flat, down
            prior_regimes = {"up": 1.0 / 3.0, "flat": 1.0 / 3.0, "down": 1.0 / 3.0}
        self.prior_regimes = prior_regimes

    # ---------- Internal helpers ----------

    def _prepare_returns(self) -> np.ndarray:
        """
        Compute log-returns from the price window, and keep only the last
        'window_size' increments.

        Returns
        -------
        np.ndarray
            1D array of log-returns.
        """
        log_p = np.log(self.prices)
        r = np.diff(log_p)
        if len(r) <= self.window_size:
            return r
        return r[-self.window_size:]

    @staticmethod
    def _loglik_gaussian(r: np.ndarray, mean: float, var: float) -> np.ndarray:
        """
        Compute log-likelihood of Gaussian increments for a batch of returns.

        Parameters
        ----------
        r : np.ndarray
            Observed returns.
        mean : float
            Mean of the Gaussian.
        var : float
            Variance of the Gaussian (must be > 0).

        Returns
        -------
        np.ndarray
            Log-likelihood values per observation.
        """
        eps = 1e-12
        var_safe = max(var, eps)
        return -0.5 * (np.log(2.0 * np.pi * var_safe) + (r - mean) ** 2 / var_safe)

    # ---------- Public API ----------

    def run(self) -> Dict[str, float]:
        """
        Run the jump–diffusion DDM and return posterior regime probabilities.

        Returns
        -------
        dict
            Dictionary with keys:
                - "P_up"
                - "P_flat"
                - "P_down"
        """
        r = self._prepare_returns()
        dt = self.dt

        # Prior probabilities over regimes
        pi_up = float(self.prior_regimes["up"])
        pi_flat = float(self.prior_regimes["flat"])
        pi_down = float(self.prior_regimes["down"])

        # Ensemble probabilities (weighted over scenarios)
        P_up = 0.0
        P_flat = 0.0
        P_down = 0.0

        weights = np.array([sc.weight for sc in self.scenarios], dtype=float)
        weights /= weights.sum()

        for w, sc in zip(weights, self.scenarios):
            # Effective drift under each regime (mu + jump contribution)
            mu_eff_up = sc.mu_up + sc.lambda_jump * sc.jump_mean
            mu_eff_flat = sc.mu_flat + sc.lambda_jump * sc.jump_mean
            mu_eff_down = sc.mu_down + sc.lambda_jump * sc.jump_mean

            # Effective variance per unit time:
            # sigma_eff^2 = sigma^2 + lambda * (Var(J) + (E[J])^2)
            sigma_eff_sq = sc.sigma ** 2 + sc.lambda_jump * (sc.jump_var + sc.jump_mean ** 2)

            mean_up = mu_eff_up * dt
            mean_flat = mu_eff_flat * dt
            mean_down = mu_eff_down * dt
            var = sigma_eff_sq * dt

            # Log-likelihood of the observed returns under each regime
            ll_up = float(np.sum(self._loglik_gaussian(r, mean_up, var)))
            ll_flat = float(np.sum(self._loglik_gaussian(r, mean_flat, var)))
            ll_down = float(np.sum(self._loglik_gaussian(r, mean_down, var)))

            # Unnormalised log-posteriors (log prior + log likelihood)
            log_post_up = np.log(pi_up) + ll_up
            log_post_flat = np.log(pi_flat) + ll_flat
            log_post_down = np.log(pi_down) + ll_down

            # Stabilised normalisation via log-sum-exp
            m = max(log_post_up, log_post_flat, log_post_down)
            e_up = np.exp(log_post_up - m)
            e_flat = np.exp(log_post_flat - m)
            e_down = np.exp(log_post_down - m)
            Z = e_up + e_flat + e_down

            post_up = e_up / Z
            post_flat = e_flat / Z
            post_down = e_down / Z

            # Weight by scenario weight in the ensemble
            P_up += w * post_up
            P_flat += w * post_flat
            P_down += w * post_down

        # Final normalisation (should be close to 1 already)
        s = P_up + P_flat + P_down
        if s <= 0:
            # Degenerate safeguard: fall back to uniform
            return {"P_up": 1.0 / 3.0, "P_flat": 1.0 / 3.0, "P_down": 1.0 / 3.0}

        return {
            "P_up": P_up / s,
            "P_flat": P_flat / s,
            "P_down": P_down / s,
        }


# ---------------------------------------------------------------------------
# 3. Main entry scoring function: GoodEntry
# ---------------------------------------------------------------------------

def GoodEntry(
    prices_window: np.ndarray,
    forecast_paths: np.ndarray,
    snr: float = 1.0,
    profit_threshold_rel: float = 0.2,
    prob_threshold: float = 0.6,
    trade_horizon: Optional[int] = None,
    jd_scenarios: Optional[List[JDScenario]] = None,
) -> Dict[str, float]:
    """
    Entry scoring using a jump–diffusion DDM and an ensemble of forecasts.

    This function is designed to act as a drop-in replacement for the old
    "prob > 0.55" scoring logic. It produces the same type of signal:

        - 'enter-long'
        - 'enter-short'
        - 'not-enter'

    but it is grounded in:
        - a jump–diffusion DDM on the *historical* price window to classify
          the local trend as up / flat / down,
        - an ensemble of future price paths (from KF/EnKF) to compute the
          expected relative profit of long vs short trades,
        - a minimum profit threshold (in %) to filter out weak signals.

    Parameters
    ----------
    prices_window : array-like
        Historical price window (e.g. smoothed offer prices for the last 240s).
        1D array of length >= 3.
    forecast_paths : array-like
        Forecasted price paths from KF/EnKF.
        Shape:
            - (N_scenarios, horizon)
            - or (horizon,) for a single scenario (will be promoted to 2D).
        Only the last step of each path is used to estimate expected profit.
    snr : float, optional
        Signal-to-noise ratio derived from your RMT covariance cleaning.
        Used to tilt priors away from "flat" towards "trend".
        - snr ~ 0 => prior heavily on "flat"
        - snr large => prior mass shifts towards "up"/"down"
    profit_threshold_rel : float, optional
        Minimum expected **relative** profit (%) required to trigger a trade.
        Typical use here: 0.2 => require at least 0.2% expected profit.
    prob_threshold : float, optional
        Minimum posterior probability for the corresponding trend regime to
        commit to long/short (e.g. 0.6 => require P_up >= 0.6 for a long).
    trade_horizon : int or None, optional
        Number of steps ahead over which the prediction is made.
        If None, inferred as forecast_paths.shape[1]. Currently used only
        for documentation purposes; the expected profit is derived from the
        last column of forecast_paths.
    jd_scenarios : list of JDScenario or None, optional
        If provided, these scenarios are used in the DDM. Otherwise, a single
        scenario is constructed from the empirical volatility of the window.

    Returns
    -------
    dict
        Dictionary with entries:
            - "decision"            : 'enter-long' / 'enter-short' / 'not-enter'
            - "P_up"                : posterior probability of up-trend
            - "P_flat"              : posterior probability of flat/random
            - "P_down"              : posterior probability of down-trend
            - "E_rel_profit_long"   : expected relative profit (%) for long
            - "E_rel_profit_short"  : expected relative profit (%) for short
            - "S0"                  : last price in the window (entry price)
            - "trend_strength_prior": scalar in (0,1) describing how much
                                      prior mass has been moved from "flat"
                                      to "trend" based on snr.
    """
    # ---------- Input preparation ----------
    prices_window = np.asarray(prices_window, dtype=float)
    if prices_window.ndim != 1 or len(prices_window) < 3:
        raise ValueError("GoodEntry: prices_window must be 1D with length >= 3.")

    X = np.asarray(forecast_paths, dtype=float)
    if X.ndim == 1:
        # Promote 1D horizon vector to (1, horizon)
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("GoodEntry: forecast_paths must have shape (N_scenarios, horizon).")

    if trade_horizon is None:
        trade_horizon = X.shape[1]

    # ---------- Build JD scenarios if none were provided ----------
    if jd_scenarios is None:
        # Crude empirical calibration from last returns.
        log_p = np.log(prices_window)
        r = np.diff(log_p)
        sigma_hat = float(np.std(r)) if np.std(r) > 0 else 1e-6

        # Choose mu_up/down as +/- k * sigma_hat per unit time.
        # k_mu is a "trend strength" parameter you can calibrate.
        k_mu = 0.5
        mu_up = k_mu * sigma_hat      # positive drift
        mu_down = -k_mu * sigma_hat   # negative drift
        mu_flat = 0.0                 # flat regime drift

        # Simple jump parameters (can be replaced by your own estimates):
        lambda_jump = 0.05             # small jump intensity
        jump_mean = 0.0                # symmetric jumps
        jump_var = (0.5 * sigma_hat) ** 2

        jd_scenarios = [
            JDScenario(
                mu_up=mu_up,
                mu_down=mu_down,
                mu_flat=mu_flat,
                sigma=sigma_hat,
                lambda_jump=lambda_jump,
                jump_mean=jump_mean,
                jump_var=jump_var,
                weight=1.0,
            )
        ]

    # ---------- Regime priors from SNR (tilting away from "flat") ----------
    # trend_strength ∈ (0,1):
    #   - near 0 => almost fully flat prior
    #   - near 1 => equally split between up and down, little flat.
    alpha_snr = 0.7  # tuning parameter; larger => more sensitive to snr
    trend_strength = 1.0 - np.exp(-alpha_snr * float(snr))
    pi_flat = 1.0 - trend_strength
    pi_up = trend_strength / 2.0
    pi_down = trend_strength / 2.0
    prior_regimes = {"up": pi_up, "flat": pi_flat, "down": pi_down}

    # ---------- DDM on historical prices: P_up / P_flat / P_down ----------
    ddm = JumpDDMEnsemble(
        prices=prices_window,
        scenarios=jd_scenarios,
        dt=1.0,  # 1 second per aggregated quote (consistent with your pipeline)
        window_size=min(len(prices_window) - 1, 120),  # last ~120 seconds
        prior_regimes=prior_regimes,
    )
    ddm_res = ddm.run()
    P_up = float(ddm_res["P_up"])
    P_flat = float(ddm_res["P_flat"])
    P_down = float(ddm_res["P_down"])

    # ---------- Expected relative profit from ensemble forecasts ----------
    # Use the *last* value of each forecast path as the forecasted price
    # at the end of the trade horizon, and compare to current price S0.
    S0 = float(prices_window[-1])
    future_prices = X[:, -1]  # shape (N_scenarios,)

    # Relative profits in the same units as your CSV logging:
    #   rel_profit = (absolute_profit * 100) / entry_price
    rel_long_s = (future_prices - S0) / S0 * 100.0   # long: price up is profit
    rel_short_s = (S0 - future_prices) / S0 * 100.0  # short: price down is profit

    E_rel_long = float(np.mean(rel_long_s))
    E_rel_short = float(np.mean(rel_short_s))

    # ---------- Decision rule with thresholds ----------
    decision = "not-enter"

    # Conditions to consider long/short entries:
    long_ok = (E_rel_long >= profit_threshold_rel) and (P_up >= prob_threshold)
    short_ok = (E_rel_short >= profit_threshold_rel) and (P_down >= prob_threshold)

    if long_ok and not short_ok:
        decision = "enter-long"
    elif short_ok and not long_ok:
        decision = "enter-short"
    elif long_ok and short_ok:
        # Both directions look good: choose the higher expected profit
        if E_rel_long >= E_rel_short:
            decision = "enter-long"
        else:
            decision = "enter-short"
    else:
        decision = "not-enter"

    # ---------- Return all diagnostics ----------
    return {
        "decision": decision,
        "P_up": P_up,
        "P_flat": P_flat,
        "P_down": P_down,
        "E_rel_profit_long": E_rel_long,
        "E_rel_profit_short": E_rel_short,
        "S0": S0,
        "trend_strength_prior": trend_strength,
    }


# ---------------------------------------------------------------------------
# 4. Minimal self-test (can be removed in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple sanity check: monotonically increasing prices + single forecast
    prices = np.linspace(100.0, 110.0, 240)  # window
    horizon = 20
    # Single forecast path: continue the trend linearly
    forecast = np.linspace(prices[-1], prices[-1] + 2.0, horizon)  # ends slightly higher
    snr_example = 2.0

    res = GoodEntry(
        prices_window=prices,
        forecast_paths=forecast,
        snr=snr_example,
        profit_threshold_rel=0.2,
        prob_threshold=0.6,
    )

    print("Self-test result (increasing prices):")
    for k, v in res.items():
        print(f"{k}: {v}")
