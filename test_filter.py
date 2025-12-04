from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np

import unittest

from scorer.good_entry_nonparam import GoodEntryNonParam
from scorer.good_entry import GoodEntry
from scorer.hybrid_scorer import GoodEntryHybrid
from scorer.timing_hybrid import analyze_entry_timing

class Test_good_entry_nonparam(unittest.TestCase):
    def test_enter_long(self):
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
        self.assertEqual(res["decision"], "enter-long")
    
    def test_enter_short(self):
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
        self.assertEqual(res["decision"], "enter-short")

class Test_good_entry(unittest.TestCase):
    def test_enter_long(self):
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
        self.assertEqual(res["decision"], "enter-long")
    
    def test_enter_short(self):
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
        self.assertEqual(res["decision"], "enter-short")

class Test_hybrid_scorer(unittest.TestCase):
    def test_enter_long(self):# Quick smoke test for GoodEntryHybrid
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
        self.assertEqual(entry_res["ddm_signal"], "enter-long")
    
    def test_enter_short(self):
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
        self.assertEqual(entry_res["ddm_signal"], "enter-short")

class Test_timing_hybrid(unittest.TestCase):
    def test_enter_long(self):
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
        self.assertEqual(res["best_offset"], 0)
    
    def test_enter_short(self):
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
        self.assertEqual(res["best_offset"], 1)

 
if __name__ == '__main__':
    unittest.main()