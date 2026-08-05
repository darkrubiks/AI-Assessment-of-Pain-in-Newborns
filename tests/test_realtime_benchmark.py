from __future__ import annotations

import unittest

import numpy as np

from benchmarking.realtime import (
    ProbabilitySmoother,
    _select_config,
    latency_stats,
    merge_rgu_xai_masks,
)


class RealtimeBenchmarkUnitTests(unittest.TestCase):
    def test_latency_stats(self) -> None:
        stats = latency_stats([1.0, 2.0, 3.0, float("nan")])
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["median"], 2.0)
        self.assertGreaterEqual(stats["p95"], 2.0)

    def test_causal_smoothing_does_not_use_future_values(self) -> None:
        smoother = ProbabilitySmoother(window=3)
        self.assertEqual(smoother.update(0.0, 0.0), (0.0, 0.0))
        probability, uncertainty = smoother.update(1.0, 0.2)
        self.assertAlmostEqual(probability, 0.5)
        self.assertAlmostEqual(uncertainty, 0.1)
        probability, _ = smoother.update(1.0, 0.4)
        self.assertAlmostEqual(probability, 2.0 / 3.0)

    def test_disabled_expensive_scenario_can_be_selected_explicitly(self) -> None:
        config = {
            "source": {"kind": "camera"},
            "scenarios": [
                {"name": "base", "enabled": True},
                {"name": "full", "enabled": False},
            ],
        }
        selected = _select_config(config, ["full"])
        self.assertEqual([item["name"] for item in selected["scenarios"]], ["full"])
        self.assertTrue(selected["scenarios"][0]["enabled"])

    def test_rgu_merge_normalizes_each_method_then_averages(self) -> None:
        first = np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
             [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
            dtype=np.float64,
        )
        second = np.array(
            [[[3.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
             [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=np.float64,
        )
        merged = merge_rgu_xai_masks([first, second])
        np.testing.assert_allclose(merged, np.full((2, 2), 0.5), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
