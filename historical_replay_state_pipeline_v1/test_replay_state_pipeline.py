"""Focused invariant tests for the strict historical replay-state contract."""

import unittest

from replay_state_pipeline import PROTECTED_TOKEN, strict_optimizer_view


class ReplayStateContractTests(unittest.TestCase):
    def test_protected_token_is_explicit(self):
        self.assertEqual(PROTECTED_TOKEN, "11353")

    def test_optimizer_refuses_xyz_as_seconds_gap(self):
        graph = {"pair_raw": [1.0, 312.5]}
        view = strict_optimizer_view(graph, None, None)
        self.assertEqual(view["status"], "unavailable")
        self.assertIsNone(view["inputs"]["front_gap_seconds"])
        self.assertEqual(view["inputs"]["attacker_target_xyz_euclidean_distance_m"], 312.5)
        self.assertEqual(view["action_scores"]["ATTACK"]["score"], None)

    def test_action_is_not_forced_when_required_state_is_missing(self):
        graph = {"pair_raw": [0.0, float("nan")]}
        view = strict_optimizer_view(graph, {"status": "available"}, None)
        self.assertIsNone(view["action"])
        self.assertEqual(view["decision_reason"], "MISSING_SOURCE_BACKED_FRONT_AND_REAR_TIME_GAPS_SECONDS")


if __name__ == "__main__":
    unittest.main()
