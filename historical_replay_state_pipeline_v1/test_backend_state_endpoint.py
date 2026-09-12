"""Read-only checks for strict replay-state serving."""

import unittest

from backend.phase5 import Phase5Service


class StrictHistoricalStateEndpointTests(unittest.TestCase):
    def test_serves_audited_state_without_action(self):
        service = Phase5Service()
        result = service.historical_state("ee38ba6e51b8d9dbba9b")
        self.assertEqual(result["status"], "available")
        state = result["state"]
        self.assertEqual(state["model_applicability"], "APPLICABLE")
        self.assertEqual(state["optimizer"]["status"], "unavailable")
        self.assertIsNone(state["optimizer"]["action"])
        self.assertIsNone(state["optimizer"]["inputs"]["front_gap_seconds"])

    def test_rejects_protected_token_before_state_access(self):
        service = Phase5Service()
        with self.assertRaises(ValueError):
            service.historical_state("11353")


if __name__ == "__main__":
    unittest.main()
