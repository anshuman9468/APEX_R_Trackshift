import unittest

import numpy as np
import pandas as pd

import phase4_experiment as p


class Phase4ContractTests(unittest.TestCase):
    def test_asof_order_excludes_future_records(self):
        records = {
            "AAA": {1: {"lap": 1, "position": 2, "session_end_sec": 10.0, "deleted": False}},
            "BBB": {1: {"lap": 1, "position": 1, "session_end_sec": 10.0, "deleted": False}},
            "CCC": {1: {"lap": 1, "position": 3, "session_end_sec": 20.0, "deleted": False}},
        }
        status, positions, laps, _ = p.latest_completed_order(records, 15.0)
        self.assertEqual(status, "UNIQUE_CONTIGUOUS_COMPLETED_ORDER_ASOF")
        self.assertEqual(positions, {1: "BBB", 2: "AAA"})
        self.assertNotIn("CCC", laps)

    def test_asof_target_conflict_is_not_silently_accepted(self):
        records = {
            "ATT": {1: {"lap": 1, "position": 2, "session_end_sec": 10.0, "deleted": False}},
            "EXPECTED": {1: {"lap": 1, "position": 1, "session_end_sec": 10.0, "deleted": False}},
            "OTHER": {1: {"lap": 1, "position": 3, "session_end_sec": 10.0, "deleted": False}},
        }
        example = {"attacker_driver": "ATT", "target_driver": "OTHER", "decision_time_session_sec": "15"}
        result = p.asof_target_check(example, records)
        self.assertEqual(result["asof_status"], "ASOF_TARGET_CONFLICT")

    def test_missing_feature_remains_nan_before_training_imputer(self):
        frame = pd.DataFrame({"lap": [4], "time": [0], "phase2_session_time_sec": [100], "phase2_weather_rainfall_flag": [""]})
        out = p.prepare_feature_frame(frame)
        self.assertTrue(np.isnan(out.loc[0, "phase2_weather_rainfall_flag"]))

    def test_feature_allowlist_has_no_outcome_fields(self):
        self.assertFalse(p.MODEL_FEATURES.intersection(p.OUTCOME_FIELDS))

    def test_curve_contains_terminal_recall_point(self):
        rows = p.curve_rows("m", "validation", np.array([0, 1]), np.array([0.2, 0.8]))
        self.assertTrue(any(row["recall"] == 1.0 and row["threshold"] == 0.2 for row in rows))
        self.assertTrue(any(row["recall"] == 0.0 and row["threshold"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
