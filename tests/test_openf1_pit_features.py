import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_phase1_openf1_pit_features.py"
SPEC = importlib.util.spec_from_file_location("openf1_pit_features", SCRIPT)
pit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pit
SPEC.loader.exec_module(pit)


class OpenF1PitCausalityTests(unittest.TestCase):
    def test_latest_event_is_backward_only(self):
        events = pit.prepare_events([
            {"driver_number": 1, "date": "2023-01-01T00:00:10+00:00", "lane_duration": 22.0},
        ])[1]
        self.assertIsNone(pit.latest_pit_event(events, pit.timestamp("2023-01-01T00:00:09+00:00")))
        self.assertEqual(pit.latest_pit_event(events, pit.timestamp("2023-01-01T00:00:10+00:00"))["lane_duration_sec"], 22.0)

    def test_recent_flag_is_bounded(self):
        events = pit.prepare_events([
            {"driver_number": 1, "date": "2023-01-01T00:00:00+00:00", "lane_duration": 22.0},
        ])[1]
        event_time = pit.timestamp("2023-01-01T00:00:00+00:00")
        self.assertEqual(pit.pit_value(events, event_time + 60.0)[0], 1.0)
        self.assertEqual(pit.pit_value(events, event_time + 91.0)[0], 0.0)

    def test_missing_endpoint_does_not_become_zero(self):
        import pandas as pd

        frame = pd.DataFrame([{
            "session_key": 9070,
            "driver_number": 1,
            "car_ahead_driver_number": 2,
            "date": "2023-01-01T00:01:00+00:00",
        }])
        output, reasons, _ = pit.add_pit_features(frame, 9070, [], {"available": False, "status": "http_404_no_records"})
        self.assertTrue(output[pit.PIT_FEATURES].isna().all().all())
        self.assertEqual(reasons["attacker_openf1_pit_recent_flag"]["pit_endpoint_unavailable_http_404_no_records"], 1)

    def test_rival_without_car_ahead_is_explicitly_missing(self):
        import pandas as pd

        frame = pd.DataFrame([{
            "session_key": 9070,
            "driver_number": 1,
            "car_ahead_driver_number": pd.NA,
            "date": "2023-01-01T00:01:00+00:00",
        }])
        output, reasons, _ = pit.add_pit_features(frame, 9070, [], {"available": True, "status": "fetched"})
        self.assertTrue(pd.isna(output.iloc[0]["rival_openf1_pit_recent_flag"]))
        self.assertEqual(reasons["rival_openf1_pit_recent_flag"]["no_car_ahead_identity"], 1)


if __name__ == "__main__":
    unittest.main()
