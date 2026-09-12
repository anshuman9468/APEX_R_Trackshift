import datetime as dt
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_pipeline as p


class Phase2SafetyTests(unittest.TestCase):
    def test_archive_and_remote_path_guards(self):
        self.assertTrue(p.safe_remote_path("Bahrain Grand Prix/Race/weather.json"))
        self.assertFalse(p.safe_remote_path("../weather.json"))
        self.assertFalse(p.safe_remote_path("/absolute.json"))
        with self.assertRaises(RuntimeError):
            p.assert_not_holdout("11353")

    def test_backward_join_does_not_use_future(self):
        rows = [
            {"source_time_sec": 10.0, "value": "past"},
            {"source_time_sec": 20.0, "value": "future"},
        ]
        match, status, age = p.latest_backward(rows, 15.0, "source_time_sec", 60.0)
        self.assertEqual(match["value"], "past")
        self.assertEqual(status, "MATCHED")
        self.assertEqual(age, 5.0)

    def test_freshness_cutoff(self):
        match, status, age = p.latest_backward([{"source_time_sec": 1.0}], 62.0, "source_time_sec", 60.0)
        self.assertIsNone(match)
        self.assertEqual(status, "STALE")
        self.assertEqual(age, 61.0)

    def test_datetime_join_is_backward_only(self):
        rows = [
            {"source_time_utc": "2022-03-20T15:00:00+00:00", "value": "past"},
            {"source_time_utc": "2022-03-20T15:02:00+00:00", "value": "future"},
        ]
        current = dt.datetime(2022, 3, 20, 15, 1, 1, tzinfo=dt.timezone.utc)
        match, status, age = p.latest_backward_datetime(rows, current, "source_time_utc", 60.0)
        self.assertIsNone(match)
        self.assertEqual(status, "STALE")
        self.assertEqual(age, 61.0)

    def test_timestamp_tie_is_deterministic(self):
        rows = [
            {"source_time_sec": 10.0, "weather_index": 0, "value": "first"},
            {"source_time_sec": 10.0, "weather_index": 1, "value": "second"},
        ]
        match, status, age = p.latest_backward(rows, 10.0, "source_time_sec", 60.0)
        self.assertEqual(status, "MATCHED")
        self.assertEqual(age, 0.0)
        # The source parser supplies stable list order for equal timestamps.
        self.assertEqual(match["value"], "first")

    def test_same_lap_pit_event_is_not_exposed(self):
        rows = [{"driver_code": "HAM", "lap": 10, "duration_sec": 25.0}]
        result = p.prior_pit_context(rows, "HAM", 10)
        self.assertEqual(result["status"], "UNKNOWN_SAME_LAP_EVENT")
        self.assertIsNone(result["recent_flag"])

    def test_pit_context_is_strictly_prior(self):
        rows = [
            {"driver_code": "HAM", "lap": 9, "duration_sec": 25.0},
            {"driver_code": "HAM", "lap": 10, "duration_sec": 26.0},
        ]
        result = p.prior_pit_context(rows, "HAM", 10)
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["last_prior_lap"], 9)
        self.assertTrue(result["recent_flag"])
        self.assertEqual(result["last_prior_duration_sec"], 25.0)

    def test_empty_context_is_not_no_event(self):
        result = p.prior_pit_context([], "HAM", 10)
        self.assertEqual(result["status"], "SOURCE_EMPTY_UNVERIFIED")
        self.assertIsNone(result["recent_flag"])

    def test_no_car_ahead_is_explicit(self):
        result = p.prior_pit_context([], None, 10)
        self.assertEqual(result["status"], "NO_CAR_AHEAD")
        self.assertIsNone(result["recent_flag"])


if __name__ == "__main__":
    unittest.main()
