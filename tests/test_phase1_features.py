import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_phase1_tracinginsights_features.py"
SPEC = importlib.util.spec_from_file_location("phase1_features", SCRIPT)
phase1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = phase1
SPEC.loader.exec_module(phase1)


class Phase1FeatureSafetyTests(unittest.TestCase):
    def test_backward_sample_never_uses_future(self):
        value, source = phase1.backward_sample([9.8, 10.1], [0, 1], 10.0, 0.5)
        self.assertEqual(value, 0)
        self.assertEqual(source, 9.8)

    def test_backward_sample_rejects_stale_value(self):
        value, source = phase1.backward_sample([9.0], [1], 10.0, 0.5)
        self.assertIsNone(value)
        self.assertIsNone(source)

    def test_common_timing_requires_both_completions(self):
        own = {4: (90.0, 30.0), 5: (110.0, 29.0)}
        rival = {4: (89.0, 29.5), 5: (99.0, 28.8)}
        common = phase1.latest_common_completed(own, rival, 100.0)
        self.assertEqual(common[1], 4)
        self.assertAlmostEqual(common[2] - common[3], 0.5)

    def test_pit_flag_is_backward_only_and_bounded(self):
        self.assertEqual(phase1.recent_pit_flag([100.0], 99.0), (0, None))
        self.assertEqual(phase1.recent_pit_flag([100.0], 150.0), (1, 100.0))
        self.assertEqual(phase1.recent_pit_flag([100.0], 191.0), (0, 100.0))

    def test_feature_groups_expand_to_expected_columns(self):
        self.assertEqual(len(phase1.PHASE1_FEATURE_GROUPS), 8)
        self.assertEqual(len(phase1.PHASE1_COLUMNS), 11)
        self.assertEqual(len(set(phase1.PHASE1_COLUMNS)), 11)

    def test_timing_quality_rejects_pit_and_non_clear_laps(self):
        clean = {"status": "1", "pin": "None", "pout": "None", "ff1G": False, "del": False}
        self.assertEqual(phase1.timing_quality_reasons(clean), [])
        self.assertIn("pit_in_or_out_lap", phase1.timing_quality_reasons({**clean, "pout": 123.4}))
        self.assertIn("non_clear_track_status", phase1.timing_quality_reasons({**clean, "status": "451"}))

    def test_physical_timing_limits_are_documented_constants_in_build(self):
        source = SCRIPT.read_text()
        self.assertIn("0 < value <= 60", source)
        self.assertIn("0 < lap_value <= 180", source)


if __name__ == "__main__":
    unittest.main()
