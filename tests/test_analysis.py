from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from babyjounce.analysis import analyze_data, generate_report


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = ROOT / "data"

    def test_known_dataset_sizes(self) -> None:
        result = analyze_data(self.data_dir)
        self.assertEqual(result.summaries["walking"].rows, 32410)
        self.assertEqual(result.summaries["running"].rows, 16764)
        self.assertEqual(result.summaries["driving"].rows, 5047)

    def test_speed_baselines_are_strong(self) -> None:
        result = analyze_data(self.data_dir)
        self.assertGreaterEqual(result.driving_vs_non_driving_by_speed.accuracy, 0.97)
        self.assertGreaterEqual(result.running_vs_walking_by_speed.accuracy, 0.90)

    def test_report_contains_sections(self) -> None:
        report = generate_report(self.data_dir)
        self.assertIn("# BabyJounce v2 Summary", report)
        self.assertIn("## Dataset Stats", report)
        self.assertIn("## Research-Informed Comparison (Not Clinical Limits)", report)
        self.assertIn("## Simple Baselines", report)

    def test_research_comparison_panel_values(self) -> None:
        result = analyze_data(self.data_dir)
        self.assertEqual(result.summaries["walking"].dynamic_band, "Moderate")
        self.assertEqual(result.summaries["running"].dynamic_band, "Moderate")
        self.assertEqual(result.summaries["driving"].dynamic_band, "Low")
        self.assertGreater(result.summaries["running"].dynamic_accel_mps2.rms, 2.0)
        self.assertLess(result.summaries["driving"].dynamic_accel_mps2.rms, 1.1)


if __name__ == "__main__":
    unittest.main()
