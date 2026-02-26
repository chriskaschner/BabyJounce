from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from babyjounce.plots import generate_plots


class PlotGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = ROOT / "data"

    def test_generate_plots_writes_expected_svgs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "plots"
            generated = generate_plots(self.data_dir, output_dir)
            self.assertEqual(
                set(generated),
                {"speed_histogram", "dynamic_accel_histogram", "route_paths", "route_heatmap"},
            )
            for path in generated.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 500)
                self.assertEqual(path.suffix, ".svg")


if __name__ == "__main__":
    unittest.main()
