from __future__ import annotations

import argparse
import os
from pathlib import Path

from .analysis import generate_report
from .plots import generate_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="babyjounce",
        description="Generate reproducible summaries from BabyJounce motion datasets.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing walking.csv, running.csv, and driving.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path for the generated Markdown report",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Optional directory to write SVG plots and embed in the report",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    plots_dir: Path | None = args.plots_dir
    if plots_dir is None and args.output is not None:
        plots_dir = args.output.parent / "plots"

    plot_paths_for_report: dict[str, str] | None = None
    if plots_dir is not None:
        generated_paths = generate_plots(args.data_dir, plots_dir)
        base_dir = args.output.parent if args.output is not None else Path.cwd()
        plot_paths_for_report = {}
        for key, value in generated_paths.items():
            relative_path = Path(os.path.relpath(value, start=base_dir))
            plot_paths_for_report[key] = relative_path.as_posix()

    report = generate_report(args.data_dir, plot_paths=plot_paths_for_report)
    if args.output is None:
        print(report)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")
    if plots_dir is not None:
        print(f"Wrote plots to {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
