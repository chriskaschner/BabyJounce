from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import generate_report


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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    report = generate_report(args.data_dir)
    if args.output is None:
        print(report)
        return 0

    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
