from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f %z"
GRAVITY_MPS2 = 9.80665
RESEARCH_BAND_LOW_RMS_MPS2 = 1.1
RESEARCH_BAND_MODERATE_RMS_MPS2 = 2.7
DATASET_FILES = {
    "walking": "walking.csv",
    "running": "running.csv",
    "driving": "driving.csv",
}


@dataclass(frozen=True)
class Record:
    label: str
    speed_mps: float
    accel_x_g: float
    accel_y_g: float
    accel_z_g: float
    latitude: float
    longitude: float
    timestamp: datetime | None
    activity: str


@dataclass(frozen=True)
class NumericSummary:
    mean: float
    rms: float
    stddev: float
    minimum: float
    maximum: float
    q50: float
    q95: float
    q99: float


@dataclass(frozen=True)
class DatasetSummary:
    label: str
    rows: int
    duration_minutes: float
    speed: NumericSummary
    accel_y: NumericSummary
    accel_norm: NumericSummary
    jerk_y_abs: NumericSummary
    dynamic_accel_mps2: NumericSummary
    dynamic_pct_ge_2_mps2: float
    dynamic_pct_ge_3_mps2: float
    dynamic_pct_ge_4_mps2: float
    dynamic_band: str
    top_activities: list[tuple[str, int]]


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    accuracy: float


@dataclass(frozen=True)
class AnalysisResult:
    summaries: dict[str, DatasetSummary]
    driving_vs_non_driving_by_speed: ThresholdResult
    running_vs_walking_by_speed: ThresholdResult


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except ValueError:
        return None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    index = int(p * (len(values) - 1))
    return values[index]


def summarize_numeric(values: Iterable[float]) -> NumericSummary:
    vals = list(values)
    if not vals:
        nan = float("nan")
        return NumericSummary(nan, nan, nan, nan, nan, nan, nan, nan)

    count = len(vals)
    mean = sum(vals) / count
    rms = math.sqrt(sum(value**2 for value in vals) / count)
    variance = (
        sum((value - mean) ** 2 for value in vals) / (count - 1) if count > 1 else 0.0
    )
    sorted_vals = sorted(vals)
    return NumericSummary(
        mean=mean,
        rms=rms,
        stddev=math.sqrt(variance),
        minimum=sorted_vals[0],
        maximum=sorted_vals[-1],
        q50=percentile(sorted_vals, 0.50),
        q95=percentile(sorted_vals, 0.95),
        q99=percentile(sorted_vals, 0.99),
    )


def load_records(csv_path: Path, label: str) -> list[Record]:
    records: list[Record] = []
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                speed = float(row["locationSpeed(m/s)"])
                accel_x = float(row["accelerometerAccelerationX(G)"])
                accel_y = float(row["accelerometerAccelerationY(G)"])
                accel_z = float(row["accelerometerAccelerationZ(G)"])
                latitude = float(row["locationLatitude(WGS84)"])
                longitude = float(row["locationLongitude(WGS84)"])
            except (KeyError, ValueError):
                continue

            records.append(
                Record(
                    label=label,
                    speed_mps=speed,
                    accel_x_g=accel_x,
                    accel_y_g=accel_y,
                    accel_z_g=accel_z,
                    latitude=latitude,
                    longitude=longitude,
                    timestamp=parse_timestamp(row.get("loggingTime(txt)", "")),
                    activity=row.get("activity(txt)", ""),
                )
            )
    return records


def classify_dynamic_band(rms_dynamic_mps2: float) -> str:
    if rms_dynamic_mps2 < RESEARCH_BAND_LOW_RMS_MPS2:
        return "Low"
    if rms_dynamic_mps2 <= RESEARCH_BAND_MODERATE_RMS_MPS2:
        return "Moderate"
    return "Elevated"


def percentage_at_or_above(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return (sum(1 for value in values if value >= threshold) / len(values)) * 100.0


def summarize_records(label: str, records: list[Record]) -> DatasetSummary:
    timestamps = [record.timestamp for record in records if record.timestamp is not None]
    duration_minutes = 0.0
    if timestamps:
        duration_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60.0

    speeds = [record.speed_mps for record in records]
    accel_y = [record.accel_y_g for record in records]
    accel_norm = [
        math.sqrt(record.accel_x_g**2 + record.accel_y_g**2 + record.accel_z_g**2)
        for record in records
    ]
    dynamic_accel_mps2 = [abs(norm - 1.0) * GRAVITY_MPS2 for norm in accel_norm]

    jerk_y_abs: list[float] = []
    prev_timestamp: datetime | None = None
    prev_accel_y: float | None = None
    for record in records:
        if prev_timestamp is not None and record.timestamp is not None and prev_accel_y is not None:
            dt = (record.timestamp - prev_timestamp).total_seconds()
            if dt > 0:
                jerk_y_abs.append(abs(record.accel_y_g - prev_accel_y) / dt)
        if record.timestamp is not None:
            prev_timestamp = record.timestamp
        prev_accel_y = record.accel_y_g

    top_activities = Counter(record.activity for record in records).most_common(5)
    dynamic_summary = summarize_numeric(dynamic_accel_mps2)

    return DatasetSummary(
        label=label,
        rows=len(records),
        duration_minutes=duration_minutes,
        speed=summarize_numeric(speeds),
        accel_y=summarize_numeric(accel_y),
        accel_norm=summarize_numeric(accel_norm),
        jerk_y_abs=summarize_numeric(jerk_y_abs),
        dynamic_accel_mps2=dynamic_summary,
        dynamic_pct_ge_2_mps2=percentage_at_or_above(dynamic_accel_mps2, 2.0),
        dynamic_pct_ge_3_mps2=percentage_at_or_above(dynamic_accel_mps2, 3.0),
        dynamic_pct_ge_4_mps2=percentage_at_or_above(dynamic_accel_mps2, 4.0),
        dynamic_band=classify_dynamic_band(dynamic_summary.rms),
        top_activities=top_activities,
    )


def best_speed_threshold(
    records: list[Record], positive_labels: set[str], threshold_step: float, threshold_max: float
) -> ThresholdResult:
    best_threshold = 0.0
    best_accuracy = 0.0
    threshold = 0.0
    while threshold <= threshold_max:
        correct = 0
        for record in records:
            prediction = record.speed_mps >= threshold
            is_positive = record.label in positive_labels
            if prediction == is_positive:
                correct += 1
        accuracy = correct / len(records) if records else 0.0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
        threshold += threshold_step
    return ThresholdResult(threshold=best_threshold, accuracy=best_accuracy)


def analyze_data(data_dir: Path) -> AnalysisResult:
    by_label: dict[str, list[Record]] = {}
    for label, filename in DATASET_FILES.items():
        by_label[label] = load_records(data_dir / filename, label=label)

    summaries = {label: summarize_records(label, records) for label, records in by_label.items()}

    all_records = [record for records in by_label.values() for record in records]
    driving_threshold = best_speed_threshold(
        records=all_records,
        positive_labels={"driving"},
        threshold_step=0.1,
        threshold_max=20.0,
    )

    run_walk_records = by_label["running"] + by_label["walking"]
    running_threshold = best_speed_threshold(
        records=run_walk_records,
        positive_labels={"running"},
        threshold_step=0.01,
        threshold_max=8.0,
    )

    return AnalysisResult(
        summaries=summaries,
        driving_vs_non_driving_by_speed=driving_threshold,
        running_vs_walking_by_speed=running_threshold,
    )


def format_summary(name: str, summary: DatasetSummary) -> str:
    activity_line = ", ".join(f"{activity}:{count}" for activity, count in summary.top_activities)
    return "\n".join(
        [
            f"- {name}:",
            f"  rows={summary.rows}",
            f"  duration_min={summary.duration_minutes:.1f}",
            (
                "  speed_mps "
                f"mean={summary.speed.mean:.3f} std={summary.speed.stddev:.3f} "
                f"min={summary.speed.minimum:.3f} max={summary.speed.maximum:.3f}"
            ),
            (
                "  accelY_g "
                f"mean={summary.accel_y.mean:.3f} std={summary.accel_y.stddev:.3f} "
                f"q95={summary.accel_y.q95:.3f} q99={summary.accel_y.q99:.3f}"
            ),
            (
                "  jerkY_abs_gps "
                f"q50={summary.jerk_y_abs.q50:.2f} q95={summary.jerk_y_abs.q95:.2f} "
                f"q99={summary.jerk_y_abs.q99:.2f}"
            ),
            f"  top_activity={activity_line}",
        ]
    )


def generate_report(data_dir: Path, plot_paths: dict[str, str] | None = None) -> str:
    result = analyze_data(data_dir)
    lines = [
        "# BabyJounce v2 Summary",
        "",
        "## Dataset Stats",
        "",
    ]
    for label in ("walking", "running", "driving"):
        lines.append(format_summary(label, result.summaries[label]))
        lines.append("")

    lines.extend(
        [
            "## Research-Informed Comparison (Not Clinical Limits)",
            "",
            (
                "Bands by dynamic-acceleration RMS (m/s^2): "
                "Low < 1.1, Moderate 1.1-2.7, Elevated > 2.7"
            ),
            "Dynamic acceleration is calculated as abs(||a|| - 1g) converted to m/s^2.",
            "",
        ]
    )
    for label in ("walking", "running", "driving"):
        summary = result.summaries[label]
        lines.append(
            (
                f"- {label}: band={summary.dynamic_band}, "
                f"dyn_rms={summary.dynamic_accel_mps2.rms:.3f}, "
                f"q95={summary.dynamic_accel_mps2.q95:.3f}, "
                f"q99={summary.dynamic_accel_mps2.q99:.3f}, "
                f"peak={summary.dynamic_accel_mps2.maximum:.3f}, "
                f">=2/3/4mps2={summary.dynamic_pct_ge_2_mps2:.2f}%/"
                f"{summary.dynamic_pct_ge_3_mps2:.2f}%/"
                f"{summary.dynamic_pct_ge_4_mps2:.2f}%"
            )
        )
    lines.append("")

    if plot_paths:
        plot_items = [
            ("speed_histogram", "Speed Histogram"),
            ("dynamic_accel_histogram", "Dynamic Acceleration Histogram"),
            ("route_paths", "Route Paths"),
            ("route_heatmap", "Route Density Heatmap"),
        ]
        lines.extend(["## Plots", ""])
        for key, title in plot_items:
            value = plot_paths.get(key)
            if value:
                lines.append(f"### {title}")
                lines.append("")
                lines.append(f"![{title}]({value})")
                lines.append("")

    driving = result.driving_vs_non_driving_by_speed
    running = result.running_vs_walking_by_speed
    lines.extend(
        [
            "## Simple Baselines",
            "",
            (
                "Driving vs non-driving by speed: "
                f"threshold={driving.threshold:.2f} m/s, accuracy={driving.accuracy:.4f}"
            ),
            (
                "Running vs walking by speed: "
                f"threshold={running.threshold:.2f} m/s, accuracy={running.accuracy:.4f}"
            ),
        ]
    )

    return "\n".join(lines)
