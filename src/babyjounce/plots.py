from __future__ import annotations

import math
from pathlib import Path

from .analysis import DATASET_FILES, GRAVITY_MPS2, Record, load_records

MODE_COLORS = {
    "walking": "#2e7d32",
    "running": "#ef6c00",
    "driving": "#1565c0",
}


def _scale(value: float, source_min: float, source_max: float, dest_min: float, dest_max: float) -> float:
    if source_max <= source_min:
        return (dest_min + dest_max) / 2.0
    return dest_min + (value - source_min) * (dest_max - dest_min) / (source_max - source_min)


def _histogram(values: list[float], bins: int, value_min: float, value_max: float) -> list[int]:
    counts = [0 for _ in range(bins)]
    if not values:
        return counts
    span = max(value_max - value_min, 1e-9)
    for value in values:
        index = int((value - value_min) / span * bins)
        if index < 0:
            index = 0
        elif index >= bins:
            index = bins - 1
        counts[index] += 1
    return counts


def _hex_to_rgb(color_hex: str) -> tuple[int, int, int]:
    raw = color_hex.lstrip("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _write_svg(path: Path, width: int, height: int, body_lines: list[str]) -> None:
    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect x="0" y="0" width="100%" height="100%" fill="#fafafa"/>',
    ]
    svg_lines.extend(body_lines)
    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines), encoding="utf-8")


def _render_histogram(
    values_by_mode: dict[str, list[float]],
    output_path: Path,
    title: str,
    x_axis_label: str,
    bins: int = 42,
) -> None:
    width = 1100
    height = 620
    margin_left = 76
    margin_right = 24
    margin_top = 60
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_values = [value for values in values_by_mode.values() for value in values]
    if not all_values:
        _write_svg(output_path, width, height, ['<text x="20" y="30">No data</text>'])
        return

    value_min = min(all_values)
    value_max = max(all_values)
    hist_by_mode: dict[str, list[float]] = {}
    max_fraction = 0.0
    for mode, values in values_by_mode.items():
        counts = _histogram(values, bins=bins, value_min=value_min, value_max=value_max)
        denom = max(len(values), 1)
        fractions = [count / denom for count in counts]
        hist_by_mode[mode] = fractions
        max_fraction = max(max_fraction, max(fractions))

    max_fraction = max(max_fraction, 1e-6)
    bin_width_px = plot_width / bins
    body = [
        f'<text x="{margin_left}" y="32" font-size="22" fill="#222">{title}</text>',
        (
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
            f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#444" stroke-width="1"/>'
        ),
        (
            f'<line x1="{margin_left}" y1="{margin_top}" '
            f'x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#444" stroke-width="1"/>'
        ),
        (
            f'<text x="{margin_left + (plot_width / 2):.1f}" y="{height - 26}" '
            f'font-size="14" fill="#333" text-anchor="middle">{x_axis_label}</text>'
        ),
        (
            f'<text x="22" y="{margin_top + (plot_height / 2):.1f}" font-size="14" '
            f'fill="#333" transform="rotate(-90 22 {margin_top + (plot_height / 2):.1f})" '
            'text-anchor="middle">Sample Fraction</text>'
        ),
    ]

    for tick in range(6):
        fraction = tick / 5
        y = margin_top + plot_height - (plot_height * fraction)
        value = max_fraction * fraction
        body.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" '
            'stroke="#ddd" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" font-size="11" fill="#555" text-anchor="end">'
            f"{value:.3f}</text>"
        )

    body.append(
        f'<text x="{margin_left}" y="{margin_top + plot_height + 20}" font-size="11" fill="#555">{value_min:.2f}</text>'
    )
    body.append(
        (
            f'<text x="{margin_left + plot_width}" y="{margin_top + plot_height + 20}" font-size="11" fill="#555" '
            f'text-anchor="end">{value_max:.2f}</text>'
        )
    )

    legend_x = margin_left + 8
    legend_y = margin_top - 14
    for index, mode in enumerate(("walking", "running", "driving")):
        color = MODE_COLORS[mode]
        rgb = _hex_to_rgb(color)
        y = legend_y + (index * 18)
        body.append(
            f'<rect x="{legend_x}" y="{y - 9}" width="14" height="10" fill="rgba({rgb[0]},{rgb[1]},{rgb[2]},0.55)"/>'
        )
        body.append(
            f'<text x="{legend_x + 20}" y="{y}" font-size="12" fill="#333">{mode}</text>'
        )

    for mode in ("walking", "running", "driving"):
        fractions = hist_by_mode[mode]
        rgb = _hex_to_rgb(MODE_COLORS[mode])
        for bin_index, fraction in enumerate(fractions):
            bar_height = _scale(fraction, 0.0, max_fraction, 0.0, plot_height)
            if bar_height <= 0:
                continue
            x = margin_left + (bin_index * bin_width_px)
            y = margin_top + plot_height - bar_height
            body.append(
                (
                    f'<rect x="{x:.3f}" y="{y:.3f}" width="{max(bin_width_px - 0.6, 0.2):.3f}" '
                    f'height="{bar_height:.3f}" fill="rgba({rgb[0]},{rgb[1]},{rgb[2]},0.35)"/>'
                )
            )

    _write_svg(output_path, width, height, body)


def _render_route_paths(records_by_mode: dict[str, list[Record]], output_path: Path) -> None:
    width = 1100
    height = 640
    margin = 46
    plot_width = width - (margin * 2)
    plot_height = height - (margin * 2)

    all_lats = [record.latitude for records in records_by_mode.values() for record in records]
    all_lons = [record.longitude for records in records_by_mode.values() for record in records]
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)

    body = [
        '<text x="46" y="28" font-size="22" fill="#222">Route Paths by Mode</text>',
        f'<rect x="{margin}" y="{margin}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#ddd"/>',
    ]

    for mode in ("walking", "running", "driving"):
        points = [(record.longitude, record.latitude) for record in records_by_mode[mode]]
        step = max(1, len(points) // 2400)
        sampled = points[::step]
        polyline_points = []
        for lon, lat in sampled:
            x = _scale(lon, lon_min, lon_max, margin + 6, margin + plot_width - 6)
            y = _scale(lat, lat_min, lat_max, margin + plot_height - 6, margin + 6)
            polyline_points.append(f"{x:.2f},{y:.2f}")
        if polyline_points:
            color = MODE_COLORS[mode]
            body.append(
                f'<polyline points="{" ".join(polyline_points)}" fill="none" stroke="{color}" stroke-width="1.4" '
                'stroke-linecap="round" stroke-linejoin="round" opacity="0.78"/>'
            )

    legend_x = margin + 10
    legend_y = margin + 16
    for index, mode in enumerate(("walking", "running", "driving")):
        y = legend_y + (index * 18)
        body.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 20}" y2="{y}" '
            f'stroke="{MODE_COLORS[mode]}" stroke-width="2"/>'
        )
        body.append(
            f'<text x="{legend_x + 26}" y="{y + 4}" font-size="12" fill="#333">{mode}</text>'
        )

    _write_svg(output_path, width, height, body)


def _render_route_heatmap(records_by_mode: dict[str, list[Record]], output_path: Path) -> None:
    width = 1200
    height = 450
    margin_x = 32
    margin_y = 42
    panel_gap = 24
    panel_count = 3
    panel_width = (width - (margin_x * 2) - (panel_gap * (panel_count - 1))) / panel_count
    panel_height = height - (margin_y * 2)
    bins = 32

    all_lats = [record.latitude for records in records_by_mode.values() for record in records]
    all_lons = [record.longitude for records in records_by_mode.values() for record in records]
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)

    body = ['<text x="32" y="28" font-size="22" fill="#222">Route Density Heatmaps</text>']

    for panel_index, mode in enumerate(("walking", "running", "driving")):
        panel_x = margin_x + panel_index * (panel_width + panel_gap)
        panel_y = margin_y
        body.append(
            f'<rect x="{panel_x:.2f}" y="{panel_y:.2f}" width="{panel_width:.2f}" '
            f'height="{panel_height:.2f}" fill="#ffffff" stroke="#ddd"/>'
        )
        body.append(
            f'<text x="{panel_x + 8:.2f}" y="{panel_y + 18:.2f}" font-size="14" fill="#333">{mode}</text>'
        )

        grid = [[0 for _ in range(bins)] for _ in range(bins)]
        for record in records_by_mode[mode]:
            x_value = _scale(record.longitude, lon_min, lon_max, 0, bins - 1e-9)
            y_value = _scale(record.latitude, lat_min, lat_max, 0, bins - 1e-9)
            x_idx = max(0, min(bins - 1, int(x_value)))
            y_idx = max(0, min(bins - 1, int(y_value)))
            grid[y_idx][x_idx] += 1

        max_count = max(max(row) for row in grid) if grid else 1
        cell_width = panel_width / bins
        cell_height = panel_height / bins
        rgb = _hex_to_rgb(MODE_COLORS[mode])
        for y_idx in range(bins):
            for x_idx in range(bins):
                count = grid[y_idx][x_idx]
                if count <= 0:
                    continue
                alpha = _scale(count, 1, max_count, 0.08, 0.92)
                draw_x = panel_x + (x_idx * cell_width)
                # invert y so north is up
                draw_y = panel_y + ((bins - y_idx - 1) * cell_height)
                body.append(
                    (
                        f'<rect x="{draw_x:.3f}" y="{draw_y:.3f}" width="{cell_width:.3f}" '
                        f'height="{cell_height:.3f}" fill="rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.3f})"/>'
                    )
                )

    _write_svg(output_path, width, height, body)


def generate_plots(data_dir: Path, plots_dir: Path) -> dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    records_by_mode = {
        mode: load_records(data_dir / filename, label=mode) for mode, filename in DATASET_FILES.items()
    }

    speeds_by_mode = {
        mode: [record.speed_mps for record in records] for mode, records in records_by_mode.items()
    }
    dynamic_accel_by_mode = {
        mode: [
            abs(
                math.sqrt(record.accel_x_g**2 + record.accel_y_g**2 + record.accel_z_g**2) - 1.0
            )
            * GRAVITY_MPS2
            for record in records
        ]
        for mode, records in records_by_mode.items()
    }

    output_paths = {
        "speed_histogram": plots_dir / "speed_histogram.svg",
        "dynamic_accel_histogram": plots_dir / "dynamic_accel_histogram.svg",
        "route_paths": plots_dir / "route_paths.svg",
        "route_heatmap": plots_dir / "route_heatmap.svg",
    }

    _render_histogram(
        values_by_mode=speeds_by_mode,
        output_path=output_paths["speed_histogram"],
        title="Speed Distribution by Mode",
        x_axis_label="Speed (m/s)",
    )
    _render_histogram(
        values_by_mode=dynamic_accel_by_mode,
        output_path=output_paths["dynamic_accel_histogram"],
        title="Dynamic Acceleration Distribution by Mode",
        x_axis_label="Dynamic Acceleration abs(||a|| - 1g) (m/s^2)",
    )
    _render_route_paths(records_by_mode=records_by_mode, output_path=output_paths["route_paths"])
    _render_route_heatmap(records_by_mode=records_by_mode, output_path=output_paths["route_heatmap"])

    return output_paths
