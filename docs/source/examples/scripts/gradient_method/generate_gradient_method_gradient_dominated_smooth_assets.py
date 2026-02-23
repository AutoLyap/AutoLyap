#!/usr/bin/env python3
"""Render gradient-method (gradient-dominated smooth) SVG/data assets.

Usage:
    python docs/source/examples/scripts/gradient_method/generate_gradient_method_gradient_dominated_smooth_assets.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence


# Output locations (relative to --output-dir).
AUTO_DATA_REL = (
    Path("data") / "gradient_method_gradient_dominated_smooth" / "auto_lyapunov.csv"
)
MERGED_DATA_REL = (
    Path("data") / "gradient_method_gradient_dominated_smooth" / "gamma_rho.csv"
)
PLOT_IMAGE_REL = (
    Path("_static") / "gradient_method_gradient_dominated_smooth_rho_vs_gamma.svg"
)

# Parameter defaults for the docs example.
DEFAULT_MU_GD = 0.5
DEFAULT_L = 1.0

# Plot configuration (matched to the other rho-vs-gamma assets).
WIDTH_PX = 960
HEIGHT_PX = WIDTH_PX // 2
PLOT_X0 = 96.0
PLOT_Y0 = 24.0
PLOT_X1 = 928.0
PLOT_Y1 = 392.0
Y_RANGE = (0.39, 1.0)
Y_TICKS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

FONT_FAMILY = "'Lato', 'Avenir Next', 'Segoe UI', 'Helvetica Neue', sans-serif"
AXIS_COLOR = "#3f3f46"
GRID_COLOR = "#9ca3af"
TICK_COLOR = "#52525b"
LABEL_COLOR = "#111827"
PLOT_BG_COLOR = "#f8fafc"
THEORY_COLOR = "#000000"
AUTOLYAP_COLOR = "#5fa8e8"

GRID_WIDTH = 1.35
AXIS_BORDER_WIDTH = 1.8
AXIS_TICK_WIDTH = 1.4
THEORY_LINE_WIDTH = 2.8
MARKER_RADIUS = 4.5
TICK_FONT_SIZE = 21
LABEL_FONT_SIZE = 28
LEGEND_FONT_SIZE = 20.0
LEGEND_W = 308.0
LEGEND_H = 82.0
LEGEND_MARGIN_TOP = 16.0


def _build_parser() -> argparse.ArgumentParser:
    default_output = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Generate the gradient-dominated smooth gradient-method data table and "
            "SVG plot asset."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where data files and plot assets will be written.",
    )
    parser.add_argument(
        "--mu-gd",
        type=float,
        default=DEFAULT_MU_GD,
        help="Gradient-dominance parameter mu_gd.",
    )
    parser.add_argument(
        "--L",
        type=float,
        default=DEFAULT_L,
        help="Smoothness parameter L.",
    )
    return parser


def _read_xy_rows(path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        _ = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            points.append((float(row[0]), float(row[1])))
    if not points:
        raise RuntimeError(f"No data rows found in {path}.")
    return points


def _write_csv_rows(path: Path, header: str, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for row in rows:
            handle.write(row + "\n")


def _rho_theorem3_specialized(mu_gd: float, L: float, gamma: float) -> float:
    """Theorem 3 in Abbaszadehpeivasti et al. (2023), specialized to mu = -L."""
    if not (mu_gd > 0.0 and L > 0.0):
        raise ValueError(f"Require mu_gd > 0 and L > 0. Got mu_gd={mu_gd}, L={L}.")
    if not (0.0 < gamma < (2.0 / L)):
        raise ValueError(f"Require 0 < gamma < 2/L. Got gamma={gamma}, 2/L={2.0 / L}.")

    threshold_1 = 1.0 / L
    threshold_2 = math.sqrt(3.0) / L

    if gamma < threshold_1:
        discriminant = 4.0 * L * L - (
            2.0 * L * (L + mu_gd) * mu_gd * gamma * (2.0 - L * gamma)
        )
        discriminant = max(discriminant, 0.0)
        numerator = mu_gd * (1.0 - L * gamma) + math.sqrt(discriminant)
        denominator = 2.0 * L + mu_gd
        return (numerator / denominator) ** 2

    if gamma <= threshold_2:
        return 1.0 - (mu_gd * gamma * (4.0 - (L * gamma) ** 2)) / (2.0 + mu_gd * gamma)

    numerator = (L * gamma - 1.0) ** 2
    denominator = numerator + mu_gd * gamma * (2.0 - L * gamma)
    return numerator / denominator


def _build_theory_points(
    autolyap_points: Sequence[tuple[float, float]],
    mu_gd: float,
    L: float,
) -> list[tuple[float, float]]:
    sorted_points = sorted(autolyap_points, key=lambda point: point[0])
    return [
        (gamma, _rho_theorem3_specialized(mu_gd=mu_gd, L=L, gamma=gamma))
        for gamma, _ in sorted_points
    ]


def _write_merged_rows(
    path: Path,
    autolyap_points: Sequence[tuple[float, float]],
    theory_points: Sequence[tuple[float, float]],
) -> None:
    sorted_auto = sorted(autolyap_points, key=lambda point: point[0])
    if len(sorted_auto) != len(theory_points):
        raise ValueError(
            "Expected AutoLyap and theory datasets to have identical row counts. "
            f"Got {len(sorted_auto)} and {len(theory_points)}."
        )

    rows: list[str] = []
    for (gamma_auto, rho_auto), (gamma_theory, rho_theory) in zip(
        sorted_auto,
        theory_points,
    ):
        if gamma_auto != gamma_theory:
            raise ValueError(
                "AutoLyap and theory gamma grids are not aligned after sorting."
            )
        rows.append(f"{gamma_auto:.12f},{rho_auto:.12f},{rho_theory:.12f}")
    _write_csv_rows(path, "gamma,rho_autolyap,rho_theory", rows)


def _render_svg(
    path: Path,
    autolyap_points: Sequence[tuple[float, float]],
    theory_points: Sequence[tuple[float, float]],
    L: float,
) -> None:
    x_min = 0.0
    x_max = 2.0 / L
    x_ticks = tuple(x_min + i * (x_max - x_min) / 5.0 for i in range(6))
    y_min, y_max = Y_RANGE

    plot_w = PLOT_X1 - PLOT_X0
    plot_h = PLOT_Y1 - PLOT_Y0
    sorted_auto = sorted(autolyap_points, key=lambda point: point[0])

    def x_px(value: float) -> float:
        ratio = (value - x_min) / (x_max - x_min)
        return PLOT_X0 + ratio * plot_w

    def y_px(value: float) -> float:
        ratio = (value - y_min) / (y_max - y_min)
        return PLOT_Y1 - ratio * plot_h

    theory_path_data = " ".join(
        (
            ("M" if i == 0 else "L") + f" {x_px(x):.3f} {y_px(y):.3f}"
            for i, (x, y) in enumerate(theory_points)
        )
    )

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_PX}" height="{HEIGHT_PX}" '
        f'viewBox="0 0 {WIDTH_PX} {HEIGHT_PX}" role="img" aria-label="Gradient-method rho versus gamma">'
    )
    lines.append("  <title>Gradient-method rho versus gamma</title>")
    lines.append(
        "  <desc>Rho vs gamma for gradient method: theoretical curve and AutoLyap points.</desc>"
    )
    lines.append(
        f'  <rect x="0" y="0" width="{WIDTH_PX}" height="{HEIGHT_PX}" fill="#ffffff"/>'
    )
    lines.append(
        f'  <rect x="{PLOT_X0:.3f}" y="{PLOT_Y0:.3f}" width="{plot_w:.3f}" height="{plot_h:.3f}" fill="{PLOT_BG_COLOR}"/>'
    )
    lines.append("  <defs>")
    lines.append('    <clipPath id="plot-area-clip-gradient-nonconvex">')
    lines.append(
        f'      <rect x="{PLOT_X0:.3f}" y="{PLOT_Y0:.3f}" width="{plot_w:.3f}" height="{plot_h:.3f}"/>'
    )
    lines.append("    </clipPath>")
    lines.append("  </defs>")

    for tick in x_ticks:
        xp = x_px(tick)
        lines.append(
            f'  <line x1="{xp:.3f}" y1="{PLOT_Y0:.3f}" x2="{xp:.3f}" y2="{PLOT_Y1:.3f}" stroke="{GRID_COLOR}" stroke-width="{GRID_WIDTH:.3f}"/>'
        )
    for tick in Y_TICKS:
        yp = y_px(tick)
        lines.append(
            f'  <line x1="{PLOT_X0:.3f}" y1="{yp:.3f}" x2="{PLOT_X1:.3f}" y2="{yp:.3f}" stroke="{GRID_COLOR}" stroke-width="{GRID_WIDTH:.3f}"/>'
        )

    lines.append('  <g clip-path="url(#plot-area-clip-gradient-nonconvex)">')
    lines.append(f'    <g fill="{AUTOLYAP_COLOR}" fill-opacity="0.900">')
    for x, y in sorted_auto:
        lines.append(f'    <circle cx="{x_px(x):.3f}" cy="{y_px(y):.3f}" r="{MARKER_RADIUS:.3f}"/>')
    lines.append("    </g>")
    lines.append(
        f'    <path d="{theory_path_data}" fill="none" stroke="{THEORY_COLOR}" stroke-width="{THEORY_LINE_WIDTH:.3f}" stroke-opacity="1.000"/>'
    )
    lines.append("  </g>")

    lines.append(
        f'  <rect x="{PLOT_X0:.3f}" y="{PLOT_Y0:.3f}" width="{plot_w:.3f}" height="{plot_h:.3f}" fill="none" stroke="{AXIS_COLOR}" stroke-width="{AXIS_BORDER_WIDTH:.1f}"/>'
    )

    legend_x = (PLOT_X0 + PLOT_X1 - LEGEND_W) / 2.0
    legend_y = PLOT_Y0 + LEGEND_MARGIN_TOP
    lines.append(
        f'  <rect x="{legend_x:.3f}" y="{legend_y:.3f}" width="{LEGEND_W:.3f}" height="{LEGEND_H:.3f}" rx="6" ry="6" fill="#ffffff" fill-opacity="0.94" stroke="#000000" stroke-width="1.2"/>'
    )
    lines.append(
        f'  <line x1="{legend_x + 14.0:.3f}" y1="{legend_y + 26.0:.3f}" x2="{legend_x + 54.0:.3f}" y2="{legend_y + 26.0:.3f}" stroke="{THEORY_COLOR}" stroke-width="{THEORY_LINE_WIDTH:.3f}" stroke-opacity="1.000"/>'
    )
    lines.append(
        f'  <text x="{legend_x + 64.0:.3f}" y="{legend_y + 33.0:.3f}" text-anchor="start" fill="{LABEL_COLOR}" font-family="{FONT_FAMILY}" font-size="{LEGEND_FONT_SIZE:.1f}">[AdKZ23, Theorem 3]</text>'
    )
    lines.append(
        f'  <circle cx="{legend_x + 34.0:.3f}" cy="{legend_y + 56.0:.3f}" r="{MARKER_RADIUS:.3f}" fill="{AUTOLYAP_COLOR}" fill-opacity="1.000"/>'
    )
    lines.append(
        f'  <text x="{legend_x + 64.0:.3f}" y="{legend_y + 63.0:.3f}" text-anchor="start" fill="{LABEL_COLOR}" font-family="{FONT_FAMILY}" font-size="{LEGEND_FONT_SIZE:.1f}">AutoLyap</text>'
    )

    for tick in x_ticks:
        xp = x_px(tick)
        lines.append(
            f'  <line x1="{xp:.3f}" y1="{PLOT_Y1:.3f}" x2="{xp:.3f}" y2="{PLOT_Y1 - 8.0:.3f}" stroke="{AXIS_COLOR}" stroke-width="{AXIS_TICK_WIDTH:.1f}"/>'
        )
        lines.append(
            f'  <text x="{xp:.3f}" y="{PLOT_Y1 + 46.0:.3f}" text-anchor="middle" fill="{TICK_COLOR}" font-family="{FONT_FAMILY}" font-size="{TICK_FONT_SIZE}">{tick:g}</text>'
        )
    for tick in Y_TICKS:
        yp = y_px(tick)
        lines.append(
            f'  <line x1="{PLOT_X0 + 8.0:.3f}" y1="{yp:.3f}" x2="{PLOT_X0:.3f}" y2="{yp:.3f}" stroke="{AXIS_COLOR}" stroke-width="{AXIS_TICK_WIDTH:.1f}"/>'
        )
        lines.append(
            f'  <text x="{PLOT_X0 - 14.0:.3f}" y="{yp + 7.0:.3f}" text-anchor="end" fill="{TICK_COLOR}" font-family="{FONT_FAMILY}" font-size="{TICK_FONT_SIZE}">{tick:.1f}</text>'
        )

    lines.append(
        f'  <text x="{WIDTH_PX / 2:.3f}" y="{HEIGHT_PX - 10.0:.3f}" text-anchor="middle" fill="{LABEL_COLOR}" font-family="{FONT_FAMILY}" font-size="{LABEL_FONT_SIZE}">γ</text>'
    )
    lines.append(
        f'  <text x="30.0" y="{(PLOT_Y0 + PLOT_Y1) / 2.0:.3f}" text-anchor="middle" transform="rotate(0 30.0 {(PLOT_Y0 + PLOT_Y1) / 2.0:.3f})" fill="{LABEL_COLOR}" font-family="{FONT_FAMILY}" font-size="{LABEL_FONT_SIZE}">ρ</text>'
    )
    lines.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    auto_data_path = args.output_dir / AUTO_DATA_REL
    autolyap_points = _read_xy_rows(auto_data_path)
    theory_points = _build_theory_points(
        autolyap_points=autolyap_points,
        mu_gd=args.mu_gd,
        L=args.L,
    )

    merged_path = args.output_dir / MERGED_DATA_REL
    _write_merged_rows(
        path=merged_path,
        autolyap_points=autolyap_points,
        theory_points=theory_points,
    )

    plot_svg_path = args.output_dir / PLOT_IMAGE_REL
    _render_svg(
        path=plot_svg_path,
        autolyap_points=autolyap_points,
        theory_points=theory_points,
        L=args.L,
    )

    print("Finished.")
    print(f"  AutoLyap data: {auto_data_path}")
    print(f"  Merged table:  {merged_path}")
    print(f"  Plot image:    {plot_svg_path}")
    print(f"  mu_gd={args.mu_gd}")
    print(f"  L={args.L}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
