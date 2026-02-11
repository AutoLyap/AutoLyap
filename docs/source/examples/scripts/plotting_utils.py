"""Shared helpers for docs plotting scripts.

These utilities are intentionally generic so they can be reused by future
example plot generators.

Usage:
    from pathlib import Path
    from plotting_utils import ScatterSeries, render_cartesian_svg

    points = [(0.1, -0.2), (0.3, 0.1), (0.9, 0.6)]
    render_cartesian_svg(
        path=Path("docs/source/_static/example.svg"),
        x_min=0.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        x_ticks=(0.0, 0.5, 1.0),
        y_ticks=(-1.0, 0.0, 1.0),
        scatter_series=(ScatterSeries(points=points),),
        x_label=r"$x$",
        y_label=r"$y$",
    )
"""

from __future__ import annotations

import base64
import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence
from xml.sax.saxutils import escape as xml_escape

Point2D = tuple[float, float]
DOCS_FONT_FAMILY = "'Lato', 'Avenir Next', 'Segoe UI', 'Helvetica Neue', sans-serif"
DEFAULT_SCATTER_COLOR = "#5fa8e8"

_LATEX_SYMBOLS = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\eta": "η",
    r"\kappa": "κ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\nu": "ν",
    r"\rho": "ρ",
    r"\sigma": "σ",
    r"\tau": "τ",
    r"\phi": "φ",
    r"\psi": "ψ",
    r"\omega": "ω",
    r"\Gamma": "Γ",
    r"\Delta": "Δ",
    r"\Lambda": "Λ",
    r"\Sigma": "Σ",
    r"\Phi": "Φ",
    r"\Psi": "Ψ",
    r"\Omega": "Ω",
}

_LATO_FONT_VARIANTS = (
    ("normal", 400, "lato-normal"),
    ("italic", 400, "lato-normal-italic"),
    ("normal", 700, "lato-bold"),
    ("italic", 700, "lato-bold-italic"),
)
_DOCS_FONT_DIR_CANDIDATES = ("css/fonts", "../_static/css/fonts")
_SUPERSCRIPT_LABEL_RE = re.compile(r"^\s*(.+?)\^\{(.+?)\}\s*$")
_SUBSCRIPT_LABEL_RE = re.compile(r"^\s*(.+?)_\{?(.+?)\}?\s*$")


@dataclass(frozen=True)
class CartesianStyle:
    axis_color: str = "#3f3f46"
    grid_color: str = "#e4e4e7"
    grid_width_px: float = 1.0
    label_color: str = "#111827"
    tick_color: str = "#52525b"
    plot_bg_color: str = "#f8fafc"
    font_family: str = DOCS_FONT_FAMILY


@dataclass(frozen=True)
class ScatterSeries:
    points: Sequence[Point2D]
    color: str = DEFAULT_SCATTER_COLOR
    marker_radius_px: float = 3.0
    opacity: float = 0.9


@dataclass(frozen=True)
class LineSeries:
    points: Sequence[Point2D]
    color: str = "#2563eb"
    width_px: float = 2.0
    opacity: float = 1.0
    dasharray: str | None = None


@dataclass(frozen=True)
class RegionSeries:
    points: Sequence[Point2D]
    fill_color: str = "#bfdbfe"
    fill_opacity: float = 0.35
    stroke_color: str = "#2563eb"
    stroke_width_px: float = 1.5
    stroke_opacity: float = 0.9


@dataclass(frozen=True)
class LegendItem:
    label: str
    color: str
    kind: str = "line"
    line_width_px: float = 2.0
    marker_radius_px: float = 4.0
    dasharray: str | None = None
    opacity: float = 1.0


def _latex_to_svg_label(text: str) -> tuple[str, bool]:
    """Map lightweight LaTeX-style labels to SVG-friendly text.

    Supports simple `$...$` wrappers and common Greek macros.
    """
    stripped = text.strip()
    is_math = False
    if "$" in stripped:
        is_math = True
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        stripped = stripped[1:-1].strip()
        is_math = True
    if "\\" in stripped:
        is_math = True

    converted = stripped
    for latex_macro, unicode_symbol in _LATEX_SYMBOLS.items():
        if latex_macro in converted:
            converted = converted.replace(latex_macro, unicode_symbol)

    converted = (
        converted.replace("{", "")
        .replace("}", "")
        .replace("$", "")
        .replace(r"\,", " ")
        .replace(r"\ ", " ")
        .strip()
    )
    return converted, is_math


def _svg_tick_label_markup(label: str) -> str:
    """Render lightweight ``base^{exp}`` labels using SVG superscripts."""
    label = label.strip()
    if not label:
        return ""

    match = _SUPERSCRIPT_LABEL_RE.match(label)
    if match is None:
        return xml_escape(label)

    base_text = xml_escape(match.group(1))
    exponent_text = xml_escape(match.group(2))
    return (
        f"{base_text}"
        f'<tspan baseline-shift="super" font-size="65%">{exponent_text}</tspan>'
    )


def _svg_math_label_markup(label: str, is_math: bool) -> str:
    """Render lightweight math labels, including a simple subscript form."""
    label = label.strip()
    if not label:
        return ""
    if not is_math:
        return xml_escape(label)

    sub_match = _SUBSCRIPT_LABEL_RE.match(label)
    if sub_match is None:
        return xml_escape(label)

    base_text = xml_escape(sub_match.group(1).strip())
    sub_text = xml_escape(sub_match.group(2).strip())
    return (
        f"{base_text}"
        f'<tspan baseline-shift="sub" font-size="65%">{sub_text}</tspan>'
    )


def _svg_inline_math_markup(label: str, is_math: bool) -> str:
    """Render inline lightweight math with simple sub/superscripts.

    Supports ``_x``, ``_{...}``, ``^x``, and ``^{...}`` patterns inside labels.
    """
    label = label.strip()
    if not label:
        return ""
    if not is_math:
        return xml_escape(label)

    parts: list[str] = []
    i = 0
    while i < len(label):
        ch = label[i]
        if ch in "_^" and (i + 1) < len(label):
            shift = "sub" if ch == "_" else "super"
            token: str | None = None

            if label[i + 1] == "{":
                close_brace = label.find("}", i + 2)
                if close_brace != -1:
                    token = label[i + 2 : close_brace]
                    i = close_brace + 1
                else:
                    token = label[i + 1]
                    i += 2
            else:
                token = label[i + 1]
                i += 2

            if token is not None and token != "":
                parts.append(
                    f'<tspan baseline-shift="{shift}" font-size="65%">{xml_escape(token)}</tspan>'
                )
                continue

        parts.append(xml_escape(ch))
        i += 1

    return "".join(parts)


def write_csv_rows(path: Path, header: str, rows: Iterable[str]) -> None:
    """Write a CSV-like text table with a one-line header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for row in rows:
            handle.write(row + "\n")


def read_xy_rows(path: Path) -> list[Point2D]:
    """Read x/y coordinate pairs from a CSV file with a header row."""
    points: list[Point2D] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return points

        for row in reader:
            if len(row) < 2:
                continue
            points.append((float(row[0]), float(row[1])))

    return points


def _points_to_polyline(
    points: Sequence[Point2D],
    x_to_px: Callable[[float], float],
    y_to_px: Callable[[float], float],
) -> str:
    return " ".join(f"{x_to_px(x):.3f},{y_to_px(y):.3f}" for x, y in points)


def _font_sources(font_stem: str) -> str:
    sources: list[str] = []
    data_uri = _embedded_font_data_uri(font_stem)
    if data_uri is not None:
        sources.append(f"url('{data_uri}') format('woff2')")
    for base_dir in _DOCS_FONT_DIR_CANDIDATES:
        sources.append(f"url('{base_dir}/{font_stem}.woff2') format('woff2')")
    for base_dir in _DOCS_FONT_DIR_CANDIDATES:
        sources.append(f"url('{base_dir}/{font_stem}.woff') format('woff')")
    return ",\n    ".join(sources)


def _font_face_rule(*, family: str, font_stem: str, style: str, weight: int) -> str:
    return (
        "@font-face {\n"
        f"  font-family: '{family}';\n"
        "  src:\n"
        f"    {_font_sources(font_stem)};\n"
        f"  font-style: {style};\n"
        f"  font-weight: {weight};\n"
        "}"
    )


@lru_cache(maxsize=None)
def _embedded_font_data_uri(font_stem: str) -> str | None:
    """Return an inline `data:` URI for a docs font when available."""
    for font_path in _font_binary_candidates(font_stem):
        if font_path.exists():
            encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
            return f"data:font/woff2;base64,{encoded}"
    return None


def _font_binary_candidates(font_stem: str) -> tuple[Path, ...]:
    """List likely local locations for theme font binaries."""
    here = Path(__file__).resolve()
    docs_root = here.parents[3]
    candidates: list[Path] = [
        docs_root / "build/html/_static/css/fonts" / f"{font_stem}.woff2",
    ]
    try:
        import sphinx_rtd_theme  # type: ignore

        theme_root = Path(sphinx_rtd_theme.__file__).resolve().parent
        candidates.append(theme_root / "static/css/fonts" / f"{font_stem}.woff2")
    except Exception:
        pass
    return tuple(candidates)


def _embedded_docs_font_face_css() -> str:
    """Return font-face declarations for SVGs rendered in docs HTML."""
    return "\n".join(
        _font_face_rule(
            family="Lato",
            font_stem=font_stem,
            style=font_style,
            weight=font_weight,
        )
        for font_style, font_weight, font_stem in _LATO_FONT_VARIANTS
    )


def render_cartesian_svg(
    *,
    path: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_ticks: Sequence[float],
    y_ticks: Sequence[float],
    x_tick_labels: Sequence[str] | None = None,
    y_tick_labels: Sequence[str] | None = None,
    scatter_series: Sequence[ScatterSeries] = (),
    line_series: Sequence[LineSeries] = (),
    region_series: Sequence[RegionSeries] = (),
    legend_items: Sequence[LegendItem] = (),
    legend_position: str = "top-right",
    x_label: str = "$x$",
    y_label: str = "$y$",
    title: str = "Cartesian plot",
    description: str = "Cartesian plot.",
    aria_label: str = "Cartesian plot",
    width_px: int = 960,
    height_px: int = 860,
    y_label_rotation_deg: float = -90.0,
    show_grid: bool = True,
    italic_math_labels: bool = False,
    embed_docs_font_faces: bool = True,
    style: CartesianStyle | None = None,
) -> Path:
    """Render a publication-style cartesian plot to an SVG file."""
    if x_max <= x_min:
        raise ValueError(f"Invalid x bounds: [{x_min}, {x_max}]")
    if y_max <= y_min:
        raise ValueError(f"Invalid y bounds: [{y_min}, {y_max}]")

    if style is None:
        style = CartesianStyle()

    if x_tick_labels is not None and len(x_tick_labels) != len(x_ticks):
        raise ValueError(
            "x_tick_labels must have the same length as x_ticks when provided."
        )
    if y_tick_labels is not None and len(y_tick_labels) != len(y_ticks):
        raise ValueError(
            "y_tick_labels must have the same length as y_ticks when provided."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    margin_left = 96.0
    margin_right = 32.0
    margin_top = 24.0
    margin_bottom = 88.0
    plot_width = float(width_px) - margin_left - margin_right
    plot_height = float(height_px) - margin_top - margin_bottom

    def x_to_px(x_value: float) -> float:
        return margin_left + (x_value - x_min) * plot_width / (x_max - x_min)

    def y_to_px(y_value: float) -> float:
        return margin_top + (y_max - y_value) * plot_height / (y_max - y_min)

    x_label_text, x_label_is_math = _latex_to_svg_label(x_label)
    y_label_text, y_label_is_math = _latex_to_svg_label(y_label)
    x_label_markup = _svg_math_label_markup(x_label_text, x_label_is_math)
    y_label_markup = _svg_math_label_markup(y_label_text, y_label_is_math)
    x_label_style = (
        ' font-style="italic"' if italic_math_labels and x_label_is_math else ""
    )
    y_label_style = (
        ' font-style="italic"' if italic_math_labels and y_label_is_math else ""
    )

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" '
            f'height="{height_px}" viewBox="0 0 {width_px} {height_px}" '
            f'role="img" aria-label="{xml_escape(aria_label)}">'
        ),
        f"  <title>{xml_escape(title)}</title>",
        f"  <desc>{xml_escape(description)}</desc>",
        f'  <rect x="0" y="0" width="{width_px}" height="{height_px}" fill="#ffffff"/>',
        (
            f'  <rect x="{margin_left:.3f}" y="{margin_top:.3f}" '
            f'width="{plot_width:.3f}" height="{plot_height:.3f}" fill="{style.plot_bg_color}"/>'
        ),
        "  <defs>",
        '    <clipPath id="plot-area-clip">',
        (
            f'      <rect x="{margin_left:.3f}" y="{margin_top:.3f}" '
            f'width="{plot_width:.3f}" height="{plot_height:.3f}"/>'
        ),
        "    </clipPath>",
        "  </defs>",
    ]

    if embed_docs_font_faces:
        svg_lines.extend(
            [
                '  <style type="text/css"><![CDATA[',
                _embedded_docs_font_face_css(),
                "  ]]></style>",
            ]
        )

    if show_grid:
        for tick in x_ticks:
            x_pos = x_to_px(tick)
            svg_lines.append(
                f'  <line x1="{x_pos:.3f}" y1="{margin_top:.3f}" '
                f'x2="{x_pos:.3f}" y2="{margin_top + plot_height:.3f}" '
                f'stroke="{style.grid_color}" stroke-width="{style.grid_width_px:.3f}"/>'
            )

        for tick in y_ticks:
            y_pos = y_to_px(tick)
            svg_lines.append(
                f'  <line x1="{margin_left:.3f}" y1="{y_pos:.3f}" '
                f'x2="{margin_left + plot_width:.3f}" y2="{y_pos:.3f}" '
                f'stroke="{style.grid_color}" stroke-width="{style.grid_width_px:.3f}"/>'
            )

    svg_lines.append('  <g clip-path="url(#plot-area-clip)">')

    for region in region_series:
        if len(region.points) < 3:
            continue
        points_attr = _points_to_polyline(region.points, x_to_px, y_to_px)
        svg_lines.append(
            f'    <polygon points="{points_attr}" fill="{region.fill_color}" '
            f'fill-opacity="{region.fill_opacity:.3f}" stroke="{region.stroke_color}" '
            f'stroke-width="{region.stroke_width_px:.3f}" '
            f'stroke-opacity="{region.stroke_opacity:.3f}"/>'
        )

    for scatter in scatter_series:
        for x_value, y_value in scatter.points:
            x_pos = x_to_px(x_value)
            y_pos = y_to_px(y_value)
            svg_lines.append(
                f'    <circle cx="{x_pos:.3f}" cy="{y_pos:.3f}" '
                f'r="{scatter.marker_radius_px:.3f}" fill="{scatter.color}" '
                f'fill-opacity="{scatter.opacity:.3f}"/>'
            )

    # Draw lines after points so curves stay visible above dense markers.
    for line in line_series:
        if len(line.points) < 2:
            continue
        points_attr = _points_to_polyline(line.points, x_to_px, y_to_px)
        dasharray_attr = (
            f' stroke-dasharray="{line.dasharray}"' if line.dasharray is not None else ""
        )
        svg_lines.append(
            f'    <polyline points="{points_attr}" fill="none" stroke="{line.color}" '
            f'stroke-width="{line.width_px:.3f}" stroke-opacity="{line.opacity:.3f}"'
            f"{dasharray_attr}/>"
        )

    svg_lines.append("  </g>")

    svg_lines.append(
        f'  <rect x="{margin_left:.3f}" y="{margin_top:.3f}" '
        f'width="{plot_width:.3f}" height="{plot_height:.3f}" '
        f'fill="none" stroke="{style.axis_color}" stroke-width="1.8"/>'
    )

    if legend_items:
        legend_padding_x = 14.0
        legend_padding_y = 11.0
        legend_row_height = 30.0
        legend_swatch_width = 40.0
        legend_text_gap = 10.0
        legend_font_size = 20.0
        legend_max_text_width = max(
            10.5 * len(_latex_to_svg_label(item.label)[0]) for item in legend_items
        )
        legend_width = (
            2.0 * legend_padding_x
            + legend_swatch_width
            + legend_text_gap
            + legend_max_text_width
        )
        legend_height = 2.0 * legend_padding_y + legend_row_height * len(legend_items)
        if legend_position == "top-right":
            legend_x = margin_left + plot_width - legend_width - 16.0
            legend_y = margin_top + 16.0
        elif legend_position == "top-left":
            legend_x = margin_left + 16.0
            legend_y = margin_top + 16.0
        elif legend_position == "bottom-left":
            legend_x = margin_left + 16.0
            legend_y = margin_top + plot_height - legend_height - 16.0
        elif legend_position == "bottom-right":
            legend_x = margin_left + plot_width - legend_width - 16.0
            legend_y = margin_top + plot_height - legend_height - 16.0
        else:
            raise ValueError(
                "Unsupported legend position. Use 'top-right', 'top-left', "
                f"'bottom-left', or 'bottom-right'; got {legend_position!r}."
            )

        svg_lines.append(
            f'  <rect x="{legend_x:.3f}" y="{legend_y:.3f}" '
            f'width="{legend_width:.3f}" height="{legend_height:.3f}" '
            'rx="6" ry="6" fill="#ffffff" fill-opacity="0.94" '
            'stroke="#000000" stroke-width="1.2"/>'
        )

        for row_id, item in enumerate(legend_items):
            swatch_left = legend_x + legend_padding_x
            swatch_right = swatch_left + legend_swatch_width
            y_center = (
                legend_y
                + legend_padding_y
                + row_id * legend_row_height
                + 0.5 * legend_row_height
            )

            if item.kind in ("line", "line+marker"):
                dasharray_attr = (
                    f' stroke-dasharray="{item.dasharray}"'
                    if item.dasharray is not None
                    else ""
                )
                svg_lines.append(
                    f'  <line x1="{swatch_left:.3f}" y1="{y_center:.3f}" '
                    f'x2="{swatch_right:.3f}" y2="{y_center:.3f}" stroke="{item.color}" '
                    f'stroke-width="{item.line_width_px:.3f}" '
                    f'stroke-opacity="{item.opacity:.3f}"{dasharray_attr}/>'
                )

            if item.kind in ("marker", "line+marker"):
                svg_lines.append(
                    f'  <circle cx="{0.5 * (swatch_left + swatch_right):.3f}" '
                    f'cy="{y_center:.3f}" r="{item.marker_radius_px:.3f}" '
                    f'fill="{item.color}" fill-opacity="{item.opacity:.3f}"/>'
                )

            if item.kind not in ("line", "marker", "line+marker"):
                raise ValueError(
                    "Unsupported legend item kind. Use 'line', 'marker', or "
                    f"'line+marker'; got {item.kind!r}."
                )

            legend_label_text, legend_label_is_math = _latex_to_svg_label(item.label)
            legend_label_markup = _svg_inline_math_markup(
                legend_label_text,
                legend_label_is_math,
            )
            svg_lines.append(
                f'  <text x="{swatch_right + legend_text_gap:.3f}" '
                f'y="{y_center + 7.0:.3f}" text-anchor="start" '
                f'fill="{style.label_color}" font-family="{style.font_family}" '
                f'font-size="{legend_font_size:.1f}">{legend_label_markup}</text>'
            )

    tick_length_px = 8.0

    for tick_index, tick in enumerate(x_ticks):
        x_pos = x_to_px(tick)
        tick_label = (
            x_tick_labels[tick_index]
            if x_tick_labels is not None
            else f"{tick:g}"
        )
        svg_lines.append(
            f'  <line x1="{x_pos:.3f}" y1="{margin_top + plot_height:.3f}" '
            f'x2="{x_pos:.3f}" y2="{margin_top + plot_height - tick_length_px:.3f}" '
            f'stroke="{style.axis_color}" stroke-width="1.4"/>'
        )
        svg_lines.append(
            f'  <text x="{x_pos:.3f}" y="{height_px - 42.0:.3f}" '
            f'text-anchor="middle" fill="{style.tick_color}" '
            f'font-family="{style.font_family}" font-size="21">{_svg_tick_label_markup(tick_label)}</text>'
        )

    for tick_index, tick in enumerate(y_ticks):
        y_pos = y_to_px(tick)
        tick_label = (
            y_tick_labels[tick_index]
            if y_tick_labels is not None
            else f"{tick:g}"
        )
        svg_lines.append(
            f'  <line x1="{margin_left + tick_length_px:.3f}" y1="{y_pos:.3f}" '
            f'x2="{margin_left:.3f}" y2="{y_pos:.3f}" '
            f'stroke="{style.axis_color}" stroke-width="1.4"/>'
        )
        svg_lines.append(
            f'  <text x="{margin_left - 14.0:.3f}" y="{y_pos + 7.0:.3f}" '
            f'text-anchor="end" fill="{style.tick_color}" '
            f'font-family="{style.font_family}" font-size="21">{_svg_tick_label_markup(tick_label)}</text>'
        )

    svg_lines.extend(
        [
            (
                f'  <text x="{margin_left + plot_width / 2.0:.3f}" '
                f'y="{height_px - 10.0:.3f}" text-anchor="middle" fill="{style.label_color}" '
                f'font-family="{style.font_family}" font-size="28"{x_label_style}>'
                f"{x_label_markup}</text>"
            ),
            (
                f'  <text x="30.0" y="{margin_top + plot_height / 2.0:.3f}" '
                f'text-anchor="middle" transform="rotate({y_label_rotation_deg:g} 30.0 '
                f'{margin_top + plot_height / 2.0:.3f})" fill="{style.label_color}" '
                f'font-family="{style.font_family}" font-size="28"{y_label_style}>'
                f"{y_label_markup}</text>"
            ),
            "</svg>",
        ]
    )

    path.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")
    return path


def render_scatter_svg(
    *,
    path: Path,
    points: Sequence[Point2D],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_ticks: Sequence[float],
    y_ticks: Sequence[float],
    x_tick_labels: Sequence[str] | None = None,
    y_tick_labels: Sequence[str] | None = None,
    x_label: str = "$x$",
    y_label: str = "$y$",
    title: str = "Scatter plot",
    description: str = "Scatter plot.",
    aria_label: str = "Scatter plot",
    width_px: int = 960,
    height_px: int = 860,
    y_label_rotation_deg: float = -90.0,
    marker_radius_px: float = 3.0,
    show_grid: bool = True,
    legend_items: Sequence[LegendItem] = (),
    legend_position: str = "top-right",
    italic_math_labels: bool = False,
    embed_docs_font_faces: bool = True,
    style: CartesianStyle | None = None,
) -> Path:
    """Render a scatter plot using the shared cartesian renderer."""
    return render_cartesian_svg(
        path=path,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_tick_labels=x_tick_labels,
        y_tick_labels=y_tick_labels,
        scatter_series=(ScatterSeries(points=points, marker_radius_px=marker_radius_px),),
        legend_items=legend_items,
        legend_position=legend_position,
        x_label=x_label,
        y_label=y_label,
        title=title,
        description=description,
        aria_label=aria_label,
        width_px=width_px,
        height_px=height_px,
        y_label_rotation_deg=y_label_rotation_deg,
        show_grid=show_grid,
        italic_math_labels=italic_math_labels,
        embed_docs_font_faces=embed_docs_font_faces,
        style=style,
    )
