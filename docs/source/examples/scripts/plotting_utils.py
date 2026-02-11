"""Shared helpers for docs plotting scripts.

These utilities are intentionally generic so they can be reused by
future example plot generators.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional


def write_csv_rows(path: Path, header: str, rows: Iterable[str]) -> None:
    """Write a CSV-like text table with a one-line header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for row in rows:
            handle.write(row + "\n")


def render_tex_input_to_png(
    *,
    output_dir: Path,
    tex_input_relpath: Path,
    png_output_relpath: Path,
    wrapper_stem: str = "_plot_preview_wrapper",
    font_size_pt: int = 12,
    dpi: int = 220,
    preview_border_pt: int = 2,
) -> Optional[Path]:
    """Render a TeX input file to PNG via pdflatex + Ghostscript.

    Parameters are relative to `output_dir` for portability across examples.
    """
    pdflatex = shutil.which("pdflatex")
    gs = shutil.which("gs")

    if pdflatex is None:
        print("Preview skipped: 'pdflatex' not found in PATH.")
        return None
    if gs is None:
        print("Preview skipped: 'gs' (Ghostscript) not found in PATH.")
        return None

    png_path = output_dir / png_output_relpath
    png_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper_tex = output_dir / f"{wrapper_stem}.tex"
    wrapper_aux = output_dir / f"{wrapper_stem}.aux"
    wrapper_log = output_dir / f"{wrapper_stem}.log"
    wrapper_pdf = output_dir / f"{wrapper_stem}.pdf"
    pdflatex_log = output_dir / f"{wrapper_stem}.pdflatex.log"
    gs_log = output_dir / f"{wrapper_stem}.gs.log"

    wrapper_text = (
        f"\\documentclass[{font_size_pt}pt]{{article}}\n"
        "\\usepackage{pgfplots}\n"
        "\\pgfplotsset{compat=1.18}\n"
        "\\usepackage[active,tightpage]{preview}\n"
        "\\PreviewEnvironment{tikzpicture}\n"
        f"\\setlength\\PreviewBorder{{{preview_border_pt}pt}}\n"
        "\\pagestyle{empty}\n"
        "\\begin{document}\n"
        f"\\input{{{tex_input_relpath.as_posix()}}}\n"
        "\\end{document}\n"
    )
    wrapper_tex.write_text(wrapper_text, encoding="utf-8")

    try:
        with pdflatex_log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", wrapper_tex.name],
                cwd=output_dir,
                check=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

        with gs_log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                [
                    gs,
                    "-dSAFER",
                    "-dBATCH",
                    "-dNOPAUSE",
                    "-sDEVICE=pngalpha",
                    f"-r{dpi}",
                    f"-sOutputFile={png_path}",
                    wrapper_pdf.name,
                ],
                cwd=output_dir,
                check=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
    except subprocess.CalledProcessError as exc:
        print(
            "Preview rendering failed. "
            f"Inspect logs: {pdflatex_log} and {gs_log}. Exit code={exc.returncode}"
        )
        return None
    finally:
        for path in (wrapper_tex, wrapper_aux, wrapper_log, wrapper_pdf, pdflatex_log, gs_log):
            if path.exists():
                path.unlink()

    return png_path
