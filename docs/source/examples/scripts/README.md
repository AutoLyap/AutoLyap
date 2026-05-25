---
orphan: true
---

<!-- SPDX-FileCopyrightText: 2025-2026 AutoLyap contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Example Asset Scripts

This directory contains developer scripts that generate documentation assets
(CSV tables in `docs/source/data/` and SVG files in `docs/source/_static/`).

## Layout

```text
docs/source/examples/scripts/
  shared/
    plotting_utils.py
  gradient_method/
  proximal_gradient/
  proximal_point/
  chambolle_pock/
  heavy_ball/
  nesterov_momentum/
  optimized_gradient/
  information_theoretic_exact_method/
  accelerated_proximal_point/
  nesterov_fast_gradient/
  davis_yin/
  malitsky_tam_frb/
  douglas_rachford/
```

Each method folder contains one or more `generate_*_assets.py` entry points.
Run scripts from repository root; each script defaults to `--output-dir docs/source`.
Most sweep generators support `--reuse-data` for SVG-only refresh.
