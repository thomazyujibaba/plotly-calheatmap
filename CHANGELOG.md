# Changelog

## 0.2 — (unreleased)

### Breaking Changes

- **Renamed public API** — `calplot` → `calheatmap`, `month_calplot` → `month_calheatmap` to align with the package name.

### New Features

- **Vertical orientation** — New `vertical` parameter renders months as rows instead of columns.
- **Month gap spacing** — New `month_gap` parameter adds visual separation between months.
- **Built-in aggregation** — New `agg` parameter (`"sum"`, `"mean"`, `"count"`, `"max"`) to aggregate duplicate dates automatically.
- **Hourly heatmap** — New `hourly_calheatmap()` function for hour × day-of-month heatmaps per month.
- New examples: `aggregation.py`, `hourly_heatmap.py`, `vertical_orientation.py`.

### Infrastructure

- Updated dev dependencies for Python 3.12+ compatibility (`black >=23`, `flake8 >=6`).
- Removed `streamlit` from dev dependencies.

---

## 0.1 — Initial Release (plotly-calheatmap)

Forked from [plotly-calplot](https://github.com/brunorosilva/plotly-calplot) and renamed to **plotly-calheatmap**.

### Features

- **Customizable hover templates** — Full control over hovertext via `hovertemplate` parameter, replacing the fixed format.
- **Year navigation buttons** — Optional prev/next year buttons for interactive navigation across multi-year plots.
- **Localization (i18n)** — Month and day-of-week names can be displayed in any locale supported by Babel (e.g. `locale="pt_BR"`).
- **Independent tick configuration per year** — Each year subplot maintains its own tick labels, so custom labels (like month names with totals) render correctly across multi-year plots.
- **GitHub-style contributions example** — New example showing how to recreate a GitHub contributions heatmap.
- **Improved layout options** — Additional styling parameters for fine-grained control over plot appearance.

### Fixes

- Fixed hovertext formatting for `month_calplot`.
- Fixed type hints for `month_calplot` parameters.

### Infrastructure

- Renamed package from `plotly_calplot` to `plotly_calheatmap`.
- Updated Python version requirement to `>=3.9`.
- Modernized CI/CD workflows (GitHub Actions v4/v5, multi-version test matrix, PyPI Trusted Publishers).



