# Changelog

## 0.3.1

### Bug Fixes
Fixed Pypi setup

## 0.3

### New Features

- **Skip empty years** — New `skip_empty_years` parameter to exclude years with no data from the plot. *(Originally from [juan11iguel/plotly-calplot](https://github.com/juan11iguel/plotly-calplot).)*
- **Replace NaNs with zeros** — New `replace_nans_with_zeros` parameter to display dates without entries as 0 instead of NaN.
- **Top & bottom border lines** — New `top_bottom_lines` parameter draws horizontal lines at the top and bottom edges of the calendar. Combined with `month_lines`, this fully encloses each month. *(Originally from [GiantMolecularCloud/plotly-calplot](https://github.com/GiantMolecularCloud/plotly-calplot).)*
- **End-of-year month line** — When `month_lines` is enabled, a closing line is now drawn at the end of the last month. *(Originally from [GiantMolecularCloud/plotly-calplot](https://github.com/GiantMolecularCloud/plotly-calplot).)*
- **Custom time groupings** — New `grouping` parameter accepts `"quarter"`, `"bimester"`, or `"semester"` to draw thicker separator lines at group boundaries and replace axis labels with group names (e.g. Q1, Q2). Month lines are preserved alongside the group lines. Customize with `grouping_lines_width` and `grouping_lines_color`.
- **Logarithmic color scale** — New `log_scale` parameter applies `log(1+x)` to the color mapping so that a few extreme values don't wash out the rest of the heatmap. Hover text still shows original values.
- **Dataset swap** — New `datasets` parameter for both `calheatmap()` and `hourly_calheatmap()` enables switching between multiple metrics (e.g. Sales vs Activity) on the same graph via a dropdown menu, without regenerating the figure. Each dataset can have its own colorscale, value range, and legend title. Works alongside year navigation. New examples: `dataset_swap.py`, `hourly_dataset_swap.py`.

### Bug Fixes

- **Fix tooltip on gap cells** — Cells outside the valid week range no longer show raw template strings (e.g. `%{customdata[0]}`) in the tooltip. Added `hoverongaps=False` to suppress hover on empty gap-fill cells.
---

## 0.2

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



