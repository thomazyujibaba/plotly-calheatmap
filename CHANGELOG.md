# Changelog

## 0.4.1

### New Features

- **Interactive filtering legend** — New `legend_style="legend"` parameter replaces the continuous colorbar with discrete, clickable legend items — one per color bin. Users can click legend entries to show/hide categories on the chart. Requires a discrete `scale_type` (`"quantile"`, `"quantize"`, or `"categorical"`). Customize placement and styling via the new `legend_options` parameter.
- **Colorbar customization** — New `colorbar_options` parameter allows overriding any Plotly colorbar property (`orientation`, `x`, `y`, `thickness`, `len`, `tickformat`, `title`, etc.) for both `calheatmap` and `month_calheatmap`.
- **Custom week start day** — New `week_start` parameter lets users choose `"monday"` (default, ISO 8601), `"sunday"` (US convention), or `"saturday"` as the first day of the week. Day labels, weekday positions, and week numbering all adjust automatically. New example: `week_start_day.py`.
- **Wall-calendar layout** — New `layout="calendar"` parameter on `calheatmap()` renders a grid of mini-calendars (one per month), each with days-of-week as columns and weeks as rows — like a standard wall calendar. Supports `week_start`, `cols` for grid layout, and all common styling parameters. New example: `calendar_layout.py`.
- **Multi-layer heatmap** — New `layers` parameter overlays multiple DataFrames on a single calendar, each with its own color gradient. Overlap days are automatically summed and rendered with a distinct colorscale (`overlap_colorscale`). Hover shows per-layer breakdown and total. Useful for comparing two related metrics (e.g. revenue vs expenses) on one chart. New example: `multi_layer.py`.
- **Toolbar layout for controls** — Dataset dropdown and year navigation buttons are now placed in a dedicated toolbar row at the top of the chart (dropdown left, title center, year buttons right-horizontal). The figure height grows automatically to accommodate the toolbar without compressing the plot area. Both controls remain fully customizable via `dataset_nav_options` and `nav_options`.

### Bug Fixes

- **Log-scale colorbar ticks** — Fixed colorbar displaying log-transformed tick values instead of original values when `log_scale=True` and `showscale` is enabled. Tick labels now show the reverse-transformed (`expm1`) values.

## 0.4

### New Features

- **Cell annotations** — New `annotations` and `annotations_fmt` parameters display text (values, labels) inside each cell. Use `annotations=True` for default value display or `annotations_fmt="%{z:.0f}"` for custom formatting. Works with both `calheatmap` and `month_calheatmap`. New example: `annotations.py`.
- **Smart colorscales** — New `colors` parameter accepts a simple list of colors (e.g. `["#0e4429", "#006d32", "#39d353"]`) and automatically computes the colorscale intervals from the data. Use with `scale_type` to control how colors are distributed:
  - `"linear"` (default): colors evenly spaced across data range
  - `"quantile"`: each color covers equal number of data points (good for skewed data)
  - `"quantize"`: data range split into equal mathematical intervals
  - `"diverging"`: two gradients meeting at a `pivot` point, with optional `symmetric` mode
  - `"categorical`: discrete bins range consisted on `(min, max, color)` tuples

- **Zero-value distinction** — New `zero_color` parameter assigns a dedicated color to cells with value 0, visually separating them from NaN/missing cells (which remain transparent). Works with both `colors` and manual `colorscale`.
- **Missing-data styling** — New `nan_color` parameter assigns a dedicated color to cells with no data (NaN), visually distinguishing them from cells with value 0. Works with both `colors` and manual `colorscale`, and inside the `datasets` dict.
- **Locale-aware value formatting** — `{value:FORMAT}` in `hovertemplate` now supports the `locale` parameter. E.g. with `locale="pt_BR"`, `{value:,.2f}` renders as `1.234,50` (Brazilian number format).
- **Responsive / auto-sizing** — All chart types (`calheatmap`, `month_calheatmap`, `hourly_calheatmap`) now use `autosize=True` by default, so the width automatically adapts to the container in notebooks, Dash apps, and web pages. Height is computed dynamically based on the number of years, months shown, and orientation. Users can still override with `width` and `total_height` when needed. Theme defaults and layout application are now centralized via shared `get_theme_defaults()` and `apply_figure_layout()` utilities.
- All new parameters also work inside the `datasets` dict for per-metric configuration.
- New example: `colorscale_types.py` demonstrating all scale types and zero-value distinction.

## 0.3.1

### Bug Fixes
- Fix Pypi setup
- plotly-calheatmap now requires Python 3.10+

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



