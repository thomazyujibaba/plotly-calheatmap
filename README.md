# plotly-calheatmap

A continuation of [plotly-calplot](https://github.com/brunorosilva/plotly-calplot) by Bruno Rocha Silva, which is no longer actively maintained.

This project picks up where plotly-calplot left off, providing an interactive calendar heatmap built with Plotly — similar to the contribution graphs on GitHub and GitLab profile pages.

## Features

- Interactive calendar heatmaps built with Plotly
- **Multi-layer heatmap** — overlay multiple DataFrames on one calendar with distinct color gradients per source; overlap days are summed and shown in a third colorscale (`layers` parameter)
- **Dataset swap** — switch between multiple metrics via dropdown without regenerating the graph (`datasets` parameter)
- **Built-in aggregation** — pass raw event data with `agg="sum"|"mean"|"count"|"max"` instead of pre-aggregating
- **Logarithmic color scale** — `log_scale=True` applies `log(1+x)` so extreme values don't wash out the heatmap
- **Custom time groupings** — `grouping="quarter"|"bimester"|"semester"` draws separator lines and replaces axis labels
- **Vertical orientation** — render months as rows with `vertical=True`
- **Hourly heatmap** — `hourly_calheatmap()` for hour × day grids per month
- **Skip empty years** — `skip_empty_years=True` excludes years with no data
- **Replace NaNs with zeros** — `replace_nans_with_zeros=True` displays empty dates as 0
- **Top & bottom border lines** — `top_bottom_lines=True` draws horizontal lines enclosing each month
- **Month gap spacing** — extra visual separation between months via `month_gap`
- **Custom week start day** — `week_start="sunday"` or `"saturday"` to change the first day of the week (default: `"monday"`)
- **Wall-calendar layout** — `layout="calendar"` renders a grid of mini-calendars with days as columns and weeks as rows
- **Multi-year support** with independent tick configurations per subplot
- **Year navigation buttons** (`navigation=True`)
- **Localization support** (`locale` parameter) for month and day names (e.g. `pt_BR`, `es`, `fr`)
- **Customizable hovertemplate** with friendly `{placeholder}` syntax and `customdata` columns
- **Smart colorscales** — pass a `colors` list and `scale_type` (`"linear"`, `"quantile"`, `"quantize"`, `"diverging"`, `"categorical"`) for automatic interval computation
- **Cell annotations** — `annotations=True` or `annotations_fmt="%{z:.0f}"` displays values/labels inside each cell
- **Zero-value distinction** — `zero_color` gives 0-value cells a dedicated color while missing data stays transparent
- **Missing-data styling** — `nan_color` assigns a dedicated color to NaN/missing cells, distinguishing them from zero-value cells
- **Responsive / auto-sizing** — all chart types adapt width to the container automatically; height is computed from the data (overridable via `total_height` and `width`)
- **Fully customizable colorscales** (including custom lists)
- Month separator lines, configurable month label placement, and color scale with label/ticks
- Flexible layout options: `gap`, `margin`, `font_*`, `paper_bgcolor`, `plot_bgcolor`, etc.

## Documentation

For the full API reference (all parameters for `calheatmap()` and `hourly_calheatmap()`), see [docs/API.md](docs/API.md).

## Installation

```bash
pip install plotly-calheatmap
```

## Quick Start

```python
from plotly_calheatmap import calheatmap

fig = calheatmap(df, x="date", y="value")
fig.show()
```

<img src="https://github.com/brunorosilva/plotly-calplot/blob/main/assets/images/example.png?raw=true">

### Built-in Aggregation

Pass raw (non-aggregated) event data directly — duplicate dates are grouped and aggregated automatically:

```python
from plotly_calheatmap import calheatmap

# df has multiple rows per date (e.g. individual transactions)
fig = calheatmap(df, x="date", y="amount", agg="sum")
fig.show()
```

Supported functions: `"sum"`, `"mean"`, `"count"`, `"max"`.

### Vertical Orientation

```python
fig = calheatmap(df, x="date", y="value", vertical=True, month_gap=1)
```

### Dataset Swap

Switch between multiple metrics on the same graph via a dropdown menu. Each dataset can have its own colorscale, value range, and legend title. Works with both `calheatmap()` and `hourly_calheatmap()`, alongside year navigation.

```python
fig = calheatmap(
    data=df,
    x="date",
    y="sales",
    datasets={
        "Sales": {
            "y": "sales",
            "colorscale": "greens",
            "showscale": "Sales ($)",
            "cmap_min": 0,
        },
        "Activity": {
            "y": "activity",
            "colorscale": "blues",
            "showscale": "Activity (hours)",
        },
    },
    navigation=True,
)
```

<img src="https://github.com/thomazyujibaba/plotly-calheatmap/blob/master/assets/images/swap_legends.jpg?raw=true">

### Logarithmic Color Scale

When a few extreme values wash out the rest of the heatmap, use `log_scale=True` to apply `log(1+x)` to the color mapping. Hover text still shows original values.

```python
fig = calheatmap(df, x="date", y="value", log_scale=True)
```

### Smart Colorscales & Zero Distinction

Instead of manually defining colorscale intervals, pass a list of colors and let the library compute the positions automatically:

```python
# Old way (manual positions)
colorscale=[[0.0, "#161b22"], [0.01, "#0e4429"], [0.25, "#006d32"], [1.0, "#39d353"]]

# New way (automatic)
colors=["#0e4429", "#006d32", "#26a641", "#39d353"],
zero_color="#161b22"  # dedicated color for value 0
```

**Scale types** control how colors are distributed:

```python
# linear (default): colors evenly spaced across data range
fig = calheatmap(df, x="date", y="value",
    colors=["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"])

# quantile: each color covers equal number of data points (good for skewed data)
fig = calheatmap(df, x="date", y="value",
    colors=["#edf8e9", "#bae4b3", "#74c476", "#006d2c"],
    scale_type="quantile")

# quantize: data range split into equal mathematical intervals
fig = calheatmap(df, x="date", y="value",
    colors=["#edf8e9", "#bae4b3", "#74c476", "#006d2c"],
    scale_type="quantize")

# diverging: two gradients meeting at a pivot point
fig = calheatmap(df, x="date", y="temp",
    colors=["#2166ac", "#f7f7f7", "#b2182b"],
    scale_type="diverging", pivot=20, symmetric=True)

# categorical: user-defined bins with explicit color per range
fig = calheatmap(df, x="date", y="value",
    scale_type="categorical",
    bins=[(0, 0, "gray"), (1, 3, "lightgreen"), (4, float("inf"), "darkgreen")])
```

**Zero-value distinction** — give cells with value 0 a dedicated color, separate from missing data (which stays transparent):

```python
fig = calheatmap(df, x="date", y="commits",
    colors=["#0e4429", "#006d32", "#26a641", "#39d353"],
    zero_color="#161b22")
```

**Missing-data styling** — give NaN/missing cells a visible color instead of leaving them transparent:

```python
fig = calheatmap(df, x="date", y="commits",
    colors=["#0e4429", "#006d32", "#26a641", "#39d353"],
    zero_color="#161b22",
    nan_color="#0d1117")
```

All new parameters also work inside the `datasets` dict for per-metric configuration.

### Cell Annotations

Display values or labels inside each cell:

```python
# Show integer values inside cells
fig = calheatmap(df, x="date", y="score",
    annotations=True, annotations_fmt="%{z:.0f}",
    gap=4, total_height=280)

# Simple boolean — shows raw z values
fig = calheatmap(df, x="date", y="score", annotations=True)
```


<img src="https://github.com/thomazyujibaba/plotly-calheatmap/blob/master/assets/images/cell_annotation.png?raw=true">


### Custom Time Groupings

Draw thicker separator lines at quarter, bimester, or semester boundaries and replace axis labels with group names:

```python
fig = calheatmap(df, x="date", y="value", grouping="quarter", month_lines=True)
```

<img src="https://github.com/thomazyujibaba/plotly-calheatmap/blob/master/assets/images/quarter_group.jpg?raw=true">

### Hourly Heatmap

```python
from plotly_calheatmap import hourly_calheatmap

fig = hourly_calheatmap(df, x="datetime_col", y="value")
fig.show()
```

<img src="https://github.com/thomazyujibaba/plotly-calheatmap/blob/master/assets/images/hourly_heatmap.jpg?raw=true">

### Skip Empty Years & Replace NaNs

```python
fig = calheatmap(df, x="date", y="value", skip_empty_years=True, replace_nans_with_zeros=True)
```

### Border Lines

Fully enclose each month by combining `month_lines` with `top_bottom_lines`:

```python
fig = calheatmap(df, x="date", y="value", month_lines=True, top_bottom_lines=True)
```

### Custom Week Start Day

Choose which day the week starts on — `"monday"` (default, ISO 8601), `"sunday"` (US convention), or `"saturday"`:

```python
fig = calheatmap(df, x="date", y="value", week_start="sunday")
```

### Multi-Layer Heatmap

Overlay two (or more) DataFrames on one calendar. Each source gets its own color gradient; days present in both are summed and shown with a third colorscale:

```python
fig = calheatmap(
    layers=[
        {"data": revenue,  "x": "date", "y": "value", "colorscale": "blues",  "name": "Revenue"},
        {"data": expenses, "x": "date", "y": "value", "colorscale": "reds",   "name": "Expenses"},
    ],
    overlap_colorscale="greens",
)
```

Hover shows per-source values and the combined total for overlap days.

<img src="https://github.com/thomazyujibaba/plotly-calheatmap/blob/master/assets/images/multi_layer.png?raw=true">


### Wall-Calendar Layout

Render a grid of mini-calendars (one per month), each looking like a standard wall calendar with days-of-week as columns and weeks as rows:

```python
fig = calheatmap(df, x="date", y="value", layout="calendar", cols=4)
```

<img src="https://github.com/thomazyujibaba/plotly-calheatmap/blob/master/assets/images/wall_calendar.png?raw=true">

## Credits

This project is based on the original work by [Bruno Rocha Silva](https://github.com/brunorosilva) — [plotly-calplot](https://github.com/brunorosilva/plotly-calplot).
