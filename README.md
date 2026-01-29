# plotly-calheatmap

A continuation of [plotly-calplot](https://github.com/brunorosilva/plotly-calplot) by Bruno Rocha Silva, which is no longer actively maintained.

This project picks up where plotly-calplot left off, providing an interactive calendar heatmap built with Plotly — similar to the contribution graphs on GitHub and GitLab profile pages.

## Features

- Interactive calendar heatmaps built with Plotly
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
- Multi-year support with independent tick configurations per subplot
- Year navigation buttons (`navigation=True`)
- Localization support (`locale` parameter) for month and day names (e.g. `pt_BR`, `es`, `fr`)
- Customizable hovertemplate with friendly `{placeholder}` syntax and `customdata` columns
- Fully customizable colorscales (including custom lists)
- Month separator lines, configurable month label placement, and color scale with label/ticks
- Flexible layout options: `gap`, `margin`, `font_*`, `paper_bgcolor`, `plot_bgcolor`, etc.

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

## Credits

This project is based on the original work by [Bruno Rocha Silva](https://github.com/brunorosilva) — [plotly-calplot](https://github.com/brunorosilva/plotly-calplot).
