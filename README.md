# plotly-calheatmap

A continuation of [plotly-calplot](https://github.com/brunorosilva/plotly-calplot) by Bruno Rocha Silva, which is no longer actively maintained.

This project picks up where plotly-calplot left off, providing an interactive calendar heatmap built with Plotly — similar to the contribution graphs on GitHub and GitLab profile pages.

## Features

- Interactive calendar heatmaps built with Plotly
- **Built-in aggregation** — pass raw event data with `agg="sum"|"mean"|"count"|"max"` instead of pre-aggregating
- **Vertical orientation** — render months as rows with `vertical=True`
- **Hourly heatmap** — `hourly_calheatmap()` for hour × day grids per month
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

### Hourly Heatmap

```python
from plotly_calheatmap import hourly_calheatmap

fig = hourly_calheatmap(df, x="datetime_col", y="value")
fig.show()
```

## Credits

This project is based on the original work by [Bruno Rocha Silva](https://github.com/brunorosilva) — [plotly-calplot](https://github.com/brunorosilva/plotly-calplot).
