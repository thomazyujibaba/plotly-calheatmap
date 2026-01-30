# API Reference

## `calheatmap()`

Create an interactive yearly calendar heatmap (GitHub-style contribution graph).

```python
from plotly_calheatmap import calheatmap

fig = calheatmap(df, x="date", y="value")
fig.show()
```

**Returns** → `plotly.graph_objects.Figure`

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `DataFrame` | Must contain at least one date column and one numeric value column. |
| `x` | `str` | Name of the date column. |
| `y` | `str` | Name of the value column. |

### Appearance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dark_theme` | `bool` | `False` | Use a dark background theme. |
| `colorscale` | `str \| list` | `"greens"` | Plotly colorscale name or custom list (e.g. `[[ 0, "#eee"], [1, "#333"]]`). |
| `colors` | `list[str] \| None` | `None` | List of colors for automatic colorscale computation (used with `scale_type`). |
| `scale_type` | `str \| None` | `"linear"` | Colorscale distribution: `"linear"`, `"quantile"`, `"quantize"`, `"diverging"`, or `"categorical"`. |
| `bins` | `list[tuple] \| None` | `None` | Custom bin boundaries for categorical scale type. |
| `zero_color` | `str \| None` | `None` | Dedicated color for zero-value cells. |
| `nan_color` | `str \| None` | `None` | Color for missing/NaN cells. |
| `gap` | `int` | `1` | Gap between daily squares in pixels. |
| `month_lines` | `bool` | `True` | Draw separation lines between months. |
| `month_lines_width` | `int` | `1` | Width of month separation lines. |
| `month_lines_color` | `str` | `"#9e9e9e"` | Color of month separation lines. |
| `top_bottom_lines` | `bool` | `False` | Draw horizontal lines at top/bottom edges of the calendar. |
| `showscale` | `bool \| str` | `False` | Show color legend. Pass a string (e.g. `"Temperature"`) to add a legend title. |
| `scale_ticks` | `bool` | `False` | Show tick marks on the color scale. |

### Layout

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_height` | `int \| None` | `None` | Force a specific figure height in pixels. Auto-calculated if `None`. |
| `width` | `int \| None` | `None` | Figure width in pixels. |
| `margin` | `dict \| None` | `None` | Custom margins, e.g. `{"l": 40, "r": 20, "t": 40, "b": 20}`. |
| `space_between_plots` | `float` | `0.08` | Vertical space between year subplots. |
| `vertical` | `bool` | `False` | Render months as rows instead of columns. |
| `month_gap` | `int` | `0` | Extra spacing (in week-units) between months. |
| `years_as_columns` | `bool` | `False` | Plot all years side by side in a single row. |
| `paper_bgcolor` | `str \| None` | `None` | Paper background color (e.g. `"#0d1117"`). |
| `plot_bgcolor` | `str \| None` | `None` | Plot background color. |

### Labels & Titles

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"y"` | Metric name used in hover template. |
| `title` | `str` | `""` | Figure title. |
| `years_title` | `bool` | `False` | Show the year as a title on each subplot. |
| `month_labels_side` | `str` | `"bottom"` | Position of month labels: `"top"` or `"bottom"`. |
| `locale` | `str \| None` | `None` | Locale for month/day names (e.g. `"pt_BR"`, `"es"`). |
| `font_color` | `str \| None` | `None` | Override font color. |
| `font_size` | `int \| None` | `None` | Override font size. |
| `title_font_color` | `str \| None` | `None` | Override title font color. |
| `title_font_size` | `int \| None` | `None` | Override title font size. |

### Color Range

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cmap_min` | `float \| None` | `None` | Colormap minimum. Defaults to data minimum. |
| `cmap_max` | `float \| None` | `None` | Colormap maximum. Defaults to data maximum. |
| `log_scale` | `bool` | `False` | Apply `log(1 + x)` color scale. Hover still shows original values. |
| `pivot` | `float \| None` | `None` | Center point for diverging colorscales. |
| `symmetric` | `bool` | `False` | Make color range symmetric around the pivot. |

### Hover & Annotations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hovertemplate` | `str \| None` | `None` | Custom hover template. Supports `{date}`, `{date:%d/%m/%Y}`, `{name}`, `{value}`, `{week}`, `{text}`, `{col}` placeholders. Raw Plotly syntax (`%{z}`) also works. |
| `customdata` | `list[str] \| None` | `None` | Extra columns to include as customdata, available as `{column_name}` in hover. |
| `text` | `str \| None` | `None` | Column name to include in hover text. |
| `annotations` | `bool` | `False` | Display cell values as text inside each cell. |
| `annotations_fmt` | `str \| None` | `None` | Format string for annotations (e.g. `"%{z:.0f}"`). Implies `annotations=True`. |
| `annotations_font_size` | `int \| None` | `None` | Annotation font size. Defaults to 10. |
| `annotations_font_color` | `str \| None` | `None` | Annotation font color. |
| `annotations_font_family` | `str \| None` | `None` | Annotation font family. |

### Aggregation & Grouping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agg` | `str \| None` | `None` | Aggregation when multiple rows share a date: `"sum"`, `"mean"`, `"count"`, or `"max"`. |
| `grouping` | `str \| None` | `None` | Time grouping for separator lines: `"month"`, `"bimester"`, `"quarter"`, or `"semester"`. |
| `grouping_lines_width` | `int` | `2` | Width of grouping boundary lines. |
| `grouping_lines_color` | `str \| None` | `None` | Color of grouping boundary lines. Defaults to `month_lines_color`. |

### Navigation & Datasets

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `navigation` | `bool` | `False` | Show one year at a time with navigation buttons (GitHub-style). |
| `nav_options` | `dict \| None` | `None` | Styling overrides for navigation buttons (Plotly `updatemenus` keys). |
| `datasets` | `dict \| None` | `None` | Multiple datasets to swap via dropdown. Keys are labels, values are dicts with `"y"` (required), `"colorscale"`, `"showscale"`, `"cmap_min"`, `"cmap_max"`, `"name"` (all optional). |
| `dataset_nav_options` | `dict \| None` | `None` | Styling overrides for the dataset dropdown. |

### Other

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date_fmt` | `str` | `"%Y-%m-%d"` | Date format for parsing string dates. Ignored if already datetime. |
| `start_month` | `int` | `1` | Starting month (1 = January). |
| `end_month` | `int` | `12` | Ending month (12 = December). |
| `skip_empty_years` | `bool` | `False` | Skip years where the sum of `y` is less than 1. |
| `replace_nans_with_zeros` | `bool` | `False` | Show dates without data as 0 instead of NaN. |

---

## `hourly_calheatmap()`

Create an hourly heatmap with one subplot per month. Each subplot shows day of month (x-axis) vs hour of day (y-axis), colored by the aggregated value.

```python
from plotly_calheatmap import hourly_calheatmap

fig = hourly_calheatmap(df, x="datetime", y="value")
fig.show()
```

**Returns** → `plotly.graph_objects.Figure`

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `DataFrame` | Input data with a datetime column and a numeric value column. |
| `x` | `str` | Name of the datetime column. |
| `y` | `str` | Name of the value column. |

### Behavior

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agg` | `str` | `"mean"` | Aggregation for sub-hourly data: `"mean"`, `"sum"`, `"max"`, `"min"`, `"count"`. |
| `cols` | `int` | `4` | Number of columns in the subplot grid. |

### Appearance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dark_theme` | `bool` | `False` | Use dark background theme. |
| `colorscale` | `str \| list` | `"viridis"` | Plotly colorscale name or custom list. |
| `gap` | `int` | `1` | Gap between cells in pixels. |
| `showscale` | `bool` | `False` | Show the color bar. |
| `scale_ticks` | `list \| None` | `None` | Custom tick values for the color bar. |

### Layout

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str` | `""` | Figure title. |
| `total_height` | `int \| None` | `None` | Figure height in pixels. |
| `total_width` | `int \| None` | `None` | Figure width in pixels. |
| `margin` | `dict \| None` | `None` | Figure margins (`l`, `r`, `t`, `b`). |
| `paper_bgcolor` | `str \| None` | `None` | Paper background color. |
| `plot_bgcolor` | `str \| None` | `None` | Plot background color. |

### Labels & Fonts

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `""` | Metric name for hover. |
| `locale` | `str \| None` | `None` | Locale for month names. |
| `font_color` | `str \| None` | `None` | Font color. |
| `font_size` | `int \| None` | `None` | Font size. |
| `title_font_color` | `str \| None` | `None` | Title font color. |
| `title_font_size` | `int \| None` | `None` | Title font size. |

### Color Range

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cmap_min` | `float \| None` | `None` | Colormap minimum. |
| `cmap_max` | `float \| None` | `None` | Colormap maximum. |

### Hover & Navigation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hovertemplate` | `str \| None` | `None` | Custom hover template (Plotly format: `%{x}`, `%{y}`, `%{z}`). |
| `date_fmt` | `str` | `"%Y-%m-%d %H:%M:%S"` | Date format for parsing string dates. |
| `navigation` | `bool` | `False` | Show year navigation buttons. |
| `nav_options` | `dict \| None` | `None` | Styling overrides for navigation buttons. |
| `datasets` | `dict \| None` | `None` | Multiple datasets to swap via dropdown. Same format as `calheatmap()`. |
| `dataset_nav_options` | `dict \| None` | `None` | Styling overrides for the dataset dropdown. |
