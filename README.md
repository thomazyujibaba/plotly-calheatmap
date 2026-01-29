# plotly-calheatmap

A continuation of [plotly-calplot](https://github.com/brunorosilva/plotly-calplot) by Bruno Rocha Silva, which is no longer actively maintained.

This project picks up where plotly-calplot left off, providing an interactive calendar heatmap built with Plotly — similar to the contribution graphs on GitHub and GitLab profile pages.

## Features

- Interactive calendar heatmaps built with Plotly
- Multi-year support with independent tick configurations per subplot
- Localization support (`locale` parameter) for month and day names (e.g. `pt_BR`, `es`, `fr`)
- Customizable hovertemplate with friendly `{placeholder}` syntax and `customdata` columns
- Fully customizable colorscales
- Year navigation buttons (`navigation=True`)
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

## Credits

This project is based on the original work by [Bruno Rocha Silva](https://github.com/brunorosilva) — [plotly-calplot](https://github.com/brunorosilva/plotly-calplot).
