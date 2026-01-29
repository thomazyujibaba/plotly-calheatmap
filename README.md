# plotly-calheatmap

A continuation of [plotly-calplot](https://github.com/brunorosilva/plotly-calplot) by Bruno Rocha Silva, which is no longer actively maintained.

This project picks up where plotly-calplot left off, providing an interactive calendar heatmap built with Plotly — similar to the contribution graphs on GitHub and GitLab profile pages.

## Original Description

Making it easier to visualize and customize time-relevant or time-series data with Plotly interaction.

## Installation

```bash
pip install plotly-calplot
```

## Quick Start

```python
from plotly_calplot import calplot

fig = calplot(df, x="date", y="value")
fig.show()
```

<img src="https://github.com/brunorosilva/plotly-calplot/blob/main/assets/images/example.png?raw=true">

## Credits

This project is based on the original work by [Bruno Rocha Silva](https://github.com/brunorosilva) — [plotly-calplot](https://github.com/brunorosilva/plotly-calplot).
