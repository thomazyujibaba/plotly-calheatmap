import numpy as np
import pandas as pd

from plotly_calheatmap import calheatmap

# Generate sample data with multiple business metrics
np.random.seed(42)
dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")

df = pd.DataFrame(
    {
        "date": dates,
        "sales": np.random.randint(50, 1500, len(dates)),
        "activity": np.random.uniform(0, 12, len(dates)).round(1),
        "temperature": np.random.uniform(10, 38, len(dates)).round(1),
    }
)

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
        "Employee Activity": {
            "y": "activity",
            "colorscale": "blues",
            "showscale": "Activity (hours)",
        },
        "Temperature": {
            "y": "temperature",
            "colorscale": [
                [0.0, "#313695"],
                [0.25, "#74add1"],
                [0.5, "#ffffbf"],
                [0.75, "#f46d43"],
                [1.0, "#a50026"],
            ],
            "showscale": "Temperature (°C)",
            "cmap_min": 10,
            "cmap_max": 38,
        },
    },
    dark_theme=True,
    navigation=True,
    title="Business Metrics Dashboard",
    gap=3,
    month_lines=True,
    total_height=200,
    nav_options={
        "bgcolor": "#30363d",
        "bordercolor": "#484f58",
        "font": {"color": "#c9d1d9", "size": 11},
    },
    dataset_nav_options={
        "bgcolor": "#30363d",
        "bordercolor": "#484f58",
        "font": {"color": "#c9d1d9", "size": 11},
    },
)

fig.show()
