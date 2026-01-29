import numpy as np
import pandas as pd

from plotly_calheatmap import hourly_calheatmap

# Generate hourly server metrics for 2 years
np.random.seed(42)
dates = pd.date_range("2023-01-01", "2024-12-31", freq="h")

df = pd.DataFrame(
    {
        "datetime": dates,
        "cpu": np.random.uniform(5, 95, len(dates)).round(1),
        "memory": np.random.uniform(20, 85, len(dates)).round(1),
    }
)

fig = hourly_calheatmap(
    data=df,
    x="datetime",
    y="cpu",
    datasets={
        "CPU Usage": {
            "y": "cpu",
            "colorscale": "reds",
            "showscale": "CPU %",
            "cmap_min": 0,
            "cmap_max": 100,
        },
        "Memory Usage": {
            "y": "memory",
            "colorscale": "blues",
            "showscale": "Memory %",
            "cmap_min": 0,
            "cmap_max": 100,
        },
    },
    navigation=True,
    dark_theme=True,
    cols=4,
    title="Server Metrics",
)

fig.show()
