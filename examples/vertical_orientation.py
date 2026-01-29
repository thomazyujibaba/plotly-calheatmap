import numpy as np
import pandas as pd

from plotly_calheatmap import calheatmap

# Generate daily data for one year
np.random.seed(42)
dates = pd.date_range("2024-01-01", "2024-12-31")
values = np.random.poisson(lam=4, size=len(dates))

df = pd.DataFrame({"date": dates, "value": values})

fig = calheatmap(
    data=df,
    x="date",
    y="value",
    colorscale="greens",
    title="Vertical Calendar Heatmap",
    vertical=True,
    gap=2,
    month_gap=3,
    total_height=800,
    width=250,
    showscale=True,
    month_lines=False,
)
fig.show()
