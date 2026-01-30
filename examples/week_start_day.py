import numpy as np
import pandas as pd

from plotly_calheatmap import calheatmap

# Generate daily data for one year
np.random.seed(42)
dates = pd.date_range("2024-01-01", "2024-12-31")
values = np.random.poisson(lam=4, size=len(dates))

df = pd.DataFrame({"date": dates, "value": values})

# Default: Monday start (ISO 8601)
fig_monday = calheatmap(
    data=df,
    x="date",
    y="value",
    colorscale="greens",
    title="Monday Start (default)",
    week_start="monday",
)
fig_monday.show()

# Sunday start (US convention)
fig_sunday = calheatmap(
    data=df,
    x="date",
    y="value",
    colorscale="blues",
    title="Sunday Start",
    week_start="sunday",
)
fig_sunday.show()

# Saturday start
fig_saturday = calheatmap(
    data=df,
    x="date",
    y="value",
    colorscale="purples",
    title="Saturday Start",
    week_start="saturday",
)
fig_saturday.show()
