import numpy as np
import pandas as pd

from plotly_calheatmap import calheatmap

# Generate daily data for one year
np.random.seed(42)
dates = pd.date_range("2024-01-01", "2024-12-31")
values = np.random.poisson(lam=4, size=len(dates))

df = pd.DataFrame({"date": dates, "value": values})

# Wall-calendar layout: 12 mini-calendars 4x3 grid
fig = calheatmap(
    data=df,
    x="date",
    y="value",
    colorscale="greens",
    title="Wall Calendar Heatmap",
    gap=2,
    week_start="monday",
    layout="calendar",
    cols=4,
)
fig.show()
