import numpy as np
import pandas as pd

from plotly_calheatmap import calheatmap

# Generate sample daily data for a full year
np.random.seed(42)
dates = pd.date_range("2024-01-01", "2024-12-31")
values = np.random.poisson(lam=5, size=len(dates))

df = pd.DataFrame({"date": dates, "value": values})

# Quarterly grouping — thicker lines at Q1/Q2/Q3/Q4 boundaries,
# with regular month lines still visible underneath.
fig = calheatmap(
    data=df,
    x="date",
    y="value",
    grouping="quarter",
    month_lines=True,
    title="Quarterly Grouping",
    colorscale="blues",
    gap=3,
    total_height=200,
)
fig.show()

# Semester grouping — only two groups (S1, S2)
fig = calheatmap(
    data=df,
    x="date",
    y="value",
    grouping="semester",
    month_lines=True,
    grouping_lines_width=3,
    grouping_lines_color="#e74c3c",
    title="Semester Grouping",
    colorscale="greens",
    gap=3,
    total_height=200,
)
fig.show()
