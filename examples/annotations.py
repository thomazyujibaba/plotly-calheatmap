import numpy as np
import pandas as pd

from plotly_calheatmap import calheatmap

# Generate one year of daily data with small integer values
np.random.seed(0)
dates = pd.date_range("2024-01-01", "2024-12-31")
values = np.random.randint(0, 20, size=len(dates)).astype(float)
# Sprinkle some NaNs for missing days
values[np.random.choice(len(values), size=30, replace=False)] = np.nan

df = pd.DataFrame({"date": dates, "score": values})

# Calendar heatmap with cell annotations
# Use a larger gap and generous height so that the text fits inside cells.
fig = calheatmap(
    data=df,
    x="date",
    y="score",
    title="Daily scores — annotated cells",
    colorscale="ylgn",
    gap=4,
    total_height=280,
    annotations=True,
    annotations_fmt="%{z:.0f}",
    showscale="Score",
    month_lines=True,
)

fig.show()
