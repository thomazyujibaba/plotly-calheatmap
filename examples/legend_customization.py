import numpy as np
import pandas as pd

from plotly_calheatmap import calheatmap

# Generate daily data for two years
np.random.seed(42)
dates = pd.date_range("2023-01-01", "2024-12-31")
values = np.random.poisson(lam=5, size=len(dates))

df = pd.DataFrame({"date": dates, "commits": values})

# ── Discrete clickable legend (quantize scale) ──────────────────────
# Click any legend item to hide/show that category on the calendar.
fig = calheatmap(
    data=df,
    x="date",
    y="commits",
    title="Click legend items to toggle categories",
    colors=["#ebedf0", "#9be9a8", "#40c463", "#30a14e", "#216e39"],
    scale_type="quantize",
    legend_style="legend",
    legend_options={"title": {"text": "Commits"}, "orientation": "v"},
    gap=2,
)
fig.show()

# ── Discrete clickable legend with categorical bins ─────────────────
fig2 = calheatmap(
    data=df,
    x="date",
    y="commits",
    title="Categorical bins with clickable legend",
    scale_type="categorical",
    bins=[
        (0, 3, "#ebedf0"),
        (3, 6, "#9be9a8"),
        (6, 9, "#40c463"),
        (9, 15, "#216e39"),
    ],
    legend_style="legend",
    legend_options={"orientation": "h", "y": -0.12, "x": 0.5, "xanchor": "center"},
    gap=2,
)
fig2.show()

# ── Colorbar customization (vertical, right side) ──────────────────
fig3 = calheatmap(
    data=df,
    x="date",
    y="commits",
    title="Colorbar — vertical, right side",
    colorscale="greens",
    showscale="Commits",
    colorbar_options={
        "orientation": "v",
        "x": 1.02,
        "y": 0.5,
        "yanchor": "middle",
        "thickness": 12,
        "len": 0.8,
        "tickformat": "d",
    },
    gap=2,
)
fig3.show()
