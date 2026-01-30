"""Revenue data with log_scale and clickable legend.

Tests log_scale + legend + hovertemplate with R$ formatting.
"""

import numpy as np
import pandas as pd

from plotly_calheatmap import calheatmap

# --- Generate log-normal daily revenue (simulating e-commerce) ---
np.random.seed(7)
dates = pd.date_range("2023-01-01", "2024-12-31")
values = np.random.lognormal(mean=4, sigma=1.5, size=len(dates)).round(2)
values[np.random.choice(len(values), size=30, replace=False)] = 0

df = pd.DataFrame({"date": dates, "revenue": values})

# ── 1. log_scale + quantize + legend ─────────────────────────────────
fig1 = calheatmap(
    data=df,
    x="date",
    y="revenue",
    title="Receita diária — log_scale + quantize + legend",
    colors=["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    scale_type="quantize",
    zero_color="#f0f0f0",
    log_scale=True,
    locale="pt_BR",
    legend_style="legend",
    legend_options={
        "title": {"text": "Receita"},
        "orientation": "h",
        "y": -0.12,
        "x": 0.5,
        "xanchor": "center",
    },
    hovertemplate="<b>{date:%d/%m/%Y}</b><br>R$ {value:.,2f}",
    gap=2,
)
fig1.show()

# ── 2. log_scale + categorical bins + legend ─────────────────────────
fig2 = calheatmap(
    data=df,
    x="date",
    y="revenue",
    title="Receita diária — log_scale + categorical + legend",
    scale_type="categorical",
    bins=[
        (0, 0, "#f0f0f0"),
        (0.01, 100, "#c6dbef"),
        (100.01, 1000, "#6baed6"),
        (1000.01, 5000, "#2171b5"),
        (5000.01, float("inf"), "#08306b"),
    ],
    log_scale=True,
    legend_style="legend",
    legend_options={"title": {"text": "Receita"}, "orientation": "v"},
    hovertemplate="<b>{date:%d/%m/%Y}</b><br>R$ {value:,.2f}",
    gap=2,
)
fig2.show()

# ── 3. log_scale + colorbar ──────────────────────────────────────────
fig3 = calheatmap(
    data=df,
    x="date",
    y="revenue",
    title="Receita diária — log_scale + colorbar",
    colorscale="blues",
    log_scale=True,
    showscale="Receita",
    colorbar_options={
        "orientation": "v",
        "x": 1.02,
        "y": 0.5,
        "yanchor": "middle",
        "thickness": 12,
        "len": 0.8,
    },
    hovertemplate="<b>{date:%d/%m/%Y}</b><br>R$ {value:,.2f}",
    gap=2,
)
fig3.show()
