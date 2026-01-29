"""Demonstrates all colorscale types and zero-value distinction.

Scale types:
  - linear (default): Colors evenly spaced across data range
  - quantile: Each color covers equal number of data points
  - quantize: Data range split into equal mathematical intervals
  - diverging: Two gradients meeting at a pivot point

Zero color:
  - zero_color gives value 0 a dedicated color while NaN stays transparent
"""

import numpy as np
import pandas as pd

from plotly_calheatmap import calheatmap

# --- Generate sample data ---
np.random.seed(42)
dates = pd.date_range("2024-01-01", "2024-12-31")
# Skewed distribution: lots of small values, few large ones
values = np.random.exponential(scale=5, size=len(dates)).astype(int)
# Sprinkle some zeros
values[np.random.choice(len(values), size=50, replace=False)] = 0

df = pd.DataFrame({"date": dates, "value": values})

# --- 1. Linear scale with zero color ---
fig_linear = calheatmap(
    data=df,
    x="date",
    y="value",
    colors=["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"],
    zero_color="#f0f0f0",
    scale_type="linear",
    title="Linear scale (zero = gray)",
    gap=3,
    total_height=150,
)
fig_linear.show()

# --- 2. Quantile scale ---
fig_quantile = calheatmap(
    data=df,
    x="date",
    y="value",
    colors=["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"],
    zero_color="#f0f0f0",
    scale_type="quantile",
    title="Quantile scale (each color = equal data points)",
    gap=3,
    total_height=150,
)
fig_quantile.show()

# --- 3. Quantize scale ---
fig_quantize = calheatmap(
    data=df,
    x="date",
    y="value",
    colors=["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"],
    zero_color="#f0f0f0",
    scale_type="quantize",
    title="Quantize scale (equal value intervals)",
    gap=3,
    total_height=150,
)
fig_quantize.show()

# --- 4. Diverging scale ---
# Temperature anomaly data centered around 20
temps = np.random.normal(loc=20, scale=8, size=len(dates)).round(1)
df_temp = pd.DataFrame({"date": dates, "temp": temps})

fig_diverging = calheatmap(
    data=df_temp,
    x="date",
    y="temp",
    colors=["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
    scale_type="diverging",
    pivot=20,
    symmetric=True,
    title="Diverging scale (pivot=20, symmetric)",
    gap=3,
    total_height=150,
)
fig_diverging.show()
