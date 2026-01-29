import numpy as np
import pandas as pd

from plotly_calheatmap import hourly_calheatmap

# Generate synthetic hourly temperature data for 2 years
dates = pd.date_range("2022-01-01", "2023-12-31 23:00:00", freq="h")
np.random.seed(42)
# Simulate daily temperature cycle: cooler at night, warmer during the day
hours = dates.hour
base_temp = 15 + 10 * np.sin((hours - 6) * np.pi / 12)
seasonal = 8 * np.sin((dates.dayofyear - 80) * 2 * np.pi / 365)
noise = np.random.normal(0, 2, len(dates))
temps = base_temp + seasonal + noise

df = pd.DataFrame({"datetime": dates, "temperature": temps})

# With navigation buttons to switch between years
fig = hourly_calheatmap(
    df,
    x="datetime",
    y="temperature",
    title="Hourly Temperature Heatmap",
    name="Temperature (°C)",
    colorscale="viridis",
    showscale=True,
    navigation=True,
)
fig.show()
