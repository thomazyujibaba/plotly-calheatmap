import numpy as np
import pandas as pd

from plotly_calheatmap import calheatmap

# Simulate raw event data: multiple events per day (e.g. sales transactions)
np.random.seed(0)
dates = np.random.choice(pd.date_range("2024-01-01", "2024-12-31"), size=2000)
amounts = np.round(np.random.exponential(scale=50, size=len(dates)), 2)

df = pd.DataFrame({"date": dates, "amount": amounts})

# Without agg you would need to pre-aggregate:
#   df_daily = df.groupby("date", as_index=False).agg({"amount": "sum"})
#   calheatmap(df_daily, x="date", y="amount")
#
# With agg, just pass the raw data directly:

fig = calheatmap(
    data=df,
    x="date",
    y="amount",
    agg="sum",          # also try "mean", "count", "max"
    title="Daily Revenue (sum of transactions)",
    colorscale="blues",
    showscale="Revenue",
    scale_ticks=True,
    gap=3,
    total_height=200,
)

fig.show()
