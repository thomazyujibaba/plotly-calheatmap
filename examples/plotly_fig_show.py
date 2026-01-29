import numpy as np
import pandas as pd

from plotly_calheatmap.calheatmap import calheatmap, month_calheatmap

# mock setup
dummy_start_date = "2019-01-01"
dummy_end_date = "2022-10-03"
dummy_df = pd.DataFrame(
    {
        "ds": pd.date_range(dummy_start_date, dummy_end_date, tz="Singapore"),
        "value": np.random.randint(
            -10,
            30,
            (pd.to_datetime(dummy_end_date) - pd.to_datetime(dummy_start_date)).days
            + 1,
        ),
    }
)

fig1 = calheatmap(
    dummy_df,
    x="ds",
    y="value",
    showscale="Temperature (°C)",
    scale_ticks=True,
    top_bottom_lines=True,
)

fig1.show()

# same example by month
fig2 = month_calheatmap(
    dummy_df,
    x="ds",
    y="value",
)
fig2.show()
