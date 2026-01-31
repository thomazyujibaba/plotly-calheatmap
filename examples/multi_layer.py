"""Multi-layer calendar heatmap example.

Scenario: A company tracks Revenue and Expenses across 2024.
Revenue dominates Jan–Jun, Expenses dominate Jul–Dec, and they
overlap in May–Aug (shown in green as summed values).

- Revenue → blue gradient  (Jan–Aug)
- Expenses → red gradient  (May–Dec)
- Both on same day → green gradient (May–Aug overlap, summed)
"""

import numpy as np
import pandas as pd
from plotly_calheatmap import calheatmap

rng = np.random.default_rng(42)

all_days = pd.date_range("2024-01-01", "2024-12-31")

# --- Revenue: Jan through Aug, dense on weekdays ---
revenue_dates = []
revenue_values = []
for d in all_days:
    if d.month > 8:
        continue
    if d.weekday() >= 5:
        if rng.random() < 0.30:
            revenue_dates.append(d)
            revenue_values.append(round(rng.uniform(100, 600), 2))
    else:
        if rng.random() < 0.90:
            base = 400 if d.month <= 4 else 200  # stronger early in the year
            revenue_dates.append(d)
            revenue_values.append(round(rng.uniform(base, base + 1200), 2))

revenue = pd.DataFrame({"date": revenue_dates, "value": revenue_values})

# --- Expenses: May through Dec, dense on weekdays ---
expense_dates = []
expense_values = []
for d in all_days:
    if d.month < 5:
        continue
    if d.day == 1:
        # Rent / fixed costs on the 1st
        expense_dates.append(d)
        expense_values.append(round(rng.uniform(1000, 1800), 2))
    elif d.weekday() < 5 and rng.random() < 0.85:
        base = 150 if d.month <= 8 else 300  # heavier towards year-end
        expense_dates.append(d)
        expense_values.append(round(rng.uniform(base, base + 800), 2))
    elif d.weekday() >= 5 and rng.random() < 0.25:
        expense_dates.append(d)
        expense_values.append(round(rng.uniform(50, 300), 2))

expenses = pd.DataFrame({"date": expense_dates, "value": expense_values})

print(f"Revenue entries : {len(revenue)}")
print(f"Expense entries : {len(expenses)}")
overlap = set(revenue["date"]) & set(expenses["date"])
print(f"Overlap days    : {len(overlap)}")

fig = calheatmap(
    layers=[
        {"data": revenue, "x": "date", "y": "value",
         "colorscale": [[0, "#e0f0ff"], [1, "#0840b0"]], "name": "Revenue"},
        {"data": expenses, "x": "date", "y": "value",
         "colorscale": [[0, "#ffe0e0"], [1, "#b01020"]], "name": "Expenses"},
    ],
    overlap_colorscale=[[0, "#e0ffe0"], [1, "#108020"]],
    title="Revenue vs Expenses — 2024",
    gap=3,
    dark_theme=True,
)

fig.show()
