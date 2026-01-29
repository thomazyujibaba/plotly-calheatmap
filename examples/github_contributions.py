import numpy as np
import pandas as pd

from plotly_calheatmap import calheatmap

# GitHub-style contribution graph spanning multiple years
end_date = pd.Timestamp.today().normalize()
start_date = end_date - pd.Timedelta(days=364 * 3)

np.random.seed(42)
dates = pd.date_range(start_date, end_date)
contributions = np.random.choice(
    [0, 0, 0, 0, 0, 1, 2, 3, 5, 8, 10],
    size=len(dates),
)

repos = np.random.choice(
    ["frontend", "backend", "docs", "infra"],
    size=len(dates),
)

df = pd.DataFrame(
    {
        "date": dates,
        "contributions": contributions,
        "repo": repos,
    }
)

fig = calheatmap(
    data=df,
    x="date",
    y="contributions",
    dark_theme=True,
    colorscale=[
        [0.0, "#161b22"],
        [0.01, "#0e4429"],
        [0.25, "#006d32"],
        [0.5, "#26a641"],
        [1.0, "#39d353"],
    ],
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font_color="#8b949e",
    font_size=11,
    title="Contributions",
    title_font_color="#007af4",
    title_font_size=14,
    width=900,
    total_height=200,
    margin={"l": 40, "r": 20, "t": 40, "b": 20},
    month_labels_side="top",
    gap=3,
    month_lines=True,
    navigation=True,
    nav_options={
        "bgcolor": "#c0c0c0",
        "bordercolor": "#ccc",
        "font": {"color": "#000", "size": 11},
    },
    showscale="Commits",
    scale_ticks=True,
    # Custom hover using friendly {placeholder} syntax:
    #   {date}             -> date (default format)
    #   {date:%d/%m/%Y}    -> date with custom strftime format
    #   {name}  -> metric name,  {value} -> cell value,
    #   {week}  -> week number,  {repo}  -> extra column
    hovertemplate="<b>{date:%d/%m/%Y}</b><br>{value} commits · repo: {repo}<extra></extra>",
    customdata=["repo"],
)

fig.show()
