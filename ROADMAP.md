# Roadmap

## v0.2

- [ ] **Vertical orientation** — Render months as rows instead of columns for a different layout style.
- [ ] **Built-in aggregation functions** — Allow `agg="sum"|"mean"|"count"|"max"` so users can pass raw event data without pre-aggregating.
- [ ] **Hourly heatmap**

## v0.3

- [ ] **Custom week start day** — Let users choose Sunday or Monday as the first day of the week (`week_start="sunday"`).
- [ ] **ISO week numbering** — Option to use ISO 8601 weeks instead of the current `strftime("%W")` Gregorian weeks.
- [ ] **Categorical / discrete colorscales** — Support binned color levels (e.g., 0 = gray, 1-3 = light green, 4+ = dark green) in addition to continuous scales.
- [ ] **Cell annotations** — Display text (values, labels, emojis) inside each cell via an `annotations=True` or `annotations_fmt=".0f"` parameter.
- [ ] **Highlight specific dates** — An API to mark/outline holidays, deadlines, or special dates with borders or markers.
- [ ] **Responsive / auto-sizing** — Better auto-width/height that adapts to container size in notebooks and web apps.
- [ ] **Custom time groupings** — Option to group by semesters, quarters, bimesters, and other periods instead of only months.