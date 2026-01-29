# Roadmap

## v0.4

- [ ] **Custom week start day** — Let users choose Sunday or Monday as the first day of the week (`week_start="sunday"`).
- [ ] **ISO week numbering** — Option to use ISO 8601 weeks instead of the current `strftime("%W")` Gregorian weeks.
- [ ] **Categorical / discrete colorscales** — Support binned color levels (e.g., 0 = gray, 1-3 = light green, 4+ = dark green) in addition to continuous scales.
- [ ] **Cell annotations** — Display text (values, labels, emojis) inside each cell via an `annotations=True` or `annotations_fmt=".0f"` parameter.
- [ ] **Highlight specific dates** — An API to mark/outline holidays, deadlines, or special dates with borders or markers.
- [ ] **Click / selection callbacks** — Expose click and lasso-select events so users can react to date selections in Dash or Jupyter.
- [ ] **Export helpers** — Convenience methods to export the calendar as static PNG/SVG/PDF with sensible default dimensions.
- [ ] **Hourly heatmap enhancements** — Add day-of-week labels, custom time bins, and gap highlighting to `hourly_calheatmap`.
- [ ] **Legend / colorbar customization** — More control over colorbar position, orientation, tick format, and title.
