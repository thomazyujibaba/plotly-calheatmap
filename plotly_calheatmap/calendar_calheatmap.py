import calendar
from math import log1p
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .i18n import get_localized_day_abbrs, get_localized_month_names
from .layout_formatter import (
    apply_figure_layout,
    apply_general_colorscaling,
    get_theme_defaults,
)
from .utils import validate_date_column


_WEEK_START_OFFSETS = {"monday": 0, "sunday": 1, "saturday": 2}


def _build_month_matrix(
    month_data: pd.DataFrame,
    x: str,
    y: str,
    year: int,
    month: int,
    offset: int,
    log_scale: bool = False,
):
    """Build a (weeks × 7) matrix for one month.

    Returns (z, week_rows, day_cols, dates, values) where
    day_cols are 0-6 and week_rows are 0-based week-of-month indices.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    first_day = pd.Timestamp(year, month, 1)
    first_day_col = (first_day.weekday() + offset) % 7
    n_weeks = (days_in_month - 1 + first_day_col) // 7 + 1

    z = np.full((n_weeks, 7), np.nan)
    dates_grid = np.full((n_weeks, 7), None, dtype=object)

    # Place values from data
    val_map = {}
    if not month_data.empty:
        for _, row in month_data.iterrows():
            dt = row[x]
            val_map[dt.day] = row[y]

    for day in range(1, days_in_month + 1):
        dt = pd.Timestamp(year, month, day)
        col = (dt.weekday() + offset) % 7
        week_row = (day - 1 + first_day_col) // 7
        dates_grid[week_row, col] = dt.strftime("%Y-%m-%d")
        if day in val_map:
            v = val_map[day]
            if log_scale and pd.notna(v):
                z[week_row, col] = log1p(v)
            else:
                z[week_row, col] = v

    day_cols = list(range(7))
    week_rows = list(range(n_weeks))
    return z, week_rows, day_cols, dates_grid


def _calendar_calheatmap_impl(
    data: pd.DataFrame,
    x: str,
    y: str,
    name: str = "y",
    dark_theme: bool = False,
    cols: int = 4,
    gap: int = 2,
    colorscale: Union[str, list] = "greens",
    title: str = "",
    showscale: Union[bool, str] = False,
    total_height: Optional[int] = None,
    width: Optional[int] = None,
    margin: Optional[dict] = None,
    cmap_min: Optional[float] = None,
    cmap_max: Optional[float] = None,
    log_scale: bool = False,
    date_fmt: str = "%Y-%m-%d",
    agg: Optional[Literal["sum", "mean", "count", "max"]] = None,
    locale: Optional[str] = None,
    paper_bgcolor: Optional[str] = None,
    plot_bgcolor: Optional[str] = None,
    font_color: Optional[str] = None,
    font_size: Optional[int] = None,
    title_font_color: Optional[str] = None,
    title_font_size: Optional[int] = None,
    hovertemplate: Optional[str] = None,
    start_month: int = 1,
    end_month: int = 12,
    week_start: Literal["monday", "sunday", "saturday"] = "monday",
) -> go.Figure:
    """Wall-calendar heatmap: a grid of mini-calendars, one per month.

    Each month is rendered with days-of-week as columns and weeks as rows,
    like a standard wall calendar.

    Parameters
    ----------
    data : DataFrame
        Must contain at least one date column and one numeric value column.
    x : str
        Name of the date column.
    y : str
        Name of the value column.
    cols : int
        Number of columns in the month grid (default 4 → 3×4 layout).
    week_start : str
        First day of the week: ``"monday"``, ``"sunday"``, or ``"saturday"``.
    """
    # --- Validate & prepare data ---
    data = data.copy()
    data[x] = validate_date_column(data[x], date_fmt)

    if agg is not None:
        data = data.groupby(x, as_index=False).agg({y: agg})

    offset = _WEEK_START_OFFSETS[week_start]
    day_names = get_localized_day_abbrs(locale, week_start=week_start)
    month_names_full = get_localized_month_names(locale)

    _paper_bgcolor, _plot_bgcolor, _font_color = get_theme_defaults(
        dark_theme, paper_bgcolor, plot_bgcolor, font_color,
    )

    # Determine years and months
    years = sorted(data[x].dt.year.unique())
    months = list(range(start_month, end_month + 1))

    # Build (year, month) pairs
    year_month_pairs = []
    for yr in years:
        for m in months:
            year_month_pairs.append((yr, m))

    n_panels = len(year_month_pairs)
    n_cols = min(cols, n_panels)
    n_rows = -(-n_panels // n_cols)

    subplot_titles = []
    for yr, m in year_month_pairs:
        label = month_names_full[m - 1]
        if len(years) > 1:
            label = f"{label} {yr}"
        subplot_titles.append(label)
    while len(subplot_titles) < n_rows * n_cols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )

    # Default hover
    if hovertemplate is None:
        hovertemplate = "%{customdata[0]}<br>%{z}<extra></extra>"

    # Find the max number of week rows across all months (for height calculation)
    max_week_rows = 0
    for yr, m in year_month_pairs:
        days_in_month = calendar.monthrange(yr, m)[1]
        first_day_col = (pd.Timestamp(yr, m, 1).weekday() + offset) % 7
        n_weeks = (days_in_month - 1 + first_day_col) // 7 + 1
        max_week_rows = max(max_week_rows, n_weeks)

    for idx, (yr, m) in enumerate(year_month_pairs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        month_data = data[(data[x].dt.year == yr) & (data[x].dt.month == m)]

        z, week_rows, day_cols, dates_grid = _build_month_matrix(
            month_data, x, y, yr, m, offset, log_scale=log_scale,
        )

        fig.add_trace(
            go.Heatmap(
                x=day_cols,
                y=week_rows,
                z=z.tolist(),
                customdata=[[[cell] for cell in row] for row in dates_grid.tolist()],
                colorscale=colorscale,
                xgap=gap,
                ygap=gap,
                showscale=False,
                hovertemplate=hovertemplate,
                hoverongaps=False,
            ),
            row=row,
            col=col,
        )

        # Configure axes per subplot
        axis_num = (row - 1) * n_cols + col
        axis_suffix = "" if axis_num == 1 else str(axis_num)

        is_bottom = (row == n_rows) or (idx + n_cols >= n_panels)
        fig.layout[f"xaxis{axis_suffix}"].update(
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=day_names,
            showticklabels=is_bottom,
            showgrid=False,
            zeroline=False,
            showline=False,
        )
        fig.layout[f"yaxis{axis_suffix}"].update(
            autorange="reversed",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
        )

    # Color range
    if cmap_min is not None and cmap_max is not None:
        apply_general_colorscaling(fig, cmap_min, cmap_max)

    # Colorbar
    if showscale:
        legend_title = showscale if isinstance(showscale, str) else None
        fig.data[0].showscale = True
        colorbar_cfg: Dict[str, Any] = dict(
            orientation="h",
            y=-0.1,
            x=0.5,
            xanchor="center",
            thickness=10,
            len=0.6,
        )
        if legend_title:
            colorbar_cfg["title"] = dict(text=legend_title, side="top")
        fig.data[0].colorbar = colorbar_cfg

    # Layout — compute dimensions so cells are approximately square
    effective_width = width or max(800, n_cols * 220)
    if total_height is None:
        # Cell width ≈ (figure_width - margins - gaps) / (n_cols * 7)
        # Target height so cell_height ≈ cell_width
        cell_w = (effective_width - 80) / (n_cols * 7)
        plot_h = cell_w * max_week_rows * n_rows
        # Add space for margins, titles, tick labels
        total_height = int(plot_h + 60 * n_rows + 80)

    apply_figure_layout(
        fig,
        total_height=total_height,
        paper_bgcolor=_paper_bgcolor,
        plot_bgcolor=_plot_bgcolor,
        font_color=_font_color,
        font_size=font_size,
        title=title,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        width=effective_width,
        margin=margin,
        default_margin=dict(l=40, r=40, t=60, b=60),
    )

    return fig
