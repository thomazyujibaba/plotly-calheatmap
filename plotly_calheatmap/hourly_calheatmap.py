import calendar
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .i18n import get_localized_month_names
from .calheatmap import _prepare_dataset_configs
from .layout_formatter import apply_figure_layout, apply_general_colorscaling, get_theme_defaults
from .utils import validate_date_column


def _build_hourly_matrix(
    month_data: pd.DataFrame,
    y_col: str,
    agg: str,
    year: int,
    month: int,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Pivot one month's data into a 24 × days_in_month matrix.

    Returns (z_matrix, hours, days) where hours=[0..23] and days=[1..N].
    """
    days_in_month = calendar.monthrange(year, month)[1]
    days = list(range(1, days_in_month + 1))
    hours = list(range(24))

    if month_data.empty:
        z = np.full((24, days_in_month), np.nan)
        return z, hours, days

    pivot = month_data.pivot_table(
        index="hour", columns="day", values=y_col, aggfunc=agg
    )
    pivot = pivot.reindex(index=hours, columns=days)
    return pivot.values, hours, days


def _build_year_figure(
    df: pd.DataFrame,
    x: str,
    y: str,
    year: int,
    agg: str,
    cols: int,
    month_names: List[str],
    colorscale: Union[str, list],
    gap: int,
    hover: str,
    font_color: str,
) -> Tuple[go.Figure, int]:
    """Build the subplot figure for a single year.

    Returns (fig, n_traces).
    """
    year_data = df[df["year"] == year]
    months_present = sorted(year_data["month"].unique())
    n_panels = len(months_present)
    n_cols = min(cols, n_panels)
    n_rows = -(-n_panels // n_cols)

    subplot_titles = [month_names[m - 1] for m in months_present]
    while len(subplot_titles) < n_rows * n_cols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    for idx, m in enumerate(months_present):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        month_data = year_data[year_data["month"] == m]
        z, hours, days = _build_hourly_matrix(month_data, y, agg, year, m)

        fig.add_trace(
            go.Heatmap(
                x=days,
                y=hours,
                z=z,
                colorscale=colorscale,
                xgap=gap,
                ygap=gap,
                showscale=False,
                hovertemplate=hover,
            ),
            row=row,
            col=col,
        )

        axis_num = (row - 1) * n_cols + col
        axis_suffix = "" if axis_num == 1 else str(axis_num)

        is_bottom = (row == n_rows) or (idx + n_cols >= n_panels)
        fig.layout[f"xaxis{axis_suffix}"].update(
            tickvals=[1, 10, 20, days[-1]],
            showticklabels=is_bottom,
        )
        fig.layout[f"yaxis{axis_suffix}"].update(
            autorange="reversed",
            tickvals=list(range(0, 24, 6)),
            ticktext=[f"{h}:00" for h in range(0, 24, 6)],
            showticklabels=(col == 1),
        )

    return fig, len(months_present)


def hourly_calheatmap(
    data: pd.DataFrame,
    x: str,
    y: str,
    agg: str = "mean",
    dark_theme: bool = False,
    colorscale: Union[str, list] = "viridis",
    gap: int = 1,
    cols: int = 4,
    navigation: bool = False,
    nav_options: Optional[Dict[str, Any]] = None,
    title: str = "",
    total_height: Optional[int] = None,
    total_width: Optional[int] = None,
    showscale: bool = False,
    scale_ticks: Optional[list] = None,
    cmap_min: Optional[float] = None,
    cmap_max: Optional[float] = None,
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
    locale: Optional[str] = None,
    paper_bgcolor: Optional[str] = None,
    plot_bgcolor: Optional[str] = None,
    font_color: Optional[str] = None,
    font_size: Optional[int] = None,
    title_font_color: Optional[str] = None,
    title_font_size: Optional[int] = None,
    margin: Optional[dict] = None,
    hovertemplate: Optional[str] = None,
    name: str = "",
    datasets: Optional[Dict[str, Dict[str, Any]]] = None,
    dataset_nav_options: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Create an hourly heatmap with subplots per month.

    Each subplot shows day of month (x-axis) vs hour of day (y-axis, reversed),
    colored by the aggregated value. Months are laid out in a grid with the
    given number of columns (default 4), grouped by year.

    Parameters
    ----------
    data : DataFrame
        Input data with a datetime column and a numeric value column.
    x : str
        Name of the datetime column.
    y : str
        Name of the value column.
    agg : str
        Aggregation function for sub-hourly data ("mean", "sum", "max", "min", "count").
    dark_theme : bool
        Use dark background theme.
    colorscale : str or list
        Plotly colorscale name or custom colorscale.
    gap : int
        Gap between heatmap cells in pixels.
    cols : int
        Number of columns in the subplot grid (default 4).
    navigation : bool
        Show year navigation buttons to toggle between years.
    nav_options : dict, optional
        Styling overrides for navigation buttons (Plotly ``updatemenus`` keys).
    title : str
        Figure title.
    total_height, total_width : int, optional
        Figure dimensions in pixels.
    showscale : bool
        Show the color bar.
    scale_ticks : list, optional
        Custom tick values for the color bar.
    cmap_min, cmap_max : float, optional
        Color range limits.
    date_fmt : str
        Date format for parsing string dates.
    locale : str, optional
        Locale for month names (e.g. "pt_BR", "es").
    paper_bgcolor, plot_bgcolor : str, optional
        Background colors.
    font_color : str, optional
        Font color.
    font_size : int, optional
        Font size.
    title_font_color : str, optional
        Title font color.
    title_font_size : int, optional
        Title font size.
    margin : dict, optional
        Figure margins (keys: l, r, t, b).
    hovertemplate : str, optional
        Custom hover template. Supports Plotly format with %{x}, %{y}, %{z}.
    name : str
        Name for the metric (used in default hover).
    datasets : dict, optional
        Multiple datasets to swap via dropdown. Same format as ``calheatmap()``.
    dataset_nav_options : dict, optional
        Styling overrides for the dataset dropdown menu.

    Returns
    -------
    go.Figure
    """
    # Theme defaults — hourly uses #fff/#333 for light theme (not None/#9e9e9e)
    paper_bgcolor, plot_bgcolor, font_color = get_theme_defaults(
        dark_theme,
        paper_bgcolor or ("#fff" if not dark_theme else None),
        plot_bgcolor,
        font_color or ("#333" if not dark_theme else None),
    )

    # Validate and prepare data
    df = data.copy()
    df[x] = validate_date_column(df[x], date_fmt)
    df["year"] = df[x].dt.year
    df["month"] = df[x].dt.month
    df["day"] = df[x].dt.day
    df["hour"] = df[x].dt.hour

    years = sorted(df["year"].unique())
    month_names = get_localized_month_names(locale)

    default_hover = (
        f"<b>{name}</b><br>" if name else ""
    ) + "Day %{x}, Hour %{y}:00<br>%{z}<extra></extra>"
    hover = hovertemplate or default_hover

    dataset_configs = _prepare_dataset_configs(
        datasets, y, colorscale, showscale, cmap_min, cmap_max, name
    )
    use_dataset_swap = len(dataset_configs) > 1

    if navigation and len(years) > 1:
        return _build_with_navigation(
            df, x, y, years, agg, cols, month_names, colorscale, gap, hover,
            font_color, nav_options, title, total_height, total_width,
            showscale, scale_ticks, cmap_min, cmap_max, name,
            paper_bgcolor, plot_bgcolor, font_size,
            title_font_color, title_font_size, margin,
            dataset_configs=dataset_configs,
            dataset_nav_options=dataset_nav_options,
        )

    if use_dataset_swap:
        # Dataset swap without year navigation: use navigation builder
        # with navigation menus but single-year-at-a-time approach
        return _build_with_navigation(
            df, x, y, years, agg, cols, month_names, colorscale, gap, hover,
            font_color, nav_options, title, total_height, total_width,
            showscale, scale_ticks, cmap_min, cmap_max, name,
            paper_bgcolor, plot_bgcolor, font_size,
            title_font_color, title_font_size, margin,
            dataset_configs=dataset_configs,
            dataset_nav_options=dataset_nav_options,
        )

    return _build_all_years(
        df, x, y, years, agg, cols, month_names, colorscale, gap, hover,
        font_color, title, total_height, total_width,
        showscale, scale_ticks, cmap_min, cmap_max, name,
        paper_bgcolor, plot_bgcolor, font_size,
        title_font_color, title_font_size, margin,
    )


def _build_all_years(
    df, x, y, years, agg, cols, month_names, colorscale, gap, hover,
    font_color, title, total_height, total_width,
    showscale, scale_ticks, cmap_min, cmap_max, name,
    paper_bgcolor, plot_bgcolor, font_size,
    title_font_color, title_font_size, margin,
):
    """Build figure showing all years stacked vertically."""
    # Build list of (year, month) pairs
    year_month_pairs = []
    for yr in years:
        for m in sorted(df[df["year"] == yr]["month"].unique()):
            year_month_pairs.append((yr, m))

    n_panels = len(year_month_pairs)
    n_cols = min(cols, n_panels)
    n_rows = -(-n_panels // n_cols)

    subplot_titles = [month_names[m - 1] for _, m in year_month_pairs]
    while len(subplot_titles) < n_rows * n_cols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    for idx, (yr, m) in enumerate(year_month_pairs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        month_data = df[(df["year"] == yr) & (df["month"] == m)]
        z, hours, days = _build_hourly_matrix(month_data, y, agg, yr, m)

        fig.add_trace(
            go.Heatmap(
                x=days,
                y=hours,
                z=z,
                colorscale=colorscale,
                xgap=gap,
                ygap=gap,
                showscale=False,
                hovertemplate=hover,
            ),
            row=row,
            col=col,
        )

        axis_num = (row - 1) * n_cols + col
        axis_suffix = "" if axis_num == 1 else str(axis_num)

        is_bottom = (row == n_rows) or (idx + n_cols >= n_panels)
        fig.layout[f"xaxis{axis_suffix}"].update(
            tickvals=[1, 10, 20, days[-1]],
            showticklabels=is_bottom,
        )
        fig.layout[f"yaxis{axis_suffix}"].update(
            autorange="reversed",
            tickvals=list(range(0, 24, 6)),
            ticktext=[f"{h}:00" for h in range(0, 24, 6)],
            showticklabels=(col == 1),
        )

    # Year annotations
    if len(years) > 1:
        for yr in years:
            yr_indices = [i for i, (y, _) in enumerate(year_month_pairs) if y == yr]
            first_row = yr_indices[0] // n_cols + 1
            last_row = yr_indices[-1] // n_cols + 1
            mid_row = (first_row + last_row) / 2
            y_pos = 1.0 - (mid_row - 0.5) / n_rows
            fig.add_annotation(
                text=f"<b>{yr}</b>",
                x=-0.06, y=y_pos,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14, color=font_color),
                textangle=-90,
            )

    if cmap_min is not None and cmap_max is not None:
        apply_general_colorscaling(fig, cmap_min, cmap_max)

    if showscale:
        _apply_colorbar(fig, name, scale_ticks)

    if total_height is None:
        total_height = max(400, n_rows * 200)

    apply_figure_layout(
        fig,
        total_height=total_height,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        font_color=font_color,
        font_size=font_size,
        title=title,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        width=total_width or max(800, n_cols * 250),
        margin=margin,
        default_margin=dict(l=80, r=80, t=60, b=60),
    )

    return fig


def _build_with_navigation(
    df, x, y, years, agg, cols, month_names, colorscale, gap, hover,
    font_color, nav_options, title, total_height, total_width,
    showscale, scale_ticks, cmap_min, cmap_max, name,
    paper_bgcolor, plot_bgcolor, font_size,
    title_font_color, title_font_size, margin,
    dataset_configs=None,
    dataset_nav_options=None,
):
    """Build figure with year navigation buttons (one year visible at a time).

    When *dataset_configs* contains more than one entry a dataset dropdown is
    also added.
    """
    if dataset_configs is None:
        dataset_configs = {name: {"y": y, "colorscale": colorscale,
                                   "showscale": showscale, "cmap_min": cmap_min,
                                   "cmap_max": cmap_max, "name": name}}

    use_dataset_swap = len(dataset_configs) > 1
    use_navigation = len(years) > 1

    n_cols = min(cols, 12)
    n_rows_per_year = -(-12 // n_cols)

    subplot_titles = [month_names[m - 1] for m in range(1, 13)]
    while len(subplot_titles) < n_rows_per_year * n_cols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows_per_year,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    trace_structure: List[Dict[str, Any]] = []
    trace_counts: list = []  # legacy, for single-dataset path

    for ds_label, ds_cfg in dataset_configs.items():
        ds_y = ds_cfg["y"]
        ds_colorscale = ds_cfg["colorscale"]
        ds_cmap_min = ds_cfg["cmap_min"]
        ds_cmap_max = ds_cfg["cmap_max"]

        if ds_cmap_min is None:
            ds_cmap_min = df[ds_y].min() if ds_y in df.columns else 0
        if ds_cmap_max is None:
            ds_cmap_max = df[ds_y].max() if ds_y in df.columns else 1

        ds_name = ds_cfg["name"]
        ds_hover = hover
        if ds_name and not hover.startswith(f"<b>{ds_name}"):
            ds_hover = (f"<b>{ds_name}</b><br>" if ds_name else "") + "Day %{x}, Hour %{y}:00<br>%{z}<extra></extra>"

        dataset_trace_counts: list = []

        for yr in years:
            traces_before = len(fig.data)
            year_data = df[df["year"] == yr]

            for m in range(1, 13):
                row = (m - 1) // n_cols + 1
                col = (m - 1) % n_cols + 1
                month_data = year_data[year_data["month"] == m]
                z, hours, days = _build_hourly_matrix(month_data, ds_y, agg, yr, m)

                fig.add_trace(
                    go.Heatmap(
                        x=days, y=hours, z=z,
                        colorscale=ds_colorscale,
                        xgap=gap, ygap=gap,
                        showscale=False,
                        hovertemplate=ds_hover,
                        zmin=ds_cmap_min,
                        zmax=ds_cmap_max,
                    ),
                    row=row, col=col,
                )

            tc = len(fig.data) - traces_before
            dataset_trace_counts.append(tc)
            trace_structure.append({
                "dataset": ds_label, "year": yr,
                "start": traces_before, "count": tc,
            })

        # Show colorbar for this dataset
        ds_showscale = ds_cfg["showscale"]
        if ds_showscale:
            scale_label = ds_showscale if isinstance(ds_showscale, str) else ""
            ds_start = trace_structure[-len(years)]["start"]
            colorbar = dict(
                orientation="v", yanchor="middle", y=0.5,
                xanchor="left", x=1.02, thickness=15, len=0.5,
                title=dict(text=scale_label, side="top"), nticks=5,
            )
            if scale_ticks is not None:
                colorbar["tickvals"] = scale_ticks
            # Enable on first trace of each year for this dataset
            for t in trace_structure:
                if t["dataset"] == ds_label:
                    fig.data[t["start"]].showscale = True
                    fig.data[t["start"]].colorbar = colorbar

        if not use_dataset_swap:
            trace_counts = dataset_trace_counts

    # Configure axes
    for m in range(1, 13):
        row = (m - 1) // n_cols + 1
        col = (m - 1) % n_cols + 1
        axis_num = (row - 1) * n_cols + col
        axis_suffix = "" if axis_num == 1 else str(axis_num)
        is_bottom = (row == n_rows_per_year)
        fig.layout[f"xaxis{axis_suffix}"].update(
            tickvals=[1, 10, 20, 31], showticklabels=is_bottom,
        )
        fig.layout[f"yaxis{axis_suffix}"].update(
            autorange="reversed",
            tickvals=list(range(0, 24, 6)),
            ticktext=[f"{h}:00" for h in range(0, 24, 6)],
            showticklabels=(col == 1),
        )

    # Build menus
    total_traces = len(fig.data)
    first_dataset = list(dataset_configs.keys())[0]
    first_year = years[0]

    if use_dataset_swap:
        # Set initial visibility
        for t in trace_structure:
            is_visible = t["dataset"] == first_dataset and (
                t["year"] == first_year if use_navigation else True
            )
            for i in range(t["start"], t["start"] + t["count"]):
                fig.data[i].visible = is_visible

        menus = []

        # Dataset dropdown
        ds_buttons = []
        for ds_label in dataset_configs:
            vis = [False] * total_traces
            for t in trace_structure:
                if t["dataset"] != ds_label:
                    continue
                if use_navigation and t["year"] != first_year:
                    continue
                for i in range(t["start"], t["start"] + t["count"]):
                    vis[i] = True
            ds_buttons.append(dict(method="update", args=[{"visible": vis}], label=ds_label))

        ds_menu = dict(
            type="dropdown", direction="down", active=0,
            buttons=ds_buttons, x=0, xanchor="left",
            y=1.4, yanchor="top", showactive=True,
        )
        if dataset_nav_options:
            ds_menu.update(dataset_nav_options)
        menus.append(ds_menu)

        # Year navigation
        if use_navigation:
            yr_buttons = []
            for yr in years:
                vis = [False] * total_traces
                for t in trace_structure:
                    if t["dataset"] == first_dataset and t["year"] == yr:
                        for i in range(t["start"], t["start"] + t["count"]):
                            vis[i] = True
                yr_buttons.append(dict(method="restyle", args=[{"visible": vis}], label=str(yr)))

            yr_menu = dict(
                type="buttons", direction="right", active=0,
                buttons=yr_buttons, x=1, xanchor="right",
                y=1.4, yanchor="top", showactive=True,
            )
            if nav_options:
                yr_menu.update(nav_options)
            menus.append(yr_menu)

        fig.update_layout(updatemenus=menus)
    else:
        # Single dataset with year navigation (original behavior)
        offsets = []
        acc = 0
        for count in trace_counts:
            offsets.append(acc)
            acc += count

        for i in range(trace_counts[0], total_traces):
            fig.data[i].visible = False

        buttons = []
        for idx, year in enumerate(years):
            visibility = [False] * total_traces
            start = offsets[idx]
            end = start + trace_counts[idx]
            for j in range(start, end):
                visibility[j] = True
            buttons.append(dict(method="restyle", args=[{"visible": visibility}], label=str(year)))

        menu_config = dict(
            type="buttons", direction="right", active=0,
            buttons=buttons, x=1, xanchor="right",
            y=1.4, yanchor="top", showactive=True,
        )
        if nav_options:
            menu_config.update(nav_options)
        fig.update_layout(updatemenus=[menu_config])

        if cmap_min is not None and cmap_max is not None:
            apply_general_colorscaling(fig, cmap_min, cmap_max)

        if showscale:
            _apply_colorbar(fig, name, scale_ticks, trace_offsets=offsets)

    if total_height is None:
        total_height = max(400, n_rows_per_year * 200)

    apply_figure_layout(
        fig,
        total_height=total_height,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        font_color=font_color,
        font_size=font_size,
        title=title,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        width=total_width or max(800, n_cols * 250),
        margin=margin,
        default_margin=dict(l=80, r=40, t=80, b=60),
    )

    return fig


def _apply_colorbar(
    fig: go.Figure,
    scale_title: str = "",
    scale_ticks: Optional[list] = None,
    trace_offsets: Optional[List[int]] = None,
) -> None:
    """Show colorbar on the first heatmap trace of each year group.

    When trace_offsets is provided (navigation mode), enables the colorbar
    on the first trace of each year so it stays visible when switching years.
    Otherwise, enables it on only the first heatmap trace.
    """
    colorbar = dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.02,
        thickness=15,
        len=0.5,
        title=dict(text=scale_title, side="top"),
        nticks=5,
    )
    if scale_ticks is not None:
        colorbar["tickvals"] = scale_ticks

    if trace_offsets is not None:
        # Navigation mode: enable on first trace of each year group
        for offset in trace_offsets:
            fig.data[offset].showscale = True
            fig.data[offset].colorbar = colorbar
    else:
        # Single view: enable on first heatmap trace only
        for trace in fig.data:
            if isinstance(trace, go.Heatmap):
                trace.showscale = True
                trace.colorbar = colorbar
                break
