from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from plotly import graph_objects as go

from plotly_calheatmap.i18n import get_localized_day_abbrs


def get_theme_defaults(
    dark_theme: bool,
    paper_bgcolor: Optional[str] = None,
    plot_bgcolor: Optional[str] = None,
    font_color: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """Return resolved (paper_bgcolor, plot_bgcolor, font_color) for the theme."""
    if dark_theme:
        return (paper_bgcolor or "#333", plot_bgcolor or "#333", font_color or "#fff")
    return (paper_bgcolor, plot_bgcolor or "#fff", font_color or "#9e9e9e")


def apply_figure_layout(
    fig: go.Figure,
    *,
    total_height: int,
    paper_bgcolor: Optional[str],
    plot_bgcolor: str,
    font_color: str,
    font_size: Optional[int] = None,
    title: str = "",
    title_font_color: Optional[str] = None,
    title_font_size: Optional[int] = None,
    width: Optional[int] = None,
    margin: Optional[dict] = None,
    default_margin: Optional[dict] = None,
) -> None:
    """Apply final layout settings to a figure (sizing, theme, title)."""
    layout: Dict[str, Any] = dict(
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        font=dict(color=font_color, size=font_size or 10),
        height=total_height,
        autosize=True,
        showlegend=False,
        margin=margin or default_margin or {"t": 20, "b": 20},
    )
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(
                color=title_font_color or font_color,
                size=title_font_size or 16,
            ),
        )
    if width is not None:
        layout["width"] = width
    fig.update_layout(**layout)


def decide_layout(
    dark_theme: bool,
    title: str,
    month_names: List[str],
    month_positions: Any,
    locale: Optional[str] = None,
    paper_bgcolor: Optional[str] = None,
    plot_bgcolor: Optional[str] = None,
    font_color: Optional[str] = None,
    font_size: Optional[int] = None,
    title_font_color: Optional[str] = None,
    title_font_size: Optional[int] = None,
    margin: Optional[dict] = None,
    month_labels_side: str = "bottom",
    vertical: bool = False,
    week_start: str = "monday",
) -> go.Layout:
    # Filter out None pairs (from month_gap padding or missing months)
    filtered = [
        (n, p) for n, p in zip(month_names, month_positions)
        if n is not None and p is not None
    ]
    if filtered:
        month_names, month_positions = zip(*filtered)
        month_names = list(month_names)
        month_positions = list(month_positions)

    day_names = get_localized_day_abbrs(locale, week_start=week_start)

    _paper_bgcolor, _plot_bgcolor, _font_color = get_theme_defaults(
        dark_theme, paper_bgcolor, plot_bgcolor, font_color,
    )
    _font_size = font_size or 10
    _margin = margin or {"t": 20, "b": 20}

    title_cfg: Any = title
    if title and (title_font_color or title_font_size):
        title_font = {}
        if title_font_color:
            title_font["color"] = title_font_color
        if title_font_size:
            title_font["size"] = title_font_size
        title_cfg = dict(text=title, font=title_font)

    if vertical:
        # Vertical: days across x-axis, months/weeks down y-axis
        xaxis_cfg = dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            tickmode="array",
            ticktext=day_names,
            tickvals=[0, 1, 2, 3, 4, 5, 6],
        )
        if month_labels_side == "top":
            xaxis_cfg["side"] = "top"

        yaxis_cfg = dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            tickmode="array",
            ticktext=month_names,
            tickvals=month_positions,
            autorange="reversed",
        )
    else:
        # Horizontal (default): weeks across x-axis, days down y-axis
        xaxis_cfg = dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            tickmode="array",
            ticktext=month_names,
            tickvals=month_positions,
        )
        if month_labels_side == "top":
            xaxis_cfg["side"] = "top"

        yaxis_cfg = dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            tickmode="array",
            ticktext=day_names,
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            autorange="reversed",
        )

    layout = go.Layout(
        title=title_cfg,
        yaxis=yaxis_cfg,
        xaxis=xaxis_cfg,
        font={"size": _font_size, "color": _font_color},
        paper_bgcolor=_paper_bgcolor,
        plot_bgcolor=_plot_bgcolor,
        margin=_margin,
        showlegend=False,
        autosize=True,
    )

    return layout


def create_month_lines(
    cplt: List[go.Figure],
    month_lines_color: str,
    month_lines_width: int,
    data: pd.DataFrame,
    weekdays_in_year: List[float],
    weeknumber_of_dates: List[int],
    vertical: bool = False,
    month_gap: int = 0,
) -> go.Figure:
    kwargs = dict(
        mode="lines",
        line=dict(color=month_lines_color, width=month_lines_width),
        hoverinfo="skip",
        showlegend=False,
    )
    for date, dow, wkn in zip(data, weekdays_in_year, weeknumber_of_dates):
        if date.day == 1:
            if month_gap > 0:
                # With month_gap, weeks are fully separated so draw a simple
                # full-width line at the top edge of the first week of the month.
                if vertical:
                    cplt += [go.Scatter(y=[wkn - 0.5, wkn - 0.5], x=[-0.5, 6.5], **kwargs)]
                else:
                    cplt += [go.Scatter(x=[wkn - 0.5, wkn - 0.5], y=[-0.5, 6.5], **kwargs)]
            elif vertical:
                cplt += [go.Scatter(y=[wkn - 0.5, wkn - 0.5], x=[dow - 0.5, 6.5], **kwargs)]
                if dow:
                    cplt += [
                        go.Scatter(
                            y=[wkn - 0.5, wkn + 0.5], x=[dow - 0.5, dow - 0.5], **kwargs
                        ),
                        go.Scatter(y=[wkn + 0.5, wkn + 0.5], x=[dow - 0.5, -0.5], **kwargs),
                    ]
            else:
                cplt += [go.Scatter(x=[wkn - 0.5, wkn - 0.5], y=[dow - 0.5, 6.5], **kwargs)]
                if dow:
                    cplt += [
                        go.Scatter(
                            x=[wkn - 0.5, wkn + 0.5], y=[dow - 0.5, dow - 0.5], **kwargs
                        ),
                        go.Scatter(x=[wkn + 0.5, wkn + 0.5], y=[dow - 0.5, -0.5], **kwargs),
                    ]

    # Close the end of the last month
    date = data.iloc[-1]
    dow = weekdays_in_year[-1]
    wkn = weeknumber_of_dates[-1]
    if vertical:
        cplt += [go.Scatter(y=[wkn + 0.5, wkn + 0.5], x=[-0.5, dow + 0.5], **kwargs)]
        if dow != 6:
            cplt += [
                go.Scatter(y=[wkn - 0.5, wkn + 0.5], x=[dow + 0.5, dow + 0.5], **kwargs),
                go.Scatter(y=[wkn - 0.5, wkn - 0.5], x=[dow + 0.5, 6.5], **kwargs),
            ]
    else:
        cplt += [go.Scatter(x=[wkn + 0.5, wkn + 0.5], y=[-0.5, dow + 0.5], **kwargs)]
        if dow != 6:
            cplt += [
                go.Scatter(x=[wkn - 0.5, wkn + 0.5], y=[dow + 0.5, dow + 0.5], **kwargs),
                go.Scatter(x=[wkn - 0.5, wkn - 0.5], y=[dow + 0.5, 6.5], **kwargs),
            ]

    return cplt


def create_top_bottom_lines(
    cplt: List[go.Figure],
    month_lines_color: str,
    month_lines_width: int,
    weeknumber_of_dates: List[int],
    vertical: bool = False,
) -> List[go.Figure]:
    """Draw horizontal lines at the top and bottom edges of the calendar."""
    kwargs = dict(
        mode="lines",
        line=dict(color=month_lines_color, width=month_lines_width),
        hoverinfo="skip",
        showlegend=False,
    )
    wk_min = min(weeknumber_of_dates)
    wk_max = max(weeknumber_of_dates)
    if vertical:
        # top line (left edge of days)
        cplt += [go.Scatter(y=[wk_min + 0.5, wk_max + 0.5], x=[-0.5, -0.5], **kwargs)]
        # bottom line (right edge of days)
        cplt += [go.Scatter(y=[wk_min - 0.5, wk_max - 0.5], x=[6.5, 6.5], **kwargs)]
    else:
        # top line
        cplt += [go.Scatter(x=[wk_min + 0.5, wk_max + 0.5], y=[-0.5, -0.5], **kwargs)]
        # bottom line
        cplt += [go.Scatter(x=[wk_min - 0.5, wk_max - 0.5], y=[6.5, 6.5], **kwargs)]
    return cplt


def create_grouping_lines(
    cplt: List[go.Figure],
    grouping_lines_color: str,
    grouping_lines_width: int,
    data: pd.DataFrame,
    weekdays_in_year: List[float],
    weeknumber_of_dates: List[int],
    boundary_months: List[int],
    vertical: bool = False,
    month_gap: int = 0,
) -> List[go.Figure]:
    """Draw thicker separator lines at group boundaries (quarters, semesters, etc.).

    Uses the same drawing logic as create_month_lines but only at months
    listed in *boundary_months*.
    """
    kwargs = dict(
        mode="lines",
        line=dict(color=grouping_lines_color, width=grouping_lines_width),
        hoverinfo="skip",
        showlegend=False,
    )
    for date, dow, wkn in zip(data, weekdays_in_year, weeknumber_of_dates):
        if date.day == 1 and date.month in boundary_months:
            if month_gap > 0:
                if vertical:
                    cplt += [go.Scatter(y=[wkn - 0.5, wkn - 0.5], x=[-0.5, 6.5], **kwargs)]
                else:
                    cplt += [go.Scatter(x=[wkn - 0.5, wkn - 0.5], y=[-0.5, 6.5], **kwargs)]
            elif vertical:
                cplt += [go.Scatter(y=[wkn - 0.5, wkn - 0.5], x=[dow - 0.5, 6.5], **kwargs)]
                if dow:
                    cplt += [
                        go.Scatter(
                            y=[wkn - 0.5, wkn + 0.5], x=[dow - 0.5, dow - 0.5], **kwargs
                        ),
                        go.Scatter(y=[wkn + 0.5, wkn + 0.5], x=[dow - 0.5, -0.5], **kwargs),
                    ]
            else:
                cplt += [go.Scatter(x=[wkn - 0.5, wkn - 0.5], y=[dow - 0.5, 6.5], **kwargs)]
                if dow:
                    cplt += [
                        go.Scatter(
                            x=[wkn - 0.5, wkn + 0.5], y=[dow - 0.5, dow - 0.5], **kwargs
                        ),
                        go.Scatter(x=[wkn + 0.5, wkn + 0.5], y=[dow - 0.5, -0.5], **kwargs),
                    ]

    # Close the end of the last group
    date = data.iloc[-1]
    dow = weekdays_in_year[-1]
    wkn = weeknumber_of_dates[-1]
    if vertical:
        cplt += [go.Scatter(y=[wkn + 0.5, wkn + 0.5], x=[-0.5, dow + 0.5], **kwargs)]
        if dow != 6:
            cplt += [
                go.Scatter(y=[wkn - 0.5, wkn + 0.5], x=[dow + 0.5, dow + 0.5], **kwargs),
                go.Scatter(y=[wkn - 0.5, wkn - 0.5], x=[dow + 0.5, 6.5], **kwargs),
            ]
    else:
        cplt += [go.Scatter(x=[wkn + 0.5, wkn + 0.5], y=[-0.5, dow + 0.5], **kwargs)]
        if dow != 6:
            cplt += [
                go.Scatter(x=[wkn - 0.5, wkn + 0.5], y=[dow + 0.5, dow + 0.5], **kwargs),
                go.Scatter(x=[wkn - 0.5, wkn - 0.5], y=[dow + 0.5, 6.5], **kwargs),
            ]

    return cplt


def update_plot_with_current_layout(
    fig: go.Figure,
    cplt: go.Figure,
    row: int,
    layout: go.Layout,
    total_height: Optional[int],
    years_as_columns: bool,
    width: Optional[int] = None,
) -> go.Figure:
    fig.update_layout(layout)
    fig.update_layout(height=total_height, autosize=True)
    if width is not None:
        fig.update_layout(width=width)
    if years_as_columns:
        r, c = 1, row + 1
    else:
        r, c = row + 1, 1
    fig.update_xaxes(layout["xaxis"], row=r, col=c)
    fig.update_yaxes(layout["yaxis"], row=r, col=c)
    fig.add_traces(cplt, rows=[r] * len(cplt), cols=[c] * len(cplt))
    return fig


def apply_general_colorscaling(
    fig: go.Figure, cmap_min: float, cmap_max: float
) -> go.Figure:
    return fig.update_traces(
        selector=dict(type="heatmap"), zmax=cmap_max, zmin=cmap_min
    )


def showscale_of_heatmaps(
    fig: go.Figure,
    scale_title: str = "",
    scale_ticks: list | None = None,
) -> go.Figure:
    colorbar = dict(
        orientation="h",
        yanchor="top",
        y=-0.05,
        xanchor="center",
        x=0.5,
        thickness=10,
        len=0.5,
        title=dict(text=scale_title, side="top"),
    )
    if scale_ticks is not None:
        colorbar["tickvals"] = scale_ticks

    # Add bottom margin to make room for the colorbar
    current_margin = fig.layout.margin
    bottom = current_margin.b if current_margin and current_margin.b is not None else 80
    fig.update_layout(margin=dict(b=max(bottom, 80)))

    return fig.update_traces(
        showscale=True,
        colorbar=colorbar,
        selector=dict(type="heatmap"),
    )
