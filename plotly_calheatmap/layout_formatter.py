from typing import Any, List, Optional

import pandas as pd
from plotly import graph_objects as go

from plotly_calheatmap.i18n import get_localized_day_abbrs


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
) -> go.Layout:
    day_names = get_localized_day_abbrs(locale)

    # Theme defaults
    if dark_theme:
        _paper_bgcolor = paper_bgcolor or "#333"
        _plot_bgcolor = plot_bgcolor or "#333"
        _font_color = font_color or "#fff"
    else:
        _paper_bgcolor = paper_bgcolor
        _plot_bgcolor = plot_bgcolor or "#fff"
        _font_color = font_color or "#9e9e9e"

    _font_size = font_size or 10
    _margin = margin or {"t": 20, "b": 20}

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

    title_cfg: Any = title
    if title and (title_font_color or title_font_size):
        title_font = {}
        if title_font_color:
            title_font["color"] = title_font_color
        if title_font_size:
            title_font["size"] = title_font_size
        title_cfg = dict(text=title, font=title_font)

    layout = go.Layout(
        title=title_cfg,
        yaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            tickmode="array",
            ticktext=day_names,
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            autorange="reversed",
        ),
        xaxis=xaxis_cfg,
        font={"size": _font_size, "color": _font_color},
        paper_bgcolor=_paper_bgcolor,
        plot_bgcolor=_plot_bgcolor,
        margin=_margin,
        showlegend=False,
    )

    return layout


def create_month_lines(
    cplt: List[go.Figure],
    month_lines_color: str,
    month_lines_width: int,
    data: pd.DataFrame,
    weekdays_in_year: List[float],
    weeknumber_of_dates: List[int],
) -> go.Figure:
    kwargs = dict(
        mode="lines",
        line=dict(color=month_lines_color, width=month_lines_width),
        hoverinfo="skip",
    )
    for date, dow, wkn in zip(data, weekdays_in_year, weeknumber_of_dates):
        if date.day == 1:
            cplt += [go.Scatter(x=[wkn - 0.5, wkn - 0.5], y=[dow - 0.5, 6.5], **kwargs)]
            if dow:
                cplt += [
                    go.Scatter(
                        x=[wkn - 0.5, wkn + 0.5], y=[dow - 0.5, dow - 0.5], **kwargs
                    ),
                    go.Scatter(x=[wkn + 0.5, wkn + 0.5], y=[dow - 0.5, -0.5], **kwargs),
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
    fig.update_layout(height=total_height)
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
