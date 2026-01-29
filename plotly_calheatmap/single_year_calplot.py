from typing import List, Optional, Union

from pandas.core.frame import DataFrame
from plotly import graph_objects as go

from plotly_calheatmap.date_extractors import get_date_coordinates, get_month_names
from plotly_calheatmap.layout_formatter import (
    create_month_lines,
    decide_layout,
    update_plot_with_current_layout,
)
from plotly_calheatmap.raw_heatmap import create_heatmap_without_formatting


def year_calplot(
    data: DataFrame,
    x: str,
    y: str,
    fig: go.Figure,
    row: int,
    year: int,
    name: str = "y",
    dark_theme: bool = False,
    month_lines_width: int = 1,
    month_lines_color: str = "#9e9e9e",
    gap: int = 1,
    colorscale: Union[str, list] = "greens",
    title: str = "",
    month_lines: bool = True,
    total_height: Union[int, None] = None,
    text: Optional[List[str]] = None,
    text_name: Optional[str] = None,
    years_as_columns: bool = False,
    start_month: int = 1,
    end_month: int = 12,
    locale: Optional[str] = None,
    paper_bgcolor: Optional[str] = None,
    plot_bgcolor: Optional[str] = None,
    font_color: Optional[str] = None,
    font_size: Optional[int] = None,
    title_font_color: Optional[str] = None,
    title_font_size: Optional[int] = None,
    width: Optional[int] = None,
    margin: Optional[dict] = None,
    month_labels_side: str = "bottom",
    hovertemplate: Optional[str] = None,
    extra_customdata_columns: Optional[List[str]] = None,
) -> go.Figure:
    """
    Each year is subplotted separately and added to the main plot
    """

    month_names = get_month_names(data, x, start_month, end_month, locale=locale)
    month_positions, weekdays_in_year, weeknumber_of_dates = get_date_coordinates(
        data, x
    )

    # the calendar is actually a heatmap :)
    cplt = create_heatmap_without_formatting(
        data,
        x,
        y,
        weeknumber_of_dates,
        weekdays_in_year,
        gap,
        year,
        colorscale,
        name,
        text=text,
        text_name=text_name,
        hovertemplate=hovertemplate,
        extra_customdata_columns=extra_customdata_columns,
    )

    if month_lines:
        cplt = create_month_lines(
            cplt,
            month_lines_color,
            month_lines_width,
            data[x],
            weekdays_in_year,
            weeknumber_of_dates,
        )

    layout = decide_layout(
        dark_theme,
        title,
        month_names,
        month_positions,
        locale=locale,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        font_color=font_color,
        font_size=font_size,
        title_font_color=title_font_color,
        title_font_size=title_font_size,
        margin=margin,
        month_labels_side=month_labels_side,
    )
    fig = update_plot_with_current_layout(
        fig, cplt, row, layout, total_height, years_as_columns, width=width
    )

    return fig
