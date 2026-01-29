from typing import Any, Dict, Literal, Optional, Union, List

import numpy as np
from pandas import DataFrame, Grouper, Series
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from plotly_calheatmap.layout_formatter import (
    apply_general_colorscaling,
    showscale_of_heatmaps,
)
from plotly_calheatmap.single_year_calheatmap import year_calheatmap
from plotly_calheatmap.i18n import get_localized_month_names
from plotly_calheatmap.utils import fill_empty_with_zeros, validate_date_column


def _get_subplot_layout(**kwargs: Any) -> go.Layout:
    """
    Combines the default subplot layout with the customized parameters
    """
    dark_theme: bool = kwargs.pop("dark_theme", False)
    yaxis: Dict[str, Any] = kwargs.pop("yaxis", {})
    xaxis: Dict[str, Any] = kwargs.pop("xaxis", {})

    def _dt(b: Any, a: Any) -> Any:
        return a if dark_theme else b

    # Build font with defaults, allow override
    font = {"size": 10, "color": _dt("#9e9e9e", "#fff")}
    if "font" in kwargs:
        font.update(kwargs.pop("font"))

    return go.Layout(
        **{
            "yaxis": {
                "showline": False,
                "showgrid": False,
                "zeroline": False,
                "tickmode": "array",
                "autorange": "reversed",
                **yaxis,
            },
            "xaxis": {
                "showline": False,
                "showgrid": False,
                "zeroline": False,
                "tickmode": "array",
                **xaxis,
            },
            "font": font,
            "plot_bgcolor": _dt("#fff", "#333"),
            "paper_bgcolor": _dt(None, "#333"),
            "margin": {"t": 20, "b": 20},
            "showlegend": False,
            **kwargs,
        }
    )


def _add_year_navigation(
    fig: go.Figure,
    unique_years: Any,
    trace_counts: list,
    nav_options: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Add year buttons on the right side of the chart.

    Parameters
    ----------
    nav_options : dict, optional
        Styling overrides for the button group. Supports any key accepted by
        Plotly's ``updatemenus`` (e.g. ``font``, ``bgcolor``, ``bordercolor``,
        ``borderwidth``, ``x``, ``y``, ``xanchor``, ``yanchor``, ``direction``,
        ``pad``).
    """
    total_traces = len(fig.data)

    # Build cumulative trace offsets
    offsets = []
    acc = 0
    for count in trace_counts:
        offsets.append(acc)
        acc += count

    # Hide all traces except the first year
    for i in range(trace_counts[0], total_traces):
        fig.data[i].visible = False

    # Build visibility arrays per year
    visibility_per_year = []
    for idx in range(len(unique_years)):
        visibility = [False] * total_traces
        start = offsets[idx]
        end = start + trace_counts[idx]
        for j in range(start, end):
            visibility[j] = True
        visibility_per_year.append(visibility)

    buttons = []
    for idx, year in enumerate(unique_years):
        buttons.append(
            dict(
                method="restyle",
                args=[{"visible": visibility_per_year[idx]}],
                label=str(year),
            )
        )

    menu_config = dict(
        type="buttons",
        direction="down",
        active=0,
        buttons=buttons,
        x=1.02,
        xanchor="left",
        y=1,
        yanchor="top",
        showactive=True,
    )
    if nav_options:
        menu_config.update(nav_options)

    fig.update_layout(updatemenus=[menu_config])

    return fig


def calheatmap(
    data: DataFrame,
    x: str,
    y: str,
    name: str = "y",
    dark_theme: bool = False,
    month_lines_width: int = 1,
    month_lines_color: str = "#9e9e9e",
    gap: int = 1,
    years_title: bool = False,
    colorscale: Union[str, list] = "greens",
    title: str = "",
    month_lines: bool = True,
    total_height: Union[int, None] = None,
    space_between_plots: float = 0.08,
    showscale: Union[bool, str] = False,
    scale_ticks: bool = False,
    text: Optional[str] = None,
    years_as_columns: bool = False,
    cmap_min: Optional[float] = None,
    cmap_max: Optional[float] = None,
    start_month: int = 1,
    end_month: int = 12,
    date_fmt: str = "%Y-%m-%d",
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
    navigation: bool = False,
    nav_options: Optional[Dict[str, Any]] = None,
    hovertemplate: Optional[str] = None,
    customdata: Optional[List[str]] = None,
    vertical: bool = False,
    month_gap: int = 0,
    agg: Optional[Literal["sum", "mean", "count", "max"]] = None,
) -> go.Figure:
    """
    Yearly Calendar Heatmap

    Parameters
    ----------
    data : DataFrame
        Must contain at least one date like column and
        one value column for displaying in the plot

    x : str
        The name of the date like column in data

    y : str
        The name of the value column in data

    dark_theme : bool = False
        Option for creating a dark themed plot

    month_lines: bool = True
        if true will plot a separation line between
        each month in the calendar

    month_lines_width : int = 1
        if month_lines this option controls the width of
        the line between each month in the calendar

    month_lines_color : str = "#9e9e9e"
        if month_lines this option controls the color of
        the line between each month in the calendar

    gap : int = 1
        controls the gap bewteen daily squares

    years_title : bool = False
        if true will add a title for each subplot with the
        correspondent year

    colorscale : str | list = "greens"
        controls the colorscale for the calendar, works
        with all the standard Plotly Colorscales and also
        supports custom colorscales (e.g. [[0, "#eee"], [1, "#333"]])

    title : str = ""
        title of the plot

    total_height : int = None
        if provided a value, will force the plot to have a specific
        height, otherwise the total height will be calculated
        according to the amount of years in data

    space_between_plots : float = 0.08
        controls the vertical space between the plots

    showscale : bool | str = False
        if True or a string, a horizontal color legend will be shown.
        Pass a string (e.g. "Temperature") to display a title on the legend.

    text : Optional[str] = None
        The name of the column in data to include in hovertext.

    years_as_columns : bool = False
        if True will plot all years in a single line

    cmap_min : float = None
        colomap min, defaults to min value of the data

    cmap_max : float = None
        colomap max, defaults to max value of the data

    start_month : int = 1
        starting month range to plot, defaults to 1 (January)

    end_month : int = 12
        ending month range to plot, defaults to 12 (December)

    date_fmt : str = "%Y-%m-%d"
        date format for the date column in data, defaults to "%Y-%m-%d"
        If the date column is already in datetime format, this parameter
        will be ignored.

    paper_bgcolor : str = None
        override paper background color (e.g. "#0d1117")

    plot_bgcolor : str = None
        override plot background color (e.g. "#0d1117")

    font_color : str = None
        override font color (e.g. "#8b949e")

    font_size : int = None
        override font size (e.g. 11)

    title_font_color : str = None
        override title font color (e.g. "#c9d1d9")

    title_font_size : int = None
        override title font size (e.g. 14)

    width : int = None
        figure width in pixels

    margin : dict = None
        custom margins dict, e.g. {"l": 40, "r": 20, "t": 40, "b": 20}

    month_labels_side : str = "bottom"
        position of month labels on x-axis: "top" or "bottom"

    navigation : bool = False
        if True and there are multiple years, shows one year at a time
        with year buttons on the right (GitHub-style)

    nav_options : dict = None
        styling overrides for the navigation buttons. Supports any key
        accepted by Plotly's updatemenus (e.g. font, bgcolor, bordercolor,
        borderwidth, x, y, xanchor, yanchor, direction, pad)

    hovertemplate : str = None
        custom hover template string. Supports friendly {placeholder}
        syntax that gets resolved automatically:
            {date}             -> the date value
            {date:%d/%m/%Y}    -> the date with custom strftime format
            {name}             -> the metric name
            {value}            -> the cell value (z)
            {week}             -> the week number
            {text}             -> the text column value
            {col}              -> any column name from the `customdata` list
        Raw Plotly syntax (%{z}, %{customdata[0]}, etc.) also works.
        Example: "<b>{date:%d/%m/%Y}</b><br>{value} commits · {repo}"

    customdata : list[str] = None
        list of column names from `data` to include as extra customdata.
        These become available as {column_name} in hovertemplate.

    vertical : bool = False
        if True renders months as rows instead of columns, with weeks
        flowing top-to-bottom and days of the week as columns.

    month_gap : int = 0
        extra spacing (in week-units) inserted between each month
        for clearer visual separation. 0 means no extra gap.

    agg : str = None
        aggregation function to apply when multiple rows share the same
        date.  Accepts ``"sum"``, ``"mean"``, ``"count"``, or ``"max"``.
        When provided, raw (non-aggregated) event data can be passed
        directly; dates will be grouped and aggregated automatically.
    """
    data = data.copy()
    data[x] = validate_date_column(data[x], date_fmt)

    if agg is not None:
        data[x] = data[x].dt.normalize()
        agg_cols = {y: agg}
        if text is not None:
            agg_cols[text] = "first"
        if customdata is not None:
            for col in customdata:
                if col not in agg_cols:
                    agg_cols[col] = "first"
        data = data.groupby(x, as_index=False).agg(agg_cols)
    unique_years = data[x].dt.year.unique()
    unique_years_amount = len(unique_years)

    # navigation only makes sense with multiple years
    use_navigation = navigation and unique_years_amount > 1

    if years_title:
        subplot_titles = unique_years.astype(str)
    else:
        subplot_titles = None

    if use_navigation:
        rows = 1
        cols = 1
        subplot_titles = None
    elif years_as_columns:
        rows = 1
        cols = unique_years_amount
    else:
        rows = unique_years_amount
        cols = 1

    # if single row calheatmap, the height can be constant
    if total_height is None:
        if vertical:
            total_height = 800
        elif use_navigation or years_as_columns:
            total_height = 150
        else:
            total_height = 150 * unique_years_amount

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=space_between_plots,
    )

    # getting cmap_min and cmap_max
    if cmap_min is None:
        cmap_min = data[y].min()

    if cmap_max is None:
        cmap_max = data[y].max()

    data = data[
        data[x].dt.month.isin(np.arange(start_month, end_month + 1, 1).tolist())
    ]

    # Track trace counts per year for navigation
    trace_counts = []

    for i, year in enumerate(unique_years):
        traces_before = len(fig.data)

        selected_year_data = data.loc[data[x].dt.year == year]
        selected_year_data = fill_empty_with_zeros(
            selected_year_data, x, year, start_month, end_month
        )

        year_calheatmap(
            selected_year_data,
            x,
            y,
            name=name,
            month_lines=month_lines,
            month_lines_width=month_lines_width,
            month_lines_color=month_lines_color,
            colorscale=colorscale,
            year=year,
            fig=fig,
            dark_theme=dark_theme,
            gap=gap,
            title=title,
            row=0 if use_navigation else i,
            total_height=total_height,
            text=None if text is None else selected_year_data[text].tolist(),
            text_name=text,
            years_as_columns=years_as_columns if not use_navigation else False,
            start_month=start_month,
            end_month=end_month,
            locale=locale,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
            font_color=font_color,
            font_size=font_size,
            title_font_color=title_font_color,
            title_font_size=title_font_size,
            width=width,
            margin=margin,
            month_labels_side=month_labels_side,
            hovertemplate=hovertemplate,
            extra_customdata_columns=customdata,
            vertical=vertical,
            month_gap=month_gap,
        )

        trace_counts.append(len(fig.data) - traces_before)

    fig = apply_general_colorscaling(fig, cmap_min, cmap_max)
    if showscale:
        scale_title = showscale if isinstance(showscale, str) else ""
        tick_vals = np.linspace(cmap_min, cmap_max, 5).tolist() if scale_ticks else None
        fig = showscale_of_heatmaps(fig, scale_title=scale_title, scale_ticks=tick_vals)

    if use_navigation:
        fig = _add_year_navigation(
            fig, unique_years, trace_counts, nav_options=nav_options
        )

    return fig


def month_calheatmap(
    data: Optional[DataFrame] = None,
    x: str | Series = "x",
    y: str | Series = "y",
    name: str = "y",
    dark_theme: bool = False,
    gap: int = 2,
    colorscale: Union[str, list] = "greens",
    title: str = "",
    year_height: int = 30,
    total_height: Union[int, None] = None,
    showscale: Union[bool, str] = False,
    scale_ticks: bool = False,
    date_fmt: str = "%Y-%m-%d",
    locale: Optional[str] = None,
    paper_bgcolor: Optional[str] = None,
    plot_bgcolor: Optional[str] = None,
    font_color: Optional[str] = None,
    font_size: Optional[int] = None,
    title_font_color: Optional[str] = None,
    title_font_size: Optional[int] = None,
    width: Optional[int] = None,
    margin: Optional[dict] = None,
    agg: Optional[Literal["sum", "mean", "count", "max"]] = None,
) -> go.Figure:
    """
    Yearly Calendar Heatmap by months (12 cols per row)

    Parameters
    ----------
    data : DataFrame | None
        Must contain at least one date like column and
        one value column for displaying in the plot. If data is None, x and y will
        be used

    x : str | Iterable
        The name of the date like column in data or the column if data is None

    y : str | Iterable
        The name of the value column in data or the column if data is None

    dark_theme : bool = False
        Option for creating a dark themed plot

    gap : int = 2
        controls the gap bewteen monthly squares

    colorscale : str | list = "greens"
        controls the colorscale for the calendar, works
        with all the standard Plotly Colorscales and also
        supports custom colorscales (e.g. [[0, "#eee"], [1, "#333"]])

    title : str = ""
        title of the plot

    year_height: int = 30
        the height per year to be used if total_height is None

    total_height : int = None
        if provided a value, will force the plot to have a specific
        height, otherwise the total height will be calculated
        according to the amount of years in data

    showscale : bool | str = False
        if True or a string, a horizontal color legend will be shown.
        Pass a string (e.g. "Temperature") to display a title on the legend.

    date_fmt : str = "%Y-%m-%d"
        date format for the date column in data, defaults to "%Y-%m-%d"
        If the date column is already in datetime format, this parameter
        will be ignored.

    paper_bgcolor : str = None
        override paper background color

    plot_bgcolor : str = None
        override plot background color

    font_color : str = None
        override font color

    font_size : int = None
        override font size

    title_font_color : str = None
        override title font color

    title_font_size : int = None
        override title font size

    width : int = None
        figure width in pixels

    margin : dict = None
        custom margins dict, e.g. {"l": 40, "r": 20, "t": 40, "b": 20}
    """
    if data is None:
        if not isinstance(x, Series):
            x = Series(x, dtype="datetime64[ns]", name="x")

        if not isinstance(y, Series):
            y = Series(y, dtype="float64", name="y")

        data = DataFrame({x.name: x, y.name: y})

        x = str(x.name)
        y = str(y.name)

    data = data.copy()
    data[x] = validate_date_column(data[x], date_fmt)

    if agg is not None:
        data[x] = data[x].dt.normalize()
        data = data.groupby(x, as_index=False).agg({y: agg})

    gData = data.set_index(x)[y].groupby(Grouper(freq="ME")).sum()
    unique_years = gData.index.year.unique()
    unique_years_amount = len(unique_years)

    if total_height is None:
        total_height = 20 + max(10, year_height * unique_years_amount)

    # Build styling overrides for _get_subplot_layout
    extra_kwargs: Dict[str, Any] = {}
    if paper_bgcolor is not None:
        extra_kwargs["paper_bgcolor"] = paper_bgcolor
    if plot_bgcolor is not None:
        extra_kwargs["plot_bgcolor"] = plot_bgcolor
    if width is not None:
        extra_kwargs["width"] = width

    font_overrides: Dict[str, Any] = {}
    if font_color is not None:
        font_overrides["color"] = font_color
    if font_size is not None:
        font_overrides["size"] = font_size
    if font_overrides:
        extra_kwargs["font"] = font_overrides

    if margin is not None:
        extra_kwargs["margin"] = margin

    title_val: Any = title
    if title and (title_font_color or title_font_size):
        title_font: Dict[str, Any] = {}
        if title_font_color:
            title_font["color"] = title_font_color
        if title_font_size:
            title_font["size"] = title_font_size
        title_val = dict(text=title, font=title_font)

    layout = _get_subplot_layout(
        dark_theme=dark_theme,
        height=total_height,
        title=title_val,
        yaxis={
            "tickvals": unique_years,
        },
        xaxis={
            "tickvals": list(range(1, 13)),
            "ticktext": [n[:3] for n in get_localized_month_names(locale)],
            "tickangle": 45,
        },
        **extra_kwargs,
    )

    # hovertext = _gen_hoverText(gData.index.month, gData.index.year, gData)
    hovertext = gData.apply(lambda x: f"{x: .0f}")

    scale_title = showscale if isinstance(showscale, str) else ""
    if showscale:
        colorbar_config = dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            thickness=10,
            len=0.5,
            title=dict(text=scale_title, side="top"),
        )
        if scale_ticks:
            zmin, zmax = gData.min(), gData.max()
            colorbar_config["tickvals"] = np.linspace(zmin, zmax, 5).tolist()
    else:
        colorbar_config = None

    cplt = go.Heatmap(
        x=gData.index.month,
        y=gData.index.year,
        z=gData,
        name=title,
        showscale=showscale,
        xgap=gap,
        ygap=gap,
        colorscale=colorscale,
        hoverinfo="text",
        text=hovertext,
        colorbar=colorbar_config,
    )

    fig = go.Figure(data=cplt, layout=layout)

    return fig
