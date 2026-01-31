from typing import Any, Dict, Literal, Optional, Union, List, Tuple

import numpy as np
from pandas import DataFrame, Grouper, Series
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from plotly_calheatmap.layout_formatter import (
    apply_general_colorscaling,
    get_theme_defaults,
    showscale_of_heatmaps,
)
from plotly_calheatmap.single_year_calheatmap import year_calheatmap
from plotly_calheatmap.i18n import get_localized_month_names
from plotly_calheatmap.utils import fill_empty_with_zeros, validate_date_column
from plotly_calheatmap.colorscale_utils import (
    compute_colorscale,
    _apply_zero_color,
    _apply_nan_color,
    extract_legend_bins,
)


def _get_subplot_layout(**kwargs: Any) -> go.Layout:
    """
    Combines the default subplot layout with the customized parameters
    """
    dark_theme: bool = kwargs.pop("dark_theme", False)
    yaxis: Dict[str, Any] = kwargs.pop("yaxis", {})
    xaxis: Dict[str, Any] = kwargs.pop("xaxis", {})

    _paper_bgcolor, _plot_bgcolor, _font_color = get_theme_defaults(dark_theme)

    # Build font with defaults, allow override
    font = {"size": 10, "color": _font_color}
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
            "plot_bgcolor": _plot_bgcolor,
            "paper_bgcolor": _paper_bgcolor,
            "margin": {"t": 20, "b": 20},
            "showlegend": False,
            "autosize": True,
            **kwargs,
        }
    )


def _merge_layers(
    layers: List[Dict[str, Any]],
    overlap_colorscale: Union[str, list],
    date_fmt: str = "%Y-%m-%d",
    agg: Optional[str] = None,
) -> Tuple[DataFrame, list, str, str, List[str]]:
    """Merge multiple layer DataFrames into a single DataFrame with composite z-values.

    Parameters
    ----------
    layers : list of dict
        Each dict must have ``"data"`` (DataFrame), ``"x"`` (date col),
        ``"y"`` (value col), ``"colorscale"`` (str or list), and
        optionally ``"name"`` (label).
    overlap_colorscale : str or list
        Colorscale for days present in multiple layers.
    date_fmt : str
        Date format for parsing.
    agg : str, optional
        Aggregation function (applied per-layer before merging).

    Returns
    -------
    merged_data : DataFrame
        Unified DataFrame with columns: ``_layer_date``, ``_layer_z``
        (remapped into composite bands), ``_layer_value`` (display value),
        ``_layer_source`` (source label), plus per-layer value columns.
    composite_colorscale : list
        Plotly colorscale with bands for each layer + overlap.
    x_col : str
        Name of the date column (``"_layer_date"``).
    y_col : str
        Name of the remapped z column (``"_layer_z"``).
    layer_names : list of str
        Names of each layer for hover display.
    """
    import pandas as pd
    from plotly_calheatmap.colorscale_utils import build_composite_colorscale

    n_layers = len(layers)
    layer_names = [layer.get("name", f"Layer {i}") for i, layer in enumerate(layers)]

    # Build per-layer date→value mappings
    layer_series = []
    for layer in layers:
        df = layer["data"].copy()
        x_col = layer["x"]
        y_col = layer["y"]
        df[x_col] = validate_date_column(df[x_col], date_fmt)
        if agg is not None:
            df[x_col] = df[x_col].dt.normalize()
            df = df.groupby(x_col, as_index=False).agg({y_col: agg})
        series = df.set_index(x_col)[y_col]
        # Remove duplicates by keeping the sum
        series = series.groupby(series.index).sum()
        layer_series.append(series)

    # Combine all dates
    all_dates = sorted(set().union(*(s.index for s in layer_series)))

    # Classify each date and compute values
    records = []
    for date in all_dates:
        present = [
            i
            for i, s in enumerate(layer_series)
            if date in s.index and not np.isnan(s[date])
        ]
        if len(present) == 0:
            continue
        elif len(present) == 1:
            source_idx = present[0]
            value = float(layer_series[source_idx][date])
            source_label = layer_names[source_idx]
        else:
            # Overlap — sum values
            source_idx = n_layers  # overlap band
            value = sum(float(layer_series[i][date]) for i in present)
            source_label = " + ".join(layer_names[i] for i in present)

        record = {
            "_layer_date": date,
            "_layer_value": value,
            "_layer_source_idx": source_idx,
            "_layer_source": source_label,
        }
        # Store per-layer values for hover
        for i, name in enumerate(layer_names):
            if i in present:
                record[f"_lv_{name}"] = float(layer_series[i][date])
            else:
                record[f"_lv_{name}"] = 0.0
        records.append(record)

    merged = pd.DataFrame(records)
    if merged.empty:
        merged["_layer_z"] = []
        composite = build_composite_colorscale(
            [layer["colorscale"] for layer in layers], overlap_colorscale
        )
        return merged, composite, "_layer_date", "_layer_z", layer_names

    # Remap values into composite bands
    n_bands = n_layers + 1
    band_width = 1.0 / n_bands

    # Compute min/max per band for normalization
    band_mins = {}
    band_maxs = {}
    for band_idx in range(n_bands):
        band_vals = merged.loc[merged["_layer_source_idx"] == band_idx, "_layer_value"]
        if len(band_vals) > 0:
            band_mins[band_idx] = band_vals.min()
            band_maxs[band_idx] = band_vals.max()
        else:
            band_mins[band_idx] = 0.0
            band_maxs[band_idx] = 1.0

    def remap(row):
        idx = int(row["_layer_source_idx"])
        val = row["_layer_value"]
        bmin = band_mins[idx]
        bmax = band_maxs[idx]
        if bmax == bmin:
            t = 0.5
        else:
            t = (val - bmin) / (bmax - bmin)
        # Small margin so we don't sit exactly on band boundaries
        margin = 0.01 * band_width
        band_start = idx * band_width + margin
        band_end = (idx + 1) * band_width - margin
        return band_start + t * (band_end - band_start)

    merged["_layer_z"] = merged.apply(remap, axis=1)

    # Build composite colorscale
    composite = build_composite_colorscale(
        [layer["colorscale"] for layer in layers], overlap_colorscale
    )

    return merged, composite, "_layer_date", "_layer_z", layer_names



_TOOLBAR_HEIGHT = 50  # extra pixels reserved for the toolbar row


def _apply_toolbar_layout(fig: go.Figure) -> None:
    """Reserve space for the toolbar row and move the title into it.

    Adds ``_TOOLBAR_HEIGHT`` pixels to both the top margin and the
    total figure height so the plot area is not compressed.  The title
    is repositioned to sit centered in the toolbar row alongside the
    dropdown (left) and year buttons (right).
    """
    current_margin = fig.layout.margin
    top = current_margin.t if current_margin.t is not None else 20
    new_top = max(top, 30) + _TOOLBAR_HEIGHT
    updates: dict = {"margin": {"t": new_top}}

    # Grow the figure so the plot area keeps its original size.
    total_h = fig.layout.height
    if total_h is not None:
        total_h = total_h + _TOOLBAR_HEIGHT
        updates["height"] = total_h

    # Place the title in the middle of the top margin area.
    # title.y uses container-relative coords (0=bottom, 1=top).
    if fig.layout.title and fig.layout.title.text and total_h:
        title_y = 1.0 - (new_top / 2) / total_h
        updates["title"] = {
            "y": title_y,
            "yanchor": "middle",
            "x": 0.5,
            "xanchor": "center",
        }

    fig.update_layout(**updates)


def _prepare_dataset_configs(
    datasets: Optional[Dict[str, Dict[str, Any]]],
    y: str,
    colorscale: Union[str, list],
    showscale: Union[bool, str],
    cmap_min: Optional[float],
    cmap_max: Optional[float],
    name: str,
    colors: Optional[List[str]] = None,
    scale_type: str = "linear",
    zero_color: Optional[str] = None,
    nan_color: Optional[str] = None,
    pivot: Optional[float] = None,
    symmetric: bool = False,
    bins: Optional[List[tuple]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Normalize the ``datasets`` parameter into a uniform config dict.

    When *datasets* is ``None`` a single-dataset config is created from
    the legacy positional parameters so the rest of the pipeline can
    always iterate over the returned dict.
    """
    if datasets is None:
        return {
            name: {
                "y": y,
                "colorscale": colorscale,
                "showscale": showscale,
                "cmap_min": cmap_min,
                "cmap_max": cmap_max,
                "name": name,
                "colors": colors,
                "scale_type": scale_type,
                "zero_color": zero_color,
                "nan_color": nan_color,
                "pivot": pivot,
                "symmetric": symmetric,
                "bins": bins,
            }
        }

    normalized: Dict[str, Dict[str, Any]] = {}
    for label, config in datasets.items():
        normalized[label] = {
            "y": config["y"],
            "colorscale": config.get("colorscale", colorscale),
            "showscale": config.get("showscale", label),
            "cmap_min": config.get("cmap_min", cmap_min),
            "cmap_max": config.get("cmap_max", cmap_max),
            "name": config.get("name", label),
            "colors": config.get("colors", colors),
            "scale_type": config.get("scale_type", scale_type),
            "zero_color": config.get("zero_color", zero_color),
            "nan_color": config.get("nan_color", nan_color),
            "pivot": config.get("pivot", pivot),
            "symmetric": config.get("symmetric", symmetric),
            "bins": config.get("bins", bins),
        }
    return normalized


def _add_dataset_navigation(
    fig: go.Figure,
    dataset_configs: Dict[str, Dict[str, Any]],
    trace_structure: List[Dict[str, Any]],
    unique_years: Any,
    use_navigation: bool,
    nav_options: Optional[Dict[str, Any]] = None,
) -> dict:
    """Build the dataset dropdown menu config.

    Returns a Plotly ``updatemenu`` dict (not applied to the figure).
    The dropdown is placed in the top-left of the toolbar row.
    """
    total_traces = len(fig.data)
    buttons = []

    for dataset_name, config in dataset_configs.items():
        vis = [False] * total_traces
        for t in trace_structure:
            if t["dataset"] != dataset_name:
                continue
            if use_navigation and t["year"] != unique_years[0]:
                continue
            for i in range(t["start"], t["start"] + t["count"]):
                vis[i] = True

        buttons.append(
            dict(
                method="update",
                args=[{"visible": vis}],
                label=dataset_name,
            )
        )

    menu_config = dict(
        type="dropdown",
        direction="down",
        active=0,
        buttons=buttons,
        x=0,
        xanchor="left",
        y=1.02,
        yanchor="bottom",
        showactive=True,
    )
    if nav_options:
        menu_config.update(nav_options)
    return menu_config


def _add_year_navigation(
    fig: go.Figure,
    unique_years: Any,
    trace_counts: list,
    nav_options: Optional[Dict[str, Any]] = None,
    trace_structure: Optional[List[Dict[str, Any]]] = None,
    current_dataset: Optional[str] = None,
) -> Union[go.Figure, dict]:
    """Add year navigation buttons.

    When *trace_structure* and *current_dataset* are provided (multi-dataset
    mode), returns the menu config dict instead of applying it directly.

    By default buttons are placed in the top-right of the toolbar row,
    laid out horizontally.

    Parameters
    ----------
    nav_options : dict, optional
        Styling overrides for the button group.
    trace_structure : list, optional
        Full trace metadata list (multi-dataset mode).
    current_dataset : str, optional
        The default-active dataset label.
    """
    total_traces = len(fig.data)

    if trace_structure is not None and current_dataset is not None:
        buttons = []
        for year in unique_years:
            vis = [False] * total_traces
            for t in trace_structure:
                if t["dataset"] == current_dataset and t["year"] == year:
                    for i in range(t["start"], t["start"] + t["count"]):
                        vis[i] = True
            buttons.append(
                dict(method="restyle", args=[{"visible": vis}], label=str(year))
            )
    else:
        offsets = []
        acc = 0
        for count in trace_counts:
            offsets.append(acc)
            acc += count

        for i in range(trace_counts[0], total_traces):
            fig.data[i].visible = False

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
        direction="right",
        active=0,
        buttons=buttons,
        x=1,
        xanchor="right",
        y=1.02,
        yanchor="bottom",
        showactive=True,
    )
    if nav_options:
        menu_config.update(nav_options)

    if trace_structure is not None:
        return menu_config

    fig.update_layout(updatemenus=[menu_config])
    return fig


def calheatmap(
    data: Optional[DataFrame] = None,
    x: str = "",
    y: str = "",
    name: str = "y",
    dark_theme: bool = False,
    month_lines_width: int = 1,
    month_lines_color: str = "#9e9e9e",
    gap: int = 1,
    years_title: bool = False,
    colorscale: Union[str, list] = "greens",
    colors: Optional[List[str]] = None,
    scale_type: Optional[
        Literal["linear", "diverging", "quantile", "quantize", "categorical"]
    ] = "linear",
    bins: Optional[List[tuple]] = None,
    zero_color: Optional[str] = None,
    nan_color: Optional[str] = None,
    pivot: Optional[float] = None,
    symmetric: bool = False,
    title: str = "",
    month_lines: bool = True,
    top_bottom_lines: bool = False,
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
    skip_empty_years: bool = False,
    replace_nans_with_zeros: bool = False,
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
    grouping: Optional[Literal["month", "bimester", "quarter", "semester"]] = None,
    grouping_lines_width: int = 2,
    grouping_lines_color: Optional[str] = None,
    log_scale: bool = False,
    datasets: Optional[Dict[str, Dict[str, Any]]] = None,
    dataset_nav_options: Optional[Dict[str, Any]] = None,
    annotations: bool = False,
    annotations_fmt: Optional[str] = None,
    annotations_font_size: Optional[int] = None,
    annotations_font_color: Optional[str] = None,
    annotations_font_family: Optional[str] = None,
    legend_style: Literal["colorbar", "legend"] = "colorbar",
    colorbar_options: Optional[Dict[str, Any]] = None,
    legend_options: Optional[Dict[str, Any]] = None,
    week_start: Literal["monday", "sunday", "saturday"] = "monday",
    layout: Literal["github", "calendar"] = "github",
    cols: int = 4,
    layers: Optional[List[Dict[str, Any]]] = None,
    overlap_colorscale: Union[str, list] = "greens",
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

    top_bottom_lines : bool = False
        if True, draws horizontal lines at the top and bottom edges
        of the calendar. Combined with month_lines, this fully
        encloses each month.

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

    skip_empty_years : bool = False
        if True, years where the sum of y is less than 1 will be
        skipped, preventing empty subplots.

    replace_nans_with_zeros : bool = False
        if True, dates without data will be displayed as 0 instead
        of NaN.

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

    grouping : str = None
        time grouping for separator lines and axis labels. Accepts
        ``"month"`` (default behavior), ``"bimester"``, ``"quarter"``,
        or ``"semester"``. When set, thicker lines are drawn at group
        boundaries and axis tick labels show group names (e.g. Q1, Q2).
        Month lines are still drawn when ``month_lines=True``.

    grouping_lines_width : int = 2
        line width for the grouping boundary lines. Defaults to 2
        (thicker than month_lines_width) so they stand out.

    grouping_lines_color : str = None
        color for the grouping boundary lines. Defaults to
        ``month_lines_color`` if not provided.

    log_scale : bool = False
        if True, applies a logarithmic color scale using ``log(1 + x)``
        so that extreme values don't wash out the rest of the heatmap.
        Hover text still displays the original (non-transformed) values.

    datasets : dict = None
        Multiple datasets to swap between via a dropdown menu.
        Keys are display labels, values are dicts with:
            - "y" (str, required): column name for values
            - "colorscale" (str or list, optional): dataset-specific colorscale
            - "showscale" (str or bool, optional): legend title
            - "cmap_min", "cmap_max" (float, optional): value range
            - "name" (str, optional): name for hover template
        Example::

            datasets={
                "Sales": {"y": "sales", "colorscale": "greens"},
                "Activity": {"y": "activity", "colorscale": "blues"},
            }

    dataset_nav_options : dict = None
        Styling overrides for the dataset dropdown menu. Supports any key
        accepted by Plotly's updatemenus (e.g. x, y, xanchor, yanchor,
        font, bgcolor, bordercolor, borderwidth, direction, pad).
        Example: ``dataset_nav_options={"x": 0.5, "xanchor": "center"}``

    annotations : bool = False
        if True, displays cell values as text annotations inside each cell.

    annotations_fmt : str = None
        format string for annotations, using Plotly's texttemplate syntax
        (e.g. "%{z:.0f}"). Implies ``annotations=True``.

    annotations_font_size : int = None
        Font size for cell annotations. Defaults to 10.

    annotations_font_color : str = None
        Font color for cell annotations (e.g. "white", "#333").

    annotations_font_family : str = None
        Font family for cell annotations (e.g. "Arial", "Courier New").

    legend_style : str = "colorbar"
        ``"colorbar"`` shows a continuous color bar (default).
        ``"legend"`` creates discrete clickable legend items — one per
        bin — so users can toggle categories on/off.  Requires a
        discrete ``scale_type`` (``"quantile"``, ``"quantize"``, or
        ``"categorical"``) or explicit ``bins``.

    colorbar_options : dict = None
        Override any Plotly colorbar property when ``legend_style="colorbar"``.
        Supports ``orientation``, ``x``, ``y``, ``xanchor``, ``yanchor``,
        ``thickness``, ``len``, ``tickformat``, ``nticks``, ``title``, etc.
        Example::

            colorbar_options={"orientation": "v", "x": 1.02, "thickness": 15}

    legend_options : dict = None
        Override any Plotly legend property when ``legend_style="legend"``.
        Supports ``orientation``, ``x``, ``y``, ``xanchor``, ``yanchor``,
        ``bgcolor``, ``font``, ``title``, etc.
        Example::

            legend_options={"orientation": "h", "y": -0.1, "x": 0.5}
    """
    # --- Calendar layout: delegate to calendar_calheatmap implementation ---
    if layout == "calendar":
        from .calendar_calheatmap import _calendar_calheatmap_impl

        return _calendar_calheatmap_impl(
            data=data,
            x=x,
            y=y,
            name=name,
            dark_theme=dark_theme,
            cols=cols,
            gap=gap,
            colorscale=colorscale,
            title=title,
            showscale=showscale,
            total_height=total_height,
            width=width,
            margin=margin,
            cmap_min=cmap_min,
            cmap_max=cmap_max,
            log_scale=log_scale,
            date_fmt=date_fmt,
            agg=agg,
            locale=locale,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
            font_color=font_color,
            font_size=font_size,
            title_font_color=title_font_color,
            title_font_size=title_font_size,
            hovertemplate=hovertemplate,
            start_month=start_month,
            end_month=end_month,
            week_start=week_start,
        )

    # Validate: either data+x+y or layers must be provided
    if layers is None and (data is None or not x or not y):
        raise ValueError("Either provide data/x/y or use the layers parameter.")

    # annotations_fmt implies annotations=True
    if annotations_fmt is not None:
        annotations = True

    # --- Multi-layer mode ---
    _layer_names = None
    if layers is not None:
        merged_data, composite_cs, x, y, _layer_names = _merge_layers(
            layers,
            overlap_colorscale,
            date_fmt=date_fmt,
            agg=agg,
        )
        data = merged_data
        colorscale = composite_cs
        cmap_min = 0.0
        cmap_max = 1.0
        agg = None  # already aggregated
        name = "Value"
        # Extra customdata columns (appended after pipeline's [date, name] at indices 0,1)
        # Index 2 = _layer_source, 3..N+2 = per-layer values, N+3 = total
        extra_cols = (
            ["_layer_source"] + [f"_lv_{ln}" for ln in _layer_names] + ["_layer_value"]
        )
        customdata = extra_cols
        # Build a hover template showing per-layer breakdown
        if hovertemplate is None:
            base_offset = 2  # pipeline always prepends date[0] and name[1]
            src_idx = base_offset  # _layer_source
            hover_parts = ["<b>%{customdata[0]}</b>"]
            hover_parts.append("Source: %{customdata[" + str(src_idx) + "]}")
            for i, ln in enumerate(_layer_names):
                col_idx = base_offset + 1 + i
                hover_parts.append(f"{ln}: %{{customdata[{col_idx}]}}")
            total_idx = base_offset + 1 + len(_layer_names)
            hover_parts.append("Total: %{customdata[" + str(total_idx) + "]}")
            hover_parts.append("<extra></extra>")
            hovertemplate = "<br>".join(hover_parts)
        # Pre-format numeric columns as strings (pipeline converts to str via numpy)
        for col in [f"_lv_{ln}" for ln in _layer_names] + ["_layer_value"]:
            data[col] = data[col].apply(
                lambda v: f"{v:,.2f}" if not np.isnan(v) else ""
            )

    data = data.copy()
    data[x] = validate_date_column(data[x], date_fmt)

    # Normalize dataset configs
    dataset_configs = _prepare_dataset_configs(
        datasets,
        y,
        colorscale,
        showscale,
        cmap_min,
        cmap_max,
        name,
        colors=colors,
        scale_type=scale_type,
        zero_color=zero_color,
        nan_color=nan_color,
        pivot=pivot,
        symmetric=symmetric,
        bins=bins,
    )
    use_dataset_swap = len(dataset_configs) > 1

    # Collect all y columns needed for aggregation
    all_y_cols = list({cfg["y"] for cfg in dataset_configs.values()})

    if agg is not None:
        data[x] = data[x].dt.normalize()
        agg_cols = {col: agg for col in all_y_cols}
        if text is not None:
            agg_cols[text] = "first"
        if customdata is not None:
            for col in customdata:
                if col not in agg_cols:
                    agg_cols[col] = "first"
        data = data.groupby(x, as_index=False).agg(agg_cols)

    # Use first y column for skip_empty_years check
    primary_y = list(dataset_configs.values())[0]["y"]
    unique_years = data[x].dt.year.unique()

    if skip_empty_years:
        unique_years = np.array(
            [
                yr
                for yr in unique_years
                if data.loc[data[x].dt.year == yr, primary_y].sum() >= 1
            ]
        )

    unique_years_amount = len(unique_years)

    # navigation only makes sense with multiple years
    use_navigation = navigation and unique_years_amount > 1

    if years_title:
        subplot_titles = unique_years.astype(str)
    else:
        subplot_titles = None

    if use_navigation or use_dataset_swap:
        rows = 1
        cols = 1
        subplot_titles = None
    elif years_as_columns:
        rows = 1
        cols = unique_years_amount
    else:
        rows = unique_years_amount
        cols = 1

    # Auto-compute height based on content when not explicitly provided.
    # The figure always uses autosize=True for width (adapts to container),
    # so users only need to override total_height for special cases.
    if total_height is None:
        month_count = end_month - start_month + 1
        if vertical:
            # Vertical: height scales with number of weeks shown
            total_height = max(400, int(month_count * 65))
        elif use_navigation or years_as_columns or use_dataset_swap:
            # Single-row modes: fixed comfortable height
            total_height = 200
        else:
            # Stacked years: scale per year with a comfortable minimum
            per_year = 160 if month_count >= 10 else max(100, int(month_count * 16))
            total_height = max(200, per_year * unique_years_amount)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=space_between_plots,
    )

    data = data[
        data[x].dt.month.isin(np.arange(start_month, end_month + 1, 1).tolist())
    ]

    # Track trace counts per year for navigation (legacy) and full structure
    trace_counts: list = []
    trace_structure: List[Dict[str, Any]] = []

    use_discrete_legend = legend_style == "legend"

    for dataset_label, ds_cfg in dataset_configs.items():
        ds_y = ds_cfg["y"]
        ds_colorscale = ds_cfg["colorscale"]
        ds_name = ds_cfg["name"]

        # Compute per-dataset color range
        ds_cmap_min = ds_cfg["cmap_min"]
        ds_cmap_max = ds_cfg["cmap_max"]
        if ds_cmap_min is None:
            ds_cmap_min = data[ds_y].min()
        if ds_cmap_max is None:
            ds_cmap_max = data[ds_y].max()

        # Compute colorscale from colors list or bins if provided
        ds_nan_color = ds_cfg.get("nan_color")
        ds_nan_sentinel = None
        ds_colors = ds_cfg.get("colors")
        ds_bins = ds_cfg.get("bins")
        ds_scale_type = ds_cfg.get("scale_type", "linear")
        if ds_colors is not None or ds_scale_type == "categorical":
            result = compute_colorscale(
                colors=ds_colors,
                scale_type=ds_scale_type,
                data=data[ds_y].dropna().values,
                data_min=ds_cmap_min,
                data_max=ds_cmap_max,
                pivot=ds_cfg.get("pivot"),
                symmetric=ds_cfg.get("symmetric", False),
                zero_color=ds_cfg.get("zero_color"),
                nan_color=ds_nan_color,
                bins=ds_bins,
            )
            if ds_nan_color is not None:
                ds_colorscale, ds_nan_sentinel = result
            else:
                ds_colorscale = result
            ds_cfg["colorscale"] = ds_colorscale
        elif ds_cfg.get("zero_color") is not None and isinstance(ds_colorscale, list):
            ds_colorscale = _apply_zero_color(
                ds_colorscale,
                ds_cfg["zero_color"],
                ds_cmap_min,
                ds_cmap_max,
            )
            ds_cfg["colorscale"] = ds_colorscale

        # Apply nan_color when using a pre-built colorscale (no colors list)
        if (
            ds_nan_color is not None
            and ds_nan_sentinel is None
            and isinstance(ds_colorscale, list)
        ):
            ds_colorscale, ds_nan_sentinel = _apply_nan_color(
                ds_colorscale,
                ds_nan_color,
                ds_cmap_min,
                ds_cmap_max,
            )
            ds_cfg["colorscale"] = ds_colorscale

        if log_scale:
            ds_cmap_min = np.log1p(ds_cmap_min)
            ds_cmap_max = np.log1p(ds_cmap_max)
            if ds_nan_sentinel is not None:
                # Sentinel must also be in log space for zmin to work
                ds_nan_sentinel = ds_cmap_min - (ds_cmap_max - ds_cmap_min) * 0.01

        # Extract bins for discrete legend mode
        ds_legend_bins = None
        if use_discrete_legend:
            ds_legend_bins = extract_legend_bins(
                scale_type=ds_scale_type,
                colors=ds_colors,
                data=data[ds_y].dropna().values,
                data_min=ds_cmap_min if not log_scale else float(data[ds_y].min()),
                data_max=ds_cmap_max if not log_scale else float(data[ds_y].max()),
                bins=ds_bins,
            )

        dataset_trace_counts: list = []

        for i, year in enumerate(unique_years):
            traces_before = len(fig.data)

            selected_year_data = data.loc[data[x].dt.year == year]
            selected_year_data = fill_empty_with_zeros(
                selected_year_data, x, year, start_month, end_month
            )
            if replace_nans_with_zeros:
                selected_year_data[ds_y] = selected_year_data[ds_y].fillna(0)

            year_calheatmap(
                selected_year_data,
                x,
                ds_y,
                name=ds_name,
                month_lines=month_lines,
                month_lines_width=month_lines_width,
                month_lines_color=month_lines_color,
                top_bottom_lines=top_bottom_lines,
                colorscale=ds_colorscale,
                year=year,
                fig=fig,
                dark_theme=dark_theme,
                gap=gap,
                title=title,
                row=0 if (use_navigation or use_dataset_swap) else i,
                total_height=total_height,
                text=None if text is None else selected_year_data[text].tolist(),
                text_name=text,
                years_as_columns=(
                    years_as_columns
                    if not (use_navigation or use_dataset_swap)
                    else False
                ),
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
                grouping=grouping,
                grouping_lines_width=grouping_lines_width,
                grouping_lines_color=grouping_lines_color,
                log_scale=log_scale,
                nan_sentinel=ds_nan_sentinel,
                annotations=annotations,
                annotations_fmt=annotations_fmt,
                annotations_font_size=annotations_font_size,
                annotations_font_color=annotations_font_color,
                annotations_font_family=annotations_font_family,
                legend_bins=ds_legend_bins,
                show_legend_items=(i == 0),
                week_start=week_start,
            )

            tc = len(fig.data) - traces_before
            dataset_trace_counts.append(tc)
            trace_structure.append(
                {
                    "dataset": dataset_label,
                    "year": year,
                    "start": traces_before,
                    "count": tc,
                }
            )

        # Apply colorscaling to this dataset's traces
        ds_start = trace_structure[-len(unique_years)]["start"]
        ds_end = trace_structure[-1]["start"] + trace_structure[-1]["count"]
        if not use_discrete_legend:
            for idx in range(ds_start, ds_end):
                trace = fig.data[idx]
                if hasattr(trace, "zmin"):
                    trace.zmin = (
                        ds_nan_sentinel if ds_nan_sentinel is not None else ds_cmap_min
                    )
                    trace.zmax = ds_cmap_max

        # Show colorbar for this dataset if configured (only in colorbar mode)
        ds_showscale = ds_cfg["showscale"]
        if ds_showscale and not use_discrete_legend:
            scale_title = ds_showscale if isinstance(ds_showscale, str) else ""
            tick_vals = (
                np.linspace(ds_cmap_min, ds_cmap_max, 5).tolist()
                if scale_ticks
                else None
            )
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
            if tick_vals:
                colorbar["tickvals"] = tick_vals
                if log_scale:
                    colorbar["ticktext"] = [f"{np.expm1(v):,.0f}" for v in tick_vals]
                    colorbar["tickmode"] = "array"
            # Apply user overrides
            if colorbar_options:
                if "title" in colorbar_options and isinstance(
                    colorbar_options["title"], str
                ):
                    colorbar["title"] = dict(text=colorbar_options["title"], side="top")
                else:
                    for k, v in colorbar_options.items():
                        colorbar[k] = v
            # Enable colorbar on first heatmap trace of each year so it
            # stays visible when switching years via navigation.
            ds_traces = [t for t in trace_structure if t["dataset"] == dataset_label]
            for t in ds_traces:
                for idx in range(t["start"], t["start"] + t["count"]):
                    trace = fig.data[idx]
                    if hasattr(trace, "zmin"):
                        trace.showscale = True
                        trace.colorbar = colorbar
                        break

        # Keep legacy trace_counts for single-dataset path
        if not use_dataset_swap:
            trace_counts = dataset_trace_counts

    # Apply discrete legend layout
    if use_discrete_legend:
        legend_cfg = dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
        )
        if legend_options:
            legend_cfg.update(legend_options)
        fig.update_layout(showlegend=True, legend=legend_cfg)

    # Multi-dataset: set initial visibility and build menus
    if use_dataset_swap:
        first_dataset = list(dataset_configs.keys())[0]
        first_year = unique_years[0]
        for t in trace_structure:
            is_visible = t["dataset"] == first_dataset and (
                t["year"] == first_year if use_navigation else True
            )
            for i in range(t["start"], t["start"] + t["count"]):
                fig.data[i].visible = is_visible

        menus = []
        menus.append(
            _add_dataset_navigation(
                fig,
                dataset_configs,
                trace_structure,
                unique_years,
                use_navigation,
                dataset_nav_options,
            )
        )
        if use_navigation:
            menus.append(
                _add_year_navigation(
                    fig,
                    unique_years,
                    trace_counts,
                    nav_options=nav_options,
                    trace_structure=trace_structure,
                    current_dataset=first_dataset,
                )
            )
        fig.update_layout(updatemenus=menus)
        _apply_toolbar_layout(fig)
    else:
        if use_navigation:
            fig = _add_year_navigation(
                fig, unique_years, trace_counts, nav_options=nav_options
            )
            _apply_toolbar_layout(fig)

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
    annotations: bool = False,
    annotations_fmt: Optional[str] = None,
    annotations_font_size: Optional[int] = None,
    annotations_font_color: Optional[str] = None,
    annotations_font_family: Optional[str] = None,
    colorbar_options: Optional[Dict[str, Any]] = None,
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

    colorbar_options : dict = None
        Override any Plotly colorbar property (orientation, x, y, thickness,
        len, tickformat, nticks, title, etc.).
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
        # Apply user overrides
        if colorbar_options:
            if "title" in colorbar_options and isinstance(
                colorbar_options["title"], str
            ):
                colorbar_config["title"] = dict(
                    text=colorbar_options["title"], side="top"
                )
            else:
                for k, v in colorbar_options.items():
                    colorbar_config[k] = v
    else:
        colorbar_config = None

    # annotations_fmt implies annotations=True
    if annotations_fmt is not None:
        annotations = True
    ann_texttemplate = None
    ann_textfont = None
    if annotations:
        ann_texttemplate = annotations_fmt if annotations_fmt else "%{z:.0f}"
        font = {}
        font["size"] = (
            annotations_font_size if annotations_font_size is not None else 10
        )
        if annotations_font_color is not None:
            font["color"] = annotations_font_color
        if annotations_font_family is not None:
            font["family"] = annotations_font_family
        ann_textfont = font

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
        texttemplate=ann_texttemplate,
        textfont=ann_textfont,
    )

    fig = go.Figure(data=cplt, layout=layout)

    return fig
