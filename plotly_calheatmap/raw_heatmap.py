from typing import List, Optional, Tuple

import re

import numpy as np
import pandas as pd
from plotly import graph_objects as go

from babel.numbers import format_decimal



def _format_value(value, fmt: str, locale: Optional[str] = None) -> str:
    """Format a numeric value using a Python format spec, with optional locale.

    When *locale* is provided (e.g. ``"pt_BR"``), the format spec is parsed to
    extract the decimal precision and grouping flag, and ``babel.numbers`` is
    used so that thousand/decimal separators follow locale conventions.

    Without a locale the standard Python ``format(value, fmt)`` is used.
    """
    if locale is None:
        return format(value, fmt)

    # Parse the format spec to extract grouping and precision
    use_grouping = "," in fmt
    # Extract decimal digits (default 2)
    prec_match = re.search(r"\.(\d+)", fmt)
    decimal_places = int(prec_match.group(1)) if prec_match else 2

    fmt_pattern = "#,##0." + "0" * decimal_places if use_grouping else "#0." + "0" * decimal_places
    return format_decimal(value, format=fmt_pattern, locale=locale)


def _resolve_hover_placeholders(
    template: str,
    extra_columns: Optional[List[str]] = None,
    log_scale: bool = False,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Replace friendly {name} placeholders with Plotly %{customdata[N]} refs.

    Supported placeholders:
        {date}            -> %{customdata[0]}
        {date:%d/%m/%Y}   -> %{customdata[0]}  (format applied to data)
        {name}            -> %{customdata[1]}
        {value}           -> %{z}  (or %{customdata[2]} when log_scale)
        {value:,.2f}      -> %{customdata[N]}  (format applied to data)
        {week}            -> %{x}
        {text}            -> %{text}
        {column_name}     -> %{customdata[N]}  (from extra_columns)

    Returns (resolved_template, date_fmt_or_None, value_fmt_or_None).
    """
    # When log_scale is active, customdata[2] holds the original value
    # and extra columns start at index 3 instead of 2.
    extra_offset = 3 if log_scale else 2

    mapping = {
        "date": "%{customdata[0]}",
        "name": "%{customdata[1]}",
        "value": "%{customdata[2]}" if log_scale else "%{z}",
        "week": "%{x}",
        "text": "%{text}",
    }
    if extra_columns:
        for i, col in enumerate(extra_columns):
            mapping[col] = f"%{{customdata[{i + extra_offset}]}}"

    date_fmt = None
    value_fmt = None

    def _replace(match):
        nonlocal date_fmt, value_fmt
        key = match.group(1)
        fmt = match.group(2)
        if key == "date" and fmt:
            date_fmt = fmt
        if key == "value" and fmt:
            value_fmt = fmt
            # When value has a format, it will be pre-formatted and stored
            # in customdata. The index is determined later in the caller.
            return "{__value_formatted__}"
        return mapping.get(key, match.group(0))

    resolved = re.sub(
        r"(?<!%)(?<!\{)\{(\w+)(?::([^}]+))?\}(?!\})", _replace, template
    )
    return resolved, date_fmt, value_fmt


def create_heatmap_without_formatting(
    data: pd.DataFrame,
    x: str,
    y: str,
    weeknumber_of_dates: List[int],
    weekdays_in_year: List[float],
    gap: int,
    year: int,
    colorscale: str,
    name: str,
    text: Optional[List[str]] = None,
    text_name: Optional[str] = None,
    hovertemplate: Optional[str] = None,
    extra_customdata_columns: Optional[List[str]] = None,
    vertical: bool = False,
    gap_positions: Optional[List[int]] = None,
    log_scale: bool = False,
    locale: Optional[str] = None,
    nan_sentinel: Optional[float] = None,
) -> List[go.Figure]:
    hovertemplate_extra = ""
    if text is not None:
        hovertemplate_extra = " <br>"
        if text_name is not None:
            hovertemplate_extra += f"{text_name}="
        hovertemplate_extra += "%{text}"

    # Resolve hover template and extract optional date format
    date_fmt = None
    value_fmt = None
    if hovertemplate is None:
        if log_scale:
            # Show original value from customdata instead of log-transformed z
            resolved_hovertemplate = (
                "%{customdata[0]} <br>Week=%{x} <br>%{customdata[1]}=%{customdata[2]}"
                + hovertemplate_extra
            )
        else:
            resolved_hovertemplate = (
                "%{customdata[0]} <br>Week=%{x} <br>%{customdata[1]}=%{z}"
                + hovertemplate_extra
            )
    else:
        resolved_hovertemplate, date_fmt, value_fmt = _resolve_hover_placeholders(
            hovertemplate, extra_customdata_columns, log_scale=log_scale
        )

    # Build customdata: always include date and name, then any extra columns
    if date_fmt:
        date_strings = data[x].dt.strftime(date_fmt).values
    else:
        date_strings = data[x].astype(str).values
    base_customdata = np.stack((date_strings, [name] * data.shape[0]), axis=-1)

    if log_scale:
        # Store original values so hover can display them
        original_values = data[y].astype(str).values
        base_customdata = np.column_stack([base_customdata, original_values])

    if extra_customdata_columns:
        extra_arrays = [data[col].astype(str).values for col in extra_customdata_columns]
        all_customdata = np.column_stack([base_customdata] + extra_arrays)
    else:
        all_customdata = base_customdata

    # If value has a format spec, pre-format values and append to customdata
    if value_fmt:
        formatted_values = []
        for v in data[y]:
            try:
                formatted_values.append(_format_value(v, value_fmt, locale=locale))
            except (ValueError, TypeError):
                formatted_values.append(str(v))
        formatted_arr = np.array(formatted_values)
        all_customdata = np.column_stack([all_customdata, formatted_arr])
        # Replace the placeholder with the correct customdata index
        val_idx = all_customdata.shape[1] - 1
        resolved_hovertemplate = resolved_hovertemplate.replace(
            "{__value_formatted__}", f"%{{customdata[{val_idx}]}}"
        )

    if vertical:
        hm_x = list(weekdays_in_year)
        hm_y = list(weeknumber_of_dates)
    else:
        hm_x = list(weeknumber_of_dates)
        hm_y = list(weekdays_in_year)

    z_values = data[y].tolist()
    if log_scale:
        z_values = [np.log1p(v) if v is not None and not np.isnan(v) else v for v in z_values]

    # Replace NaN with sentinel so Plotly renders them with nan_color
    if nan_sentinel is not None:
        z_values = [nan_sentinel if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in z_values]

    text_values = list(text) if text is not None else None
    customdata_list = all_customdata.tolist()

    # Fill gap positions with NaN so Plotly renders uniform cell sizes
    if gap_positions:
        n_custom_cols = len(customdata_list[0]) if customdata_list else 2
        for gp in gap_positions:
            for wd in range(7):
                if vertical:
                    hm_x.append(wd)
                    hm_y.append(gp)
                else:
                    hm_x.append(gp)
                    hm_y.append(wd)
                z_values.append(None)
                if text_values is not None:
                    text_values.append("")
                customdata_list.append([""] * n_custom_cols)

    raw_heatmap = [
        go.Heatmap(
            x=hm_x,
            y=hm_y,
            z=z_values,
            xgap=gap,
            ygap=gap,
            showscale=False,
            colorscale=colorscale,
            text=text_values,
            hovertemplate=resolved_hovertemplate,
            customdata=customdata_list,
            hoverongaps=False,
            name=str(year),
        )
    ]
    return raw_heatmap
