from typing import List, Optional, Tuple

import re

import numpy as np
import pandas as pd
from plotly import graph_objects as go


def _resolve_hover_placeholders(
    template: str,
    extra_columns: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Replace friendly {name} placeholders with Plotly %{customdata[N]} refs.

    Supported placeholders:
        {date}            -> %{customdata[0]}
        {date:%d/%m/%Y}   -> %{customdata[0]}  (format applied to data)
        {name}            -> %{customdata[1]}
        {value}           -> %{z}
        {week}            -> %{x}
        {text}            -> %{text}
        {column_name}     -> %{customdata[2+i]}  (from extra_columns)

    Returns (resolved_template, date_fmt_or_None).
    """
    mapping = {
        "date": "%{customdata[0]}",
        "name": "%{customdata[1]}",
        "value": "%{z}",
        "week": "%{x}",
        "text": "%{text}",
    }
    if extra_columns:
        for i, col in enumerate(extra_columns):
            mapping[col] = f"%{{customdata[{i + 2}]}}"

    date_fmt = None

    def _replace(match):
        nonlocal date_fmt
        key = match.group(1)
        fmt = match.group(2)
        if key == "date" and fmt:
            date_fmt = fmt
        return mapping.get(key, match.group(0))

    resolved = re.sub(
        r"(?<!%)(?<!\{)\{(\w+)(?::([^}]+))?\}(?!\})", _replace, template
    )
    return resolved, date_fmt


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
) -> List[go.Figure]:
    hovertemplate_extra = ""
    if text is not None:
        hovertemplate_extra = " <br>"
        if text_name is not None:
            hovertemplate_extra += f"{text_name}="
        hovertemplate_extra += "%{text}"

    # Resolve hover template and extract optional date format
    date_fmt = None
    if hovertemplate is None:
        resolved_hovertemplate = (
            "%{customdata[0]} <br>Week=%{x} <br>%{customdata[1]}=%{z}"
            + hovertemplate_extra
        )
    else:
        resolved_hovertemplate, date_fmt = _resolve_hover_placeholders(
            hovertemplate, extra_customdata_columns
        )

    # Build customdata: always include date and name, then any extra columns
    if date_fmt:
        date_strings = data[x].dt.strftime(date_fmt).values
    else:
        date_strings = data[x].astype(str).values
    base_customdata = np.stack((date_strings, [name] * data.shape[0]), axis=-1)
    if extra_customdata_columns:
        extra_arrays = [data[col].astype(str).values for col in extra_customdata_columns]
        all_customdata = np.column_stack([base_customdata] + extra_arrays)
    else:
        all_customdata = base_customdata

    raw_heatmap = [
        go.Heatmap(
            x=weeknumber_of_dates,
            y=weekdays_in_year,
            z=data[y],
            xgap=gap,
            ygap=gap,
            showscale=False,
            colorscale=colorscale,
            text=text,
            hovertemplate=resolved_hovertemplate,
            customdata=all_customdata,
            name=str(year),
        )
    ]
    return raw_heatmap
