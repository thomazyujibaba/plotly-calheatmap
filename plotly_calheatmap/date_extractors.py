from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from plotly_calheatmap.i18n import get_localized_month_names


def get_month_names(
    data: pd.DataFrame,
    x: str,
    start_month: int = 1,
    end_month: int = 12,
    locale: Optional[str] = None,
) -> List[str]:
    all_month_names = get_localized_month_names(locale)
    present_months = sorted(data[x].dt.month.unique())
    names = [all_month_names[m - 1] for m in present_months]
    start_month_names_filler = [None] * (start_month - 1)
    end_month_names_filler = [None] * (12 - end_month)
    month_names = list(start_month_names_filler + names + end_month_names_filler)
    return month_names


def get_date_coordinates(
    data: pd.DataFrame, x: str, month_gap: int = 0, week_start: str = "monday",
) -> Tuple[Any, List[float], List[int], List[int]]:
    month_days = []
    for m in data[x].dt.month.unique():
        month_days.append(data.loc[data[x].dt.month == m, x].max().day)

    # week_start offset: shift weekday values so the chosen start day becomes 0
    _week_start_offsets = {"monday": 0, "sunday": 1, "saturday": 2}
    offset = _week_start_offsets[week_start]

    weekdays_in_year = [(i.weekday() + offset) % 7 for i in data[x]]

    # sometimes the last week of the current year conflicts with next year's january
    # pandas uses ISO weeks, which will give those weeks the number 52 or 53, but this
    # is bad news for this plot therefore we need a correction to use Gregorian weeks,
    # for a more in-depth explanation check
    # https://stackoverflow.com/questions/44372048/python-pandas-timestamp-week-returns-52-for-first-day-of-year
    if offset == 0:
        weeknumber_of_dates = data[x].dt.strftime("%W").astype(int).tolist()
    else:
        # Compute week numbers without shifting dates across year boundaries.
        # Mirrors strftime %W/%U logic but for any start day.
        _week_start_weekdays = {"sunday": 6, "saturday": 5}
        ws = _week_start_weekdays[week_start]
        yday = data[x].dt.dayofyear                    # 1-based
        wday = data[x].dt.weekday                       # 0=Mon...6=Sun
        days_since_start = (wday - ws) % 7
        weeknumber_of_dates = ((yday - 1 - days_since_start + 7) // 7).tolist()

    gap_positions: List[int] = []

    if month_gap > 0:
        months = data[x].dt.month.values
        sorted_unique_months = sorted(data[x].dt.month.unique())
        month_to_index = {m: i for i, m in enumerate(sorted_unique_months)}
        weeknumber_of_dates = [
            wk + month_gap * month_to_index[m]
            for wk, m in zip(weeknumber_of_dates, months)
        ]

        # Compute gap positions between each pair of consecutive months
        for i in range(len(sorted_unique_months) - 1):
            m_curr = sorted_unique_months[i]
            m_next = sorted_unique_months[i + 1]
            curr_max = max(wk for wk, mo in zip(weeknumber_of_dates, months) if mo == m_curr)
            next_min = min(wk for wk, mo in zip(weeknumber_of_dates, months) if mo == m_next)
            for pos in range(curr_max + 1, next_min):
                gap_positions.append(pos)

        # Build 12-element month_positions matching month_names structure
        month_positions_map = {}
        for m in sorted_unique_months:
            mask = months == m
            wks = [wk for wk, is_m in zip(weeknumber_of_dates, mask) if is_m]
            month_positions_map[m] = (min(wks) + max(wks)) / 2

        month_positions = [month_positions_map.get(m, None) for m in range(1, 13)]
    else:
        month_positions = np.linspace(1.5, 50, 12)

    return month_positions, weekdays_in_year, weeknumber_of_dates, gap_positions


GROUPINGS = {
    "bimester": {
        "boundaries": [1, 3, 5, 7, 9, 11],
        "labels": ["B1", "B2", "B3", "B4", "B5", "B6"],
    },
    "quarter": {
        "boundaries": [1, 4, 7, 10],
        "labels": ["Q1", "Q2", "Q3", "Q4"],
    },
    "semester": {
        "boundaries": [1, 7],
        "labels": ["S1", "S2"],
    },
}


def get_group_names_and_positions(
    data: pd.DataFrame,
    x: str,
    grouping: str,
    weeknumber_of_dates: List[int],
) -> Tuple[List[str], List[float]]:
    """Compute tick labels and positions for a given grouping.

    Returns (group_names, group_positions) lists suitable for axis ticks.
    """
    cfg = GROUPINGS[grouping]
    boundaries = cfg["boundaries"]
    labels = cfg["labels"]

    months = data[x].dt.month.values

    group_names: List[str] = []
    group_positions: List[float] = []

    for i, boundary_month in enumerate(boundaries):
        # Determine which months belong to this group
        if i + 1 < len(boundaries):
            next_boundary = boundaries[i + 1]
        else:
            next_boundary = 13  # past December

        group_months = list(range(boundary_month, next_boundary))
        # Find week numbers for dates in this group
        wks = [
            wk for wk, m in zip(weeknumber_of_dates, months)
            if m in group_months
        ]
        if wks:
            group_names.append(labels[i])
            group_positions.append((min(wks) + max(wks)) / 2)

    return group_names, group_positions
