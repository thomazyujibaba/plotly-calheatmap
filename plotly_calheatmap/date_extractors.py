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
    data: pd.DataFrame, x: str
) -> Tuple[Any, List[float], List[int]]:
    month_days = []
    for m in data[x].dt.month.unique():
        month_days.append(data.loc[data[x].dt.month == m, x].max().day)

    month_positions = np.linspace(1.5, 50, 12)
    weekdays_in_year = [i.weekday() for i in data[x]]

    # sometimes the last week of the current year conflicts with next year's january
    # pandas uses ISO weeks, which will give those weeks the number 52 or 53, but this
    # is bad news for this plot therefore we need a correction to use Gregorian weeks,
    # for a more in-depth explanation check
    # https://stackoverflow.com/questions/44372048/python-pandas-timestamp-week-returns-52-for-first-day-of-year
    weeknumber_of_dates = data[x].dt.strftime("%W").astype(int).tolist()

    return month_positions, weekdays_in_year, weeknumber_of_dates
