from typing import List, Optional

from babel.dates import get_day_names, get_month_names


def get_localized_month_names(locale: Optional[str] = None) -> List[str]:
    if locale is None:
        locale = "en"
    names = get_month_names("wide", locale=locale)
    return [names[i] for i in range(1, 13)]


def get_localized_day_abbrs(locale: Optional[str] = None, week_start: str = "monday") -> List[str]:
    if locale is None:
        locale = "en"
    names = get_day_names("abbreviated", locale=locale)
    # names is indexed 0=Monday ... 6=Sunday
    all_days = [names[i] for i in range(7)]
    # Rotate so the chosen start day is first
    _week_start_offsets = {"monday": 0, "sunday": 1, "saturday": 2}
    offset = _week_start_offsets[week_start]
    if offset == 0:
        return all_days
    # Rotate: for sunday start (offset=1), order becomes [Sun, Mon, Tue, ..., Sat]
    return all_days[-offset:] + all_days[:-offset]
