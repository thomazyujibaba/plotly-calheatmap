from typing import List, Optional

from babel.dates import get_day_names, get_month_names


def get_localized_month_names(locale: Optional[str] = None) -> List[str]:
    if locale is None:
        locale = "en"
    names = get_month_names("wide", locale=locale)
    return [names[i] for i in range(1, 13)]


def get_localized_day_abbrs(locale: Optional[str] = None) -> List[str]:
    if locale is None:
        locale = "en"
    names = get_day_names("abbreviated", locale=locale)
    # names is indexed 0=Monday ... 6=Sunday
    return [names[i] for i in range(7)]
