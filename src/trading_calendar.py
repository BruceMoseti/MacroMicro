"""US equity-market trading calendar for the frozen research window.

The calendar is computed rather than downloaded so the frozen dataset can be rebuilt
without network access. Validation compares it against the dates actually present in
live source files.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from . import config


def _easter(year: int) -> dt.date:
    """Anonymous Gregorian algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    next_month = dt.date(year + (month == 12), (month % 12) + 1, 1)
    last = next_month - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(date: dt.date) -> dt.date | None:
    """NYSE observance: Saturday holidays roll back to Friday, Sunday to Monday."""
    if date.weekday() == 5:
        return date - dt.timedelta(days=1)
    if date.weekday() == 6:
        return date + dt.timedelta(days=1)
    return date


# Unscheduled full-day closures inside the research window.
AD_HOC_CLOSURES = (
    dt.date(2018, 12, 5),  # National day of mourning, George H. W. Bush
    dt.date(2025, 1, 9),  # National day of mourning, Jimmy Carter
)


def nyse_holidays(start_year: int, end_year: int) -> set[dt.date]:
    out: set[dt.date] = set()
    for year in range(start_year, end_year + 1):
        easter = _easter(year)
        candidates = [
            _observed(dt.date(year, 1, 1)),
            _nth_weekday(year, 1, 0, 3),  # MLK
            _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
            easter - dt.timedelta(days=2),  # Good Friday
            _last_weekday(year, 5, 0),  # Memorial Day
            _observed(dt.date(year, 7, 4)),
            _nth_weekday(year, 9, 0, 1),  # Labor Day
            _nth_weekday(year, 11, 3, 4),  # Thanksgiving
            _observed(dt.date(year, 12, 25)),
        ]
        if year >= 2022:
            candidates.append(_observed(dt.date(year, 6, 19)))  # Juneteenth
        out.update(c for c in candidates if c is not None)
    out.update(AD_HOC_CLOSURES)
    return out


def trading_days(
    start: pd.Timestamp = config.START_DATE, end: pd.Timestamp = config.END_DATE
) -> pd.DatetimeIndex:
    """Business days in [start, end] excluding NYSE holidays."""
    days = pd.bdate_range(start, end)
    holidays = nyse_holidays(start.year, end.year)
    keep = [d for d in days if d.date() not in holidays]
    return pd.DatetimeIndex(keep, name="date")


def cot_report_dates(
    start: pd.Timestamp = config.START_DATE, end: pd.Timestamp = config.END_DATE
) -> pd.DatetimeIndex:
    """Tuesday as-of dates for CFTC Commitments of Traders reports."""
    tuesdays = pd.date_range(start, end, freq="W-TUE")
    return pd.DatetimeIndex(tuesdays, name="report_date")
