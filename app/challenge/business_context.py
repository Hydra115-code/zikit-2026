from __future__ import annotations

import numpy as np
import pandas as pd

from app.challenge.contracts import TARGET_TENANT


CONTEXT_SCHEMA_VERSION = "challenge-business-context-v1"


def _minutes(timestamp: pd.Series) -> pd.Series:
    return timestamp.dt.hour * 60 + timestamp.dt.minute


def _between(minutes: pd.Series, start: tuple[int, int], end: tuple[int, int]) -> pd.Series:
    lower = start[0] * 60 + start[1]
    upper = end[0] * 60 + end[1]
    return minutes.ge(lower) & minutes.lt(upper)


def _date_is(timestamp: pd.Series, value: str) -> pd.Series:
    return timestamp.dt.strftime("%Y-%m-%d").eq(value)


def context_features(tenant: str, timestamp: pd.Series) -> pd.DataFrame:
    """Encode only the business context published in Kaiku's contexto.md."""
    if tenant != TARGET_TENANT:
        raise ValueError(f"unsupported tenant: {tenant}")
    if not pd.api.types.is_datetime64_any_dtype(timestamp):
        raise TypeError("timestamp must use a pandas datetime dtype")

    minute = _minutes(timestamp)
    weekday = timestamp.dt.dayofweek
    weekend = weekday.ge(5)
    staffed = pd.Series(False, index=timestamp.index)
    critical = pd.Series(False, index=timestamp.index)
    backup = pd.Series(False, index=timestamp.index)
    campaign = pd.Series(False, index=timestamp.index)
    load_event = pd.Series(False, index=timestamp.index)
    maintenance = pd.Series(False, index=timestamp.index)
    reindex = pd.Series(False, index=timestamp.index)
    operational_peak = pd.Series(False, index=timestamp.index)

    staffed = weekday.le(5) & _between(minute, (9, 0), (19, 0))
    critical = pd.Series(True, index=timestamp.index)
    campaign = _date_is(timestamp, "2026-10-09") & _between(minute, (12, 0), (14, 0))
    reindex = weekday.eq(6) & _between(minute, (4, 0), (5, 30))
    backup = _between(minute, (3, 0), (3, 30))

    known_event = backup | campaign | load_event | maintenance | reindex | operational_peak
    minute_angle = 2 * np.pi * minute.astype(float) / (24 * 60)
    weekday_angle = 2 * np.pi * weekday.astype(float) / 7
    return pd.DataFrame(
        {
            "context_staffed": staffed.astype(bool),
            "context_critical": critical.astype(bool),
            "context_known_event": known_event.astype(bool),
            "context_backup": backup.astype(bool),
            "context_campaign": campaign.astype(bool),
            "context_load_event": load_event.astype(bool),
            "context_maintenance": maintenance.astype(bool),
            "context_reindex": reindex.astype(bool),
            "context_operational_peak": operational_peak.astype(bool),
            "context_weekend": weekend.astype(bool),
            "time_minute_sin": np.sin(minute_angle).astype("float32"),
            "time_minute_cos": np.cos(minute_angle).astype("float32"),
            "time_weekday_sin": np.sin(weekday_angle).astype("float32"),
            "time_weekday_cos": np.cos(weekday_angle).astype("float32"),
        },
        index=timestamp.index,
    )
