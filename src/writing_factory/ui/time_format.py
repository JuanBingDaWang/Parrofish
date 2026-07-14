"""Format UTC persistence timestamps for the mainland China desktop UI."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

_DISPLAY_TIMEZONE = timezone(timedelta(hours=8), name="UTC+08:00")


def format_china_datetime(value: object) -> str:
    """Convert an ISO UTC timestamp to an East-8 display string."""

    text = str(value or "").strip()
    if not text or len(text) <= 10:
        return text
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text[:19].replace("T", " ")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(_DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
