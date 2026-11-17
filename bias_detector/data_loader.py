"""CSV loading helpers. Upload inputs remain byte buffers in the caller's session."""

from __future__ import annotations

import csv
from io import BytesIO, StringIO

import pandas as pd


def read_uploaded_csv(file_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Read CSV bytes without writing files; detect duplicate raw headers."""
    try:
        decoded = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("This CSV must use UTF-8 encoding. Save it as UTF-8 and upload it again.") from exc
    try:
        header = next(csv.reader(StringIO(decoded)))
    except StopIteration as exc:
        raise ValueError("This CSV is empty.") from exc
    stripped = [h.strip() for h in header]
    duplicates = sorted({h for h in stripped if stripped.count(h) > 1})
    if duplicates:
        return pd.DataFrame(), duplicates
    frame = pd.read_csv(BytesIO(file_bytes), low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame, []
