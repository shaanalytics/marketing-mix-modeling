"""Input validation helpers for MMM datasets."""
from __future__ import annotations
import numpy as np
import pandas as pd

def validate_mmm_data(data: pd.DataFrame, *, target: str, media_features: list[str], date_column: str | None = None) -> None:
    """Raise ValueError for common structural and numeric issues before modeling."""
    if data.empty:
        raise ValueError("data must contain at least one row.")
    required = [target, *media_features]
    if date_column:
        required.append(date_column)
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not media_features:
        raise ValueError("At least one media feature is required.")
    numeric = [target, *media_features]
    non_numeric = [column for column in numeric if not pd.api.types.is_numeric_dtype(data[column])]
    if non_numeric:
        raise ValueError(f"Expected numeric columns: {non_numeric}")
    if data[numeric].isna().any().any():
        raise ValueError("Target and media features must not contain missing values.")
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Target and media features must contain finite values.")
    if (data[media_features] < 0).any().any():
        raise ValueError("Media features must be non-negative.")
