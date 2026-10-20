"""Tests for profiling and data quality checks (FR-02)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datasense.profiling import profile_dataframe


def test_profile_counts_and_types(sales_frame):
    profile = profile_dataframe(sales_frame)
    assert profile.row_count == len(sales_frame)
    assert profile.column_count == sales_frame.shape[1]
    assert set(profile.numeric_columns) == {"biaya_iklan", "harga", "penjualan"}
    assert profile.categorical_columns == ["promosi"]
    assert profile.missing_cells == int(sales_frame.isna().sum().sum())


def test_profile_detects_duplicates():
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    profile = profile_dataframe(frame)
    assert profile.duplicate_rows == 1
    assert profile.duplicate_pct == pytest.approx(33.333, rel=1e-3)


def test_profile_numeric_stats():
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    profile = profile_dataframe(frame)
    column = profile.column("value")
    assert column is not None
    assert column.stats["mean"] == pytest.approx(3.0)
    assert column.stats["median"] == pytest.approx(3.0)
    assert column.stats["min"] == 1.0
    assert column.stats["max"] == 5.0


def test_profile_categorical_stats():
    frame = pd.DataFrame({"category": ["a", "a", "b", "c"]})
    profile = profile_dataframe(frame)
    column = profile.column("category")
    assert column.kind == "categorical"
    assert column.stats["top"] == "a"
    assert column.stats["top_count"] == 2


def test_profile_detects_datetime():
    frame = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])})
    profile = profile_dataframe(frame)
    assert profile.datetime_columns == ["when"]
    column = profile.column("when")
    assert column.stats["span_days"] == 60


def test_quality_score_bounds():
    clean = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    profile = profile_dataframe(clean)
    assert 0 <= profile.quality_score <= 100
    assert profile.quality_score == pytest.approx(100.0)


def test_quality_score_drops_with_missing_data():
    dirty = pd.DataFrame({"a": [1, None, None, 4], "b": [None, None, 3, 4]})
    profile = profile_dataframe(dirty)
    assert profile.quality_score < 100


def test_profile_warns_about_small_dataset():
    frame = pd.DataFrame({"a": range(5)})
    profile = profile_dataframe(frame)
    assert any("sedikit" in warning for warning in profile.warnings)


def test_profile_flags_high_null_columns():
    frame = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan, np.nan], "b": [1, 2, 3, 4, 5]})
    profile = profile_dataframe(frame)
    column = profile.column("a")
    assert any("nilai kosong" in issue for issue in column.issues)


def test_profile_rejects_empty_frame():
    with pytest.raises(ValueError):
        profile_dataframe(pd.DataFrame())


def test_profile_to_dict_is_serializable(sales_frame):
    import json

    profile = profile_dataframe(sales_frame)
    payload = json.dumps(profile.to_dict(), allow_nan=False)
    assert "quality_score" in payload
