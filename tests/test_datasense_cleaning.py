"""Tests for explicit data cleaning (FR-03)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datasense.cleaning import (
    CleaningPlan,
    ColumnCleaningRule,
    apply_cleaning_plan,
)


def test_drop_duplicates_records_action():
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    result = apply_cleaning_plan(frame, CleaningPlan(drop_duplicates=True))
    assert result.after_shape[0] == 2
    assert any("duplikat" in action for action in result.actions)
    assert result.rows_removed == 1


def test_fill_median_for_numeric():
    frame = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
    result = apply_cleaning_plan(
        frame,
        CleaningPlan(rules=[ColumnCleaningRule("a", strategy="fill_median")]),
    )
    assert result.dataframe["a"].isna().sum() == 0
    assert result.dataframe["a"].iloc[2] == pytest.approx(2.0)


def test_fill_mode_for_categorical():
    frame = pd.DataFrame({"c": ["a", "a", "b", None]})
    result = apply_cleaning_plan(
        frame,
        CleaningPlan(rules=[ColumnCleaningRule("c", strategy="fill_mode")]),
    )
    assert result.dataframe["c"].iloc[3] == "a"


def test_fill_constant():
    frame = pd.DataFrame({"a": [1.0, np.nan]})
    result = apply_cleaning_plan(
        frame,
        CleaningPlan(rules=[ColumnCleaningRule("a", strategy="fill_constant", constant=0)]),
    )
    assert result.dataframe["a"].iloc[1] == 0


def test_drop_rows_for_missing():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    result = apply_cleaning_plan(
        frame,
        CleaningPlan(rules=[ColumnCleaningRule("a", strategy="drop_rows")]),
    )
    assert result.after_shape[0] == 2
    assert any("baris" in action for action in result.actions)


def test_drop_column():
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = apply_cleaning_plan(
        frame,
        CleaningPlan(rules=[ColumnCleaningRule("b", strategy="drop_column")]),
    )
    assert "b" not in result.dataframe.columns
    assert result.columns_removed == 1


def test_keep_columns_subset():
    frame = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = apply_cleaning_plan(frame, CleaningPlan(keep_columns=["a", "c"]))
    assert list(result.dataframe.columns) == ["a", "c"]


def test_keep_columns_rejects_unknown_only():
    frame = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        apply_cleaning_plan(frame, CleaningPlan(keep_columns=["zzz"]))


def test_convert_to_numeric_flags_failures():
    frame = pd.DataFrame({"a": ["1", "2", "abc"]})
    result = apply_cleaning_plan(
        frame,
        CleaningPlan(rules=[ColumnCleaningRule("a", convert_to="numeric")]),
    )
    assert result.dataframe["a"].dtype.kind in "fi"
    assert result.dataframe["a"].isna().sum() == 1
    assert any("tidak dapat dikonversi" in action for action in result.actions)


def test_convert_to_datetime():
    frame = pd.DataFrame({"when": ["2024-01-01", "invalid"]})
    result = apply_cleaning_plan(
        frame,
        CleaningPlan(rules=[ColumnCleaningRule("when", convert_to="datetime")]),
    )
    assert pd.api.types.is_datetime64_any_dtype(result.dataframe["when"])
    assert result.dataframe["when"].isna().sum() == 1


def test_strip_whitespace():
    frame = pd.DataFrame({"a": ["  x  ", "y"]})
    result = apply_cleaning_plan(frame, CleaningPlan(strip_whitespace=True))
    assert result.dataframe["a"].tolist() == ["x", "y"]
    assert any("spasi" in action for action in result.actions)


def test_drop_fully_empty_columns():
    frame = pd.DataFrame({"a": [1, 2], "empty": [np.nan, np.nan]})
    result = apply_cleaning_plan(frame, CleaningPlan(drop_fully_empty_columns=True))
    assert "empty" not in result.dataframe.columns


def test_cleaning_does_not_mutate_original():
    frame = pd.DataFrame({"a": [1.0, np.nan]})
    original = frame.copy(deep=True)
    apply_cleaning_plan(
        frame,
        CleaningPlan(rules=[ColumnCleaningRule("a", strategy="fill_mean")]),
    )
    pd.testing.assert_frame_equal(frame, original)


def test_no_actions_logged_when_plan_empty():
    frame = pd.DataFrame({"a": [1, 2]})
    result = apply_cleaning_plan(frame, CleaningPlan(strip_whitespace=False))
    assert result.actions == ["Tidak ada tindakan pembersihan yang dipilih; data tidak berubah."]


def test_all_rows_removed_raises():
    frame = pd.DataFrame({"a": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="Seluruh baris"):
        apply_cleaning_plan(
            frame,
            CleaningPlan(rules=[ColumnCleaningRule("a", strategy="drop_rows")]),
        )


def test_unknown_rule_column_is_skipped_with_warning():
    frame = pd.DataFrame({"a": [1, 2]})
    result = apply_cleaning_plan(
        frame,
        CleaningPlan(rules=[ColumnCleaningRule("missing", strategy="fill_mean")]),
    )
    assert any("tidak ditemukan" in warning for warning in result.warnings)


def test_summary_frame_and_to_dict():
    frame = pd.DataFrame({"a": [1, 1, 2]})
    result = apply_cleaning_plan(frame, CleaningPlan(drop_duplicates=True))
    summary = result.summary_frame()
    assert list(summary.columns) == ["no", "tindakan"]
    payload = result.to_dict()
    assert payload["before_shape"] == [3, 1]
    assert payload["after_shape"] == [2, 1]
