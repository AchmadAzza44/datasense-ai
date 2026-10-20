"""Tests for exploratory analysis helpers (FR-04)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datasense.eda import (
    categorical_distribution,
    correlation_matrix,
    group_summary,
    numeric_distribution,
    top_correlations,
    value_counts_table,
)


def test_numeric_distribution_histogram(sales_frame):
    distribution = numeric_distribution(sales_frame, "penjualan", bins=10)
    assert distribution["count"] == int(sales_frame["penjualan"].notna().sum())
    assert len(distribution["counts"]) == len(distribution["bins"]) - 1
    assert sum(distribution["counts"]) == distribution["count"]
    assert distribution["min"] <= distribution["median"] <= distribution["max"]


def test_numeric_distribution_unknown_column(sales_frame):
    with pytest.raises(KeyError):
        numeric_distribution(sales_frame, "tidak_ada")


def test_numeric_distribution_all_null():
    frame = pd.DataFrame({"a": [np.nan, np.nan]})
    with pytest.raises(ValueError):
        numeric_distribution(frame, "a")


def test_categorical_distribution(sales_frame):
    distribution = categorical_distribution(sales_frame, "promosi")
    assert distribution["unique_count"] == sales_frame["promosi"].nunique()
    assert sum(distribution["values"]) <= len(sales_frame)
    assert distribution["top"] in sales_frame["promosi"].dropna().unique()


def test_correlation_matrix(sales_frame):
    matrix = correlation_matrix(sales_frame)
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.loc["biaya_iklan", "biaya_iklan"] == pytest.approx(1.0)
    assert matrix.loc["biaya_iklan", "penjualan"] > 0.5


def test_correlation_matrix_requires_two_columns():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        correlation_matrix(frame)


def test_correlation_method_validation(sales_frame):
    with pytest.raises(ValueError):
        correlation_matrix(sales_frame, method="unknown")


def test_top_correlations_sorted(sales_frame):
    matrix = correlation_matrix(sales_frame)
    pairs = top_correlations(matrix, limit=2)
    assert len(pairs) == 2
    assert abs(pairs[0]["correlation"]) >= abs(pairs[1]["correlation"])


def test_value_counts_table(sales_frame):
    table = value_counts_table(sales_frame, "promosi")
    assert list(table.columns) == ["nilai", "jumlah", "persen"]
    assert table["persen"].sum() <= 100.0 + 1e-6


def test_group_summary(sales_frame):
    summary = group_summary(sales_frame, "promosi", "penjualan")
    assert "rata_rata" in summary.columns
    assert summary["jumlah"].sum() == sales_frame["penjualan"].notna().sum()


def test_group_summary_missing_column(sales_frame):
    with pytest.raises(KeyError):
        group_summary(sales_frame, "tidak_ada", "penjualan")
