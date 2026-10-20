"""Exploratory data analysis helpers for DataSense AI (FR-04)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def numeric_distribution(df: pd.DataFrame, column: str, bins: int = 30) -> dict:
    """Compute histogram bins and summary statistics for a numeric column."""
    if column not in df.columns:
        raise KeyError(f"Kolom '{column}' tidak ditemukan.")
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"Kolom '{column}' tidak memiliki nilai numerik yang valid.")
    bin_count = max(5, min(bins, max(5, int(np.sqrt(len(series)) * 2))))
    counts, edges = np.histogram(series.to_numpy(dtype="float64"), bins=bin_count)
    return {
        "column": column,
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()) if series.count() > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "bins": [float(edge) for edge in edges],
        "counts": [int(value) for value in counts],
        "skew": float(series.skew()) if series.count() > 2 else 0.0,
        "outliers": int(((series < series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25))) | (series > series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25)))).sum()),
    }


def categorical_distribution(df: pd.DataFrame, column: str, top_n: int = 15) -> dict:
    """Return value counts for a categorical column."""
    if column not in df.columns:
        raise KeyError(f"Kolom '{column}' tidak ditemukan.")
    series = df[column].dropna().astype(str)
    if series.empty:
        raise ValueError(f"Kolom '{column}' tidak memiliki nilai yang dapat dihitung.")
    counts = series.value_counts().head(top_n)
    return {
        "column": column,
        "unique_count": int(series.nunique()),
        "labels": [str(index) for index in counts.index],
        "values": [int(value) for value in counts.to_numpy()],
        "top": str(counts.index[0]),
        "top_count": int(counts.iloc[0]),
        "coverage_pct": round(float(counts.sum()) / len(series) * 100, 2),
    }


def correlation_matrix(
    df: pd.DataFrame, columns: Iterable[str] | None = None, method: str = "pearson"
) -> pd.DataFrame:
    """Compute a correlation matrix for numeric columns."""
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("Metode korelasi harus pearson, spearman, atau kendall.")
    if columns is None:
        numeric = df.select_dtypes(include=[np.number])
    else:
        selected = [col for col in columns if col in df.columns]
        if not selected:
            raise ValueError("Tidak ada kolom numerik valid untuk korelasi.")
        numeric = df[selected].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        raise ValueError("Korelasi membutuhkan minimal dua kolom numerik.")
    return numeric.corr(method=method, numeric_only=True)


def top_correlations(matrix: pd.DataFrame, limit: int = 5) -> list[dict]:
    """List the strongest absolute pairwise correlations."""
    if matrix is None or matrix.shape[0] < 2:
        return []
    pairs: list[dict] = []
    columns = list(matrix.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = matrix.loc[left, right]
            if pd.isna(value):
                continue
            pairs.append({"left": str(left), "right": str(right), "correlation": float(value)})
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return pairs[:limit]


def value_counts_table(df: pd.DataFrame, column: str, top_n: int = 20) -> pd.DataFrame:
    """Return a tidy frequency table for a column."""
    if column not in df.columns:
        raise KeyError(f"Kolom '{column}' tidak ditemukan.")
    all_counts = df[column].value_counts(dropna=False)
    counts = all_counts.head(top_n)
    total = int(all_counts.sum())
    return pd.DataFrame(
        {
            "nilai": [str(index) for index in counts.index],
            "jumlah": counts.to_numpy(dtype=int),
            "persen": [round(value / total * 100, 2) if total else 0.0 for value in counts.to_numpy()],
        }
    )


def group_summary(df: pd.DataFrame, group_column: str, value_column: str) -> pd.DataFrame:
    """Aggregate a numeric column by a categorical column."""
    for column in (group_column, value_column):
        if column not in df.columns:
            raise KeyError(f"Kolom '{column}' tidak ditemukan.")
    numeric = pd.to_numeric(df[value_column], errors="coerce")
    groups = df[group_column].astype(object).where(df[group_column].notna(), "(kosong)").astype(str)
    working = pd.DataFrame({group_column: groups, value_column: numeric})
    working = working.dropna(subset=[value_column])
    if working.empty:
        raise ValueError("Tidak ada data numerik valid untuk agregasi.")
    summary = (
        working.groupby(group_column)[value_column]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "jumlah",
                "mean": "rata_rata",
                "median": "median",
                "std": "standar_deviasi",
                "min": "minimum",
                "max": "maksimum",
            }
        )
    )
    return summary.sort_values("jumlah", ascending=False).reset_index(drop=True)
