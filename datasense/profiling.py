"""Data profiling and quality checks for DataSense AI (FR-02)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """Profile information for a single column."""

    name: str
    dtype: str
    kind: str
    non_null: int
    null_count: int
    null_pct: float
    unique_count: int
    duplicate_count: int
    sample_values: list[Any]
    stats: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "kind": self.kind,
            "non_null": self.non_null,
            "null_count": self.null_count,
            "null_pct": round(self.null_pct, 2),
            "unique_count": self.unique_count,
            "duplicate_count": self.duplicate_count,
            "sample_values": self.sample_values,
            "stats": self.stats,
            "issues": self.issues,
        }


@dataclass
class ProfileReport:
    """Complete profiling result for a dataset."""

    row_count: int
    column_count: int
    memory_bytes: int
    duplicate_rows: int
    duplicate_pct: float
    total_cells: int
    missing_cells: int
    missing_pct: float
    columns: list[ColumnProfile]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    boolean_columns: list[str]
    text_columns: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """A 0-100 heuristic score combining completeness and uniqueness."""
        if self.total_cells == 0:
            return 0.0
        completeness = 1.0 - (self.missing_cells / self.total_cells)
        uniqueness = 1.0 - (self.duplicate_rows / self.row_count if self.row_count else 0.0)
        score = (0.7 * completeness + 0.3 * uniqueness) * 100
        return round(max(0.0, min(100.0, score)), 1)

    def column(self, name: str) -> ColumnProfile | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_bytes": self.memory_bytes,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_pct": round(self.duplicate_pct, 2),
            "total_cells": self.total_cells,
            "missing_cells": self.missing_cells,
            "missing_pct": round(self.missing_pct, 2),
            "quality_score": self.quality_score,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "boolean_columns": self.boolean_columns,
            "text_columns": self.text_columns,
            "warnings": self.warnings,
            "columns": [column.to_dict() for column in self.columns],
        }


def _kind_of(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    non_null = series.dropna()
    if non_null.empty:
        return "empty"
    unique_ratio = non_null.nunique() / len(non_null)
    if non_null.nunique() <= max(20, int(len(non_null) * 0.2)) or unique_ratio < 0.05:
        return "categorical"
    return "text"


def _numeric_stats(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {}
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std()) if clean.count() > 1 else 0.0,
        "min": float(clean.min()),
        "q1": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q3": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "skew": float(clean.skew()) if clean.count() > 2 else 0.0,
        "zero_count": int((clean == 0).sum()),
        "negative_count": int((clean < 0).sum()),
    }


def _categorical_stats(series: pd.Series) -> dict[str, Any]:
    clean = series.dropna().astype(str)
    if clean.empty:
        return {}
    counts = clean.value_counts()
    return {
        "top": str(counts.index[0]),
        "top_count": int(counts.iloc[0]),
        "top_pct": round(float(counts.iloc[0]) / len(clean) * 100, 2),
        "values": {str(key): int(value) for key, value in counts.head(10).items()},
    }


def _datetime_stats(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_datetime(series, errors="coerce").dropna()
    if clean.empty:
        return {}
    return {
        "min": clean.min().isoformat(),
        "max": clean.max().isoformat(),
        "span_days": int((clean.max() - clean.min()).days),
    }


def _detect_column_issues(series: pd.Series, kind: str, name: str) -> list[str]:
    issues: list[str] = []
    null_ratio = series.isna().mean()
    if null_ratio >= 0.5:
        issues.append(f"Lebih dari 50% nilai kosong ({null_ratio * 100:.1f}%).")
    elif null_ratio > 0:
        issues.append(f"Memiliki {int(series.isna().sum())} nilai kosong.")

    if kind == "text":
        issues.append("Teks dengan kardinalitas tinggi; pertimbangkan encoding atau pengabaian.")

    if kind == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        infinite = int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
        if infinite:
            issues.append(f"Memiliki {infinite} nilai tak hingga (inf).")

    if kind == "categorical":
        counts = series.dropna().astype(str).value_counts()
        if len(counts) > 1 and counts.iloc[-1] == 1:
            issues.append("Terdapat kategori yang hanya muncul satu kali (rare category).")

    if name.strip() != name:
        issues.append("Nama kolom memiliki spasi di awal/akhir.")

    return issues


def profile_dataframe(df: pd.DataFrame) -> ProfileReport:
    """Compute a profiling report including data quality issues."""
    if df is None or df.shape[1] == 0:
        raise ValueError("Dataframe kosong tidak dapat diprofilkan.")

    duplicate_rows = int(df.duplicated().sum())
    row_count, column_count = int(df.shape[0]), int(df.shape[1])
    total_cells = row_count * column_count
    missing_cells = int(df.isna().sum().sum())

    columns: list[ColumnProfile] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []
    boolean_columns: list[str] = []
    text_columns: list[str] = []

    for name in df.columns:
        series = df[name]
        kind = _kind_of(series)
        if kind == "numeric":
            numeric_columns.append(name)
            stats = _numeric_stats(series)
        elif kind == "categorical":
            categorical_columns.append(name)
            stats = _categorical_stats(series)
        elif kind == "datetime":
            datetime_columns.append(name)
            stats = _datetime_stats(series)
        elif kind == "boolean":
            boolean_columns.append(name)
            stats = _categorical_stats(series)
        elif kind == "text":
            text_columns.append(name)
            stats = _categorical_stats(series)
        else:
            stats = {}

        samples = series.dropna().head(5).tolist()
        samples = [value.item() if hasattr(value, "item") else value for value in samples]
        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                kind=kind,
                non_null=int(series.notna().sum()),
                null_count=int(series.isna().sum()),
                null_pct=float(series.isna().mean() * 100),
                unique_count=int(series.nunique(dropna=True)),
                duplicate_count=int(row_count - series.nunique(dropna=True)),
                sample_values=[str(value) for value in samples],
                stats=stats,
                issues=_detect_column_issues(series, kind, str(name)),
            )
        )

    warnings: list[str] = []
    if duplicate_rows:
        warnings.append(f"Ditemukan {duplicate_rows} baris duplikat.")
    if missing_cells:
        warnings.append(
            f"Ditemukan {missing_cells} sel kosong dari {total_cells} sel "
            f"({missing_cells / total_cells * 100:.1f}%)."
        )
    if not numeric_columns:
        warnings.append("Tidak ada kolom numerik; analisis korelasi dan regresi tidak tersedia.")
    if row_count < 30:
        warnings.append("Jumlah baris sedikit (<30); hasil model berpotensi tidak stabil.")

    return ProfileReport(
        row_count=row_count,
        column_count=column_count,
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        duplicate_rows=duplicate_rows,
        duplicate_pct=(duplicate_rows / row_count * 100) if row_count else 0.0,
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_pct=(missing_cells / total_cells * 100) if total_cells else 0.0,
        columns=columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        boolean_columns=boolean_columns,
        text_columns=text_columns,
        warnings=warnings,
    )
