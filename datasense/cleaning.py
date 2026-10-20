"""Explicit, auditable data cleaning for DataSense AI (FR-03).

Every transformation is recorded in a human-readable action log so the user can
see exactly what changed. Nothing is applied silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

MissingStrategy = Literal["keep", "drop_rows", "drop_column", "fill_mean", "fill_median", "fill_mode", "fill_constant"]


@dataclass
class ColumnCleaningRule:
    """A cleaning rule targeting one column."""

    column: str
    strategy: MissingStrategy = "keep"
    constant: str | float | int | None = None
    convert_to: Literal["none", "numeric", "categorical", "datetime"] = "none"


@dataclass
class CleaningPlan:
    """User-selected cleaning configuration."""

    drop_duplicates: bool = False
    keep_columns: list[str] | None = None
    strip_whitespace: bool = True
    drop_fully_empty_columns: bool = False
    rules: list[ColumnCleaningRule] = field(default_factory=list)


@dataclass
class CleaningResult:
    """Cleaned data plus an explicit record of what was done."""

    dataframe: pd.DataFrame
    actions: list[str]
    before_shape: tuple[int, int]
    after_shape: tuple[int, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def rows_removed(self) -> int:
        return self.before_shape[0] - self.after_shape[0]

    @property
    def columns_removed(self) -> int:
        return self.before_shape[1] - self.after_shape[1]

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"no": range(1, len(self.actions) + 1), "tindakan": self.actions})

    def to_dict(self) -> dict:
        return {
            "before_shape": list(self.before_shape),
            "after_shape": list(self.after_shape),
            "rows_removed": self.rows_removed,
            "columns_removed": self.columns_removed,
            "actions": self.actions,
            "warnings": self.warnings,
        }


def _fill_value(series: pd.Series, strategy: str, constant):
    if strategy == "fill_mean":
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.dropna().empty:
            return None
        return float(numeric.mean())
    if strategy == "fill_median":
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.dropna().empty:
            return None
        return float(numeric.median())
    if strategy == "fill_mode":
        modes = series.dropna().mode()
        return modes.iloc[0] if not modes.empty else None
    if strategy == "fill_constant":
        return constant
    return None


def _convert(series: pd.Series, target: str) -> tuple[pd.Series, int]:
    """Convert a column, returning the series and the count of failed conversions."""
    if target == "numeric":
        converted = pd.to_numeric(series, errors="coerce")
    elif target == "datetime":
        converted = pd.to_datetime(series, errors="coerce")
    elif target == "categorical":
        converted = series.astype("category")
    else:
        return series, 0
    failures = int(converted.isna().sum() - series.isna().sum())
    return converted, max(0, failures)


def apply_cleaning_plan(df: pd.DataFrame, plan: CleaningPlan) -> CleaningResult:
    """Apply a cleaning plan and return the cleaned data with an action log."""
    if df is None or df.shape[1] == 0:
        raise ValueError("Dataframe kosong tidak dapat dibersihkan.")

    working = df.copy()
    actions: list[str] = []
    warnings: list[str] = []
    before_shape = (int(working.shape[0]), int(working.shape[1]))

    if plan.keep_columns is not None:
        unknown = [col for col in plan.keep_columns if col not in working.columns]
        if unknown:
            warnings.append("Kolom tidak ditemukan dan diabaikan: " + ", ".join(unknown))
        selected = [col for col in plan.keep_columns if col in working.columns]
        if not selected:
            raise ValueError("Tidak ada kolom valid yang dipilih untuk dipertahankan.")
        removed = [col for col in working.columns if col not in selected]
        if removed:
            working = working[selected]
            actions.append(
                f"Memilih {len(selected)} kolom; menghapus {len(removed)} kolom: "
                + ", ".join(map(str, removed))
            )
        else:
            actions.append(f"Mempertahankan seluruh {len(selected)} kolom yang dipilih.")

    if plan.drop_fully_empty_columns:
        empty_cols = [col for col in working.columns if working[col].isna().all()]
        if empty_cols:
            working = working.drop(columns=empty_cols)
            actions.append("Menghapus kolom yang seluruhnya kosong: " + ", ".join(map(str, empty_cols)))
        else:
            actions.append("Tidak ada kolom yang seluruhnya kosong.")

    if plan.strip_whitespace:
        stripped_count = 0
        for col in working.columns:
            if pd.api.types.is_object_dtype(working[col]) or pd.api.types.is_string_dtype(working[col]):
                original = working[col]
                working[col] = original.map(
                    lambda value: value.strip() if isinstance(value, str) else value
                )
                stripped_count += int((original.astype(str) != working[col].astype(str)).sum())
        if stripped_count:
            actions.append(f"Menghapus spasi berlebih pada {stripped_count} nilai teks.")

    for rule in plan.rules:
        if rule.column not in working.columns:
            warnings.append(f"Kolom '{rule.column}' tidak ditemukan; aturan dilewati.")
            continue

        if rule.convert_to != "none":
            before_nulls = int(working[rule.column].isna().sum())
            working[rule.column], failures = _convert(working[rule.column], rule.convert_to)
            after_nulls = int(working[rule.column].isna().sum())
            if failures:
                actions.append(
                    f"Mengubah tipe kolom '{rule.column}' menjadi {rule.convert_to}; "
                    f"{failures} nilai tidak dapat dikonversi menjadi kosong "
                    f"({before_nulls} -> {after_nulls} nilai kosong)."
                )
                warnings.append(
                    f"Konversi '{rule.column}' menghasilkan {failures} nilai kosong baru."
                )
            else:
                actions.append(f"Mengubah tipe kolom '{rule.column}' menjadi {rule.convert_to}.")

        strategy = rule.strategy
        if strategy in ("drop_rows", "drop_column"):
            if strategy == "drop_column":
                working = working.drop(columns=[rule.column])
                actions.append(f"Menghapus kolom '{rule.column}'.")
            else:
                before_rows = int(working.shape[0])
                working = working.dropna(subset=[rule.column])
                removed = before_rows - int(working.shape[0])
                actions.append(f"Menghapus {removed} baris dengan nilai kosong pada '{rule.column}'.")
            continue

        if strategy in ("fill_mean", "fill_median", "fill_mode", "fill_constant"):
            nulls = int(working[rule.column].isna().sum())
            if nulls == 0:
                actions.append(f"Kolom '{rule.column}' tidak memiliki nilai kosong; pengisian dilewati.")
                continue
            value = _fill_value(working[rule.column], strategy, rule.constant)
            if value is None:
                warnings.append(
                    f"Pengisian '{rule.column}' dengan {strategy} gagal: tidak ada nilai acuan."
                )
                actions.append(f"Pengisian kolom '{rule.column}' dilewati karena tidak ada nilai acuan.")
                continue
            working[rule.column] = working[rule.column].fillna(value)
            display = f"{value:.4g}" if isinstance(value, float) else repr(value)
            actions.append(f"Mengisi {nulls} nilai kosong pada '{rule.column}' dengan {display}.")

    if plan.drop_duplicates:
        before_rows = int(working.shape[0])
        working = working.drop_duplicates().reset_index(drop=True)
        removed = before_rows - int(working.shape[0])
        actions.append(f"Menghapus {removed} baris duplikat." if removed else "Tidak ada baris duplikat.")

    if not actions:
        actions.append("Tidak ada tindakan pembersihan yang dipilih; data tidak berubah.")

    working = working.reset_index(drop=True)
    after_shape = (int(working.shape[0]), int(working.shape[1]))

    if after_shape[0] == 0:
        raise ValueError(
            "Seluruh baris terhapus oleh aturan pembersihan. Longgarkan aturan dan coba lagi."
        )

    remaining_missing = int(working.isna().sum().sum())
    if remaining_missing:
        warnings.append(f"Masih terdapat {remaining_missing} sel kosong setelah pembersihan.")

    return CleaningResult(
        dataframe=working,
        actions=actions,
        before_shape=before_shape,
        after_shape=after_shape,
        warnings=warnings,
    )
