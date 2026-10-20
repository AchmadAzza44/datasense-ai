"""Tests for dataset upload validation (FR-01)."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from common.config import MAX_UPLOAD_BYTES
from datasense.io import (
    DatasetValidationError,
    load_dataset,
    read_dataframe,
    validate_upload,
)


def test_validate_upload_accepts_csv_and_xlsx():
    assert validate_upload("data.csv", b"a,b\n1,2\n") == ".csv"
    assert validate_upload("data.XLSX", b"PK\x03\x04fake") == ".xlsx"


def test_validate_upload_rejects_unsupported_extension():
    with pytest.raises(DatasetValidationError, match="tidak didukung"):
        validate_upload("data.txt", b"hello")


def test_validate_upload_rejects_empty_file():
    with pytest.raises(DatasetValidationError, match="kosong"):
        validate_upload("data.csv", b"")


def test_validate_upload_rejects_oversized_file():
    with pytest.raises(DatasetValidationError, match="melebihi batas"):
        validate_upload("data.csv", b"a" * (MAX_UPLOAD_BYTES + 1))


def test_load_dataset_csv(sales_frame, csv_bytes):
    loaded = load_dataset("sales.csv", csv_bytes(sales_frame))
    assert loaded.row_count == len(sales_frame)
    assert loaded.column_count == sales_frame.shape[1]
    assert loaded.extension == ".csv"
    assert len(loaded.sha256) == 64
    assert loaded.sheet_name is None


def test_load_dataset_xlsx(sales_frame):
    buffer = io.BytesIO()
    sales_frame.to_excel(buffer, index=False, engine="openpyxl")
    loaded = load_dataset("sales.xlsx", buffer.getvalue())
    assert loaded.row_count == len(sales_frame)
    assert loaded.extension == ".xlsx"
    assert loaded.sheet_name is not None


def test_load_dataset_strips_column_whitespace(csv_bytes):
    frame = pd.DataFrame({" a ": [1, 2], "b": [3, 4]})
    loaded = load_dataset("dirty.csv", csv_bytes(frame))
    assert list(loaded.dataframe.columns) == ["a", "b"]


def test_load_dataset_rejects_duplicate_columns(csv_bytes):
    payload = b"a,a\n1,2\n"
    with pytest.raises(DatasetValidationError, match="duplikat"):
        load_dataset("dup.csv", payload)


def test_load_dataset_rejects_empty_dataframe():
    with pytest.raises(DatasetValidationError, match="baris"):
        load_dataset("empty.csv", b"a,b\n")


def test_load_dataset_rejects_unnamed_columns():
    payload = b",,\n1,2,3\n"
    with pytest.raises(DatasetValidationError, match="Header kolom"):
        load_dataset("unnamed.csv", payload)


def test_load_dataset_drops_fully_empty_rows(csv_bytes):
    frame = pd.DataFrame({"a": [1, None, 3], "b": [4, None, 6]})
    loaded = load_dataset("rows.csv", csv_bytes(frame))
    assert loaded.row_count == 2


def test_read_dataframe_returns_sheet_name_for_xlsx(sales_frame):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sales_frame.to_excel(writer, index=False, sheet_name="Data")
    frame, sheet, warnings = read_dataframe("book.xlsx", buffer.getvalue())
    assert sheet == "Data"
    assert frame.shape[0] == len(sales_frame)
    assert warnings == []
