"""Dataset loading and validation for DataSense AI (FR-01)."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from common.config import ALLOWED_DATA_EXTENSIONS, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB
from common.utils import sha256_bytes


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset cannot be accepted."""


@dataclass
class LoadedDataset:
    """A dataframe plus provenance information about its source file."""

    name: str
    dataframe: pd.DataFrame
    size_bytes: int
    sha256: str
    extension: str
    sheet_name: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return int(self.dataframe.shape[0])

    @property
    def column_count(self) -> int:
        return int(self.dataframe.shape[1])


def _extension_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_upload(filename: str, payload: bytes) -> str:
    """Validate filename, extension, and size. Returns the normalized extension."""
    if not filename:
        raise DatasetValidationError("Nama file tidak ditemukan. Silakan unggah ulang file.")
    extension = _extension_of(filename)
    if extension not in ALLOWED_DATA_EXTENSIONS:
        allowed = ", ".join(ext.upper().lstrip(".") for ext in ALLOWED_DATA_EXTENSIONS)
        raise DatasetValidationError(
            f"Format file '{extension or 'tanpa ekstensi'}' tidak didukung. "
            f"Gunakan salah satu format berikut: {allowed}."
        )
    if not payload:
        raise DatasetValidationError("File kosong. Unggah file yang memiliki data.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise DatasetValidationError(
            f"Ukuran file {len(payload) / (1024 * 1024):.1f} MB melebihi batas "
            f"{MAX_UPLOAD_MB} MB. Pecah dataset atau gunakan file yang lebih kecil."
        )
    return extension


def _csv_header(payload: bytes, encoding: str) -> list[str]:
    text = payload.decode(encoding, errors="replace")
    reader = csv.reader(io.StringIO(text))
    try:
        return [cell.strip() for cell in next(reader)]
    except StopIteration:
        return []


def _check_duplicate_headers(header: list[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in header:
        if not name:
            continue
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise DatasetValidationError(
            "Nama kolom duplikat ditemukan: " + ", ".join(duplicates) + ". Perbaiki header dataset."
        )


def _read_csv(payload: bytes) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            _check_duplicate_headers(_csv_header(payload, encoding))
            frame = pd.read_csv(io.BytesIO(payload), encoding=encoding)
            if encoding != "utf-8-sig":
                warnings.append(f"File CSV dibaca menggunakan encoding {encoding}.")
            return frame, warnings
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise DatasetValidationError("File CSV tidak berisi data yang dapat dibaca.") from exc
        except pd.errors.ParserError as exc:
            raise DatasetValidationError(
                "Struktur CSV tidak valid. Periksa jumlah kolom pada setiap baris."
            ) from exc
    raise DatasetValidationError(
        "Encoding file CSV tidak dapat dikenali. Simpan ulang sebagai UTF-8."
    ) from last_error


def _read_excel(payload: bytes) -> tuple[pd.DataFrame, list[str], str | None]:
    try:
        workbook = pd.ExcelFile(io.BytesIO(payload), engine="openpyxl")
    except Exception as exc:  # openpyxl raises several unrelated types
        raise DatasetValidationError(
            "File Excel tidak dapat dibaca. Pastikan file tidak rusak dan berformat .xlsx."
        ) from exc
    if not workbook.sheet_names:
        raise DatasetValidationError("File Excel tidak memiliki worksheet.")
    sheet_name = workbook.sheet_names[0]
    frame = pd.read_excel(workbook, sheet_name=sheet_name)
    warnings: list[str] = []
    if len(workbook.sheet_names) > 1:
        warnings.append(
            f"File memiliki {len(workbook.sheet_names)} sheet; yang dibaca adalah '{sheet_name}'."
        )
    return frame, warnings, sheet_name


def read_dataframe(filename: str, payload: bytes) -> tuple[pd.DataFrame, str | None, list[str]]:
    """Read CSV/XLSX bytes into a dataframe, returning (frame, sheet, warnings)."""
    extension = validate_upload(filename, payload)
    if extension == ".csv":
        frame, warnings = _read_csv(payload)
        sheet = None
    else:
        frame, warnings, sheet = _read_excel(payload)

    if frame.empty:
        raise DatasetValidationError("Dataset tidak memiliki baris data.")
    if frame.shape[1] == 0:
        raise DatasetValidationError("Dataset tidak memiliki kolom yang dapat dibaca.")

    unnamed = [str(col) for col in frame.columns if str(col).startswith("Unnamed:")]
    if len(unnamed) == frame.shape[1]:
        raise DatasetValidationError(
            "Header kolom tidak terdeteksi. Pastikan baris pertama berisi nama kolom."
        )

    frame.columns = [str(col).strip() for col in frame.columns]
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        raise DatasetValidationError(
            "Nama kolom duplikat ditemukan: " + ", ".join(duplicated) + ". Perbaiki header dataset."
        )
    if any(not col for col in frame.columns):
        raise DatasetValidationError("Terdapat kolom tanpa nama. Perbaiki header dataset.")

    frame = frame.dropna(how="all").reset_index(drop=True)
    if frame.empty:
        raise DatasetValidationError("Dataset hanya berisi baris kosong.")
    return frame, sheet, warnings


def load_dataset(filename: str, payload: bytes) -> LoadedDataset:
    """Validate and load an uploaded dataset into memory."""
    frame, sheet, warnings = read_dataframe(filename, payload)
    return LoadedDataset(
        name=filename,
        dataframe=frame,
        size_bytes=len(payload),
        sha256=sha256_bytes(payload),
        extension=_extension_of(filename),
        sheet_name=sheet,
        warnings=warnings,
    )


def load_dataset_from_path(path: str | Path) -> LoadedDataset:
    """Convenience loader for sample datasets stored on disk."""
    path = Path(path)
    return load_dataset(path.name, path.read_bytes())


def stream_to_bytes(stream: BinaryIO) -> bytes:
    """Read a file-like object fully into bytes, restoring its position."""
    position = None
    try:
        position = stream.tell()
    except (AttributeError, OSError):
        position = None
    data = stream.read()
    if position is not None:
        try:
            stream.seek(position)
        except (AttributeError, OSError):
            pass
    return data
