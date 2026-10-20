"""Automatic, rule-based interpretation for DataSense AI (FR-06).

The summary is generated locally from statistics. No data leaves the machine and
no external generative AI is called, so privacy is preserved by construction.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from common.config import DISCLAIMER_ANALYTICS
from datasense.eda import top_correlations
from datasense.modeling import ModelResult
from datasense.profiling import ProfileReport


def _describe_quality(profile: ProfileReport) -> list[str]:
    notes: list[str] = []
    score = profile.quality_score
    if score >= 90:
        notes.append(f"Kualitas data tergolong baik (skor {score}/100).")
    elif score >= 70:
        notes.append(f"Kualitas data cukup (skor {score}/100); perbaikan minor disarankan.")
    else:
        notes.append(
            f"Kualitas data rendah (skor {score}/100); bersihkan nilai kosong dan duplikat "
            "sebelum menarik kesimpulan."
        )
    if profile.duplicate_rows:
        notes.append(f"Terdapat {profile.duplicate_rows} baris duplikat ({profile.duplicate_pct:.1f}%).")
    if profile.missing_cells:
        notes.append(
            f"Terdapat {profile.missing_cells} sel kosong ({profile.missing_pct:.1f}% dari total sel)."
        )
    worst = sorted(profile.columns, key=lambda col: col.null_pct, reverse=True)[:3]
    worst = [col for col in worst if col.null_pct > 0]
    if worst:
        detail = ", ".join(f"{col.name} ({col.null_pct:.1f}%)" for col in worst)
        notes.append(f"Kolom dengan nilai kosong terbanyak: {detail}.")
    return notes


def _describe_correlations(df: pd.DataFrame, profile: ProfileReport) -> list[str]:
    if len(profile.numeric_columns) < 2:
        return ["Korelasi tidak dapat dihitung karena kolom numerik kurang dari dua."]
    try:
        matrix = df[profile.numeric_columns].apply(pd.to_numeric, errors="coerce").corr()
    except Exception:
        return ["Korelasi tidak dapat dihitung pada dataset ini."]
    pairs = top_correlations(matrix, limit=3)
    if not pairs:
        return ["Tidak ditemukan pasangan kolom dengan korelasi yang dapat dihitung."]
    notes = []
    for pair in pairs:
        strength = "kuat" if abs(pair["correlation"]) >= 0.7 else "sedang" if abs(pair["correlation"]) >= 0.4 else "lemah"
        direction = "positif" if pair["correlation"] > 0 else "negatif"
        notes.append(
            f"Korelasi {strength} {direction} antara '{pair['left']}' dan '{pair['right']}' "
            f"(r = {pair['correlation']:.3f})."
        )
    notes.append(
        "Korelasi menunjukkan keterkaitan statistik, bukan bukti sebab-akibat."
    )
    return notes


def _describe_model(result: ModelResult) -> list[str]:
    notes: list[str] = []
    if result.task == "regression":
        r2 = result.metrics.get("r2")
        rmse = result.metrics.get("rmse")
        mae = result.metrics.get("mae")
        if r2 is not None:
            notes.append(f"Model regresi {result.model_name} mencapai R² = {r2:.3f} pada data uji.")
        if rmse is not None and mae is not None:
            notes.append(f"RMSE = {rmse:.4g} dan MAE = {mae:.4g} pada skala target '{result.target}'.")
    else:
        accuracy = result.metrics.get("accuracy")
        f1 = result.metrics.get("f1_weighted")
        if accuracy is not None:
            notes.append(f"Model klasifikasi {result.model_name} mencapai akurasi {accuracy:.3f} pada data uji.")
        if f1 is not None:
            notes.append(f"F1-score tertimbang = {f1:.3f}.")

    cv = result.cv_metrics
    if cv.get("mean") is not None:
        notes.append(
            f"Cross-validation ({cv.get('metric')}) rata-rata {cv['mean']:.3f} "
            f"dengan simpangan baku {cv.get('std', 0):.3f}."
        )
    if result.feature_importance:
        top = result.feature_importance[0]
        notes.append(
            f"Fitur paling berpengaruh menurut model: '{top['feature']}' "
            f"({top.get('importance_pct', 0):.1f}% dari total kepentingan)."
        )
    return notes


def build_interpretation(
    df: pd.DataFrame,
    profile: ProfileReport,
    cleaning_actions: list[str] | None = None,
    model_result: ModelResult | None = None,
) -> dict[str, Any]:
    """Build a structured interpretation report from statistics and model output."""
    cleaning_actions = cleaning_actions or []

    overview = [
        f"Dataset memiliki {profile.row_count} baris dan {profile.column_count} kolom.",
        f"Terdapat {len(profile.numeric_columns)} kolom numerik, "
        f"{len(profile.categorical_columns)} kolom kategorikal, "
        f"{len(profile.datetime_columns)} kolom tanggal/waktu, dan "
        f"{len(profile.text_columns)} kolom teks.",
    ]
    if profile.numeric_columns:
        examples = profile.numeric_columns[:3]
        for column in examples:
            stats = profile.column(column).stats if profile.column(column) else {}
            if stats:
                overview.append(
                    f"Kolom '{column}' memiliki rata-rata {stats['mean']:.4g}, "
                    f"median {stats['median']:.4g}, dan rentang {stats['min']:.4g} sampai {stats['max']:.4g}."
                )

    diagnostics = _describe_correlations(df, profile)

    recommendations: list[str] = []
    if profile.duplicate_rows:
        recommendations.append("Hapus atau tinjau baris duplikat sebelum analisis lanjutan.")
    if profile.missing_pct > 5:
        recommendations.append(
            "Tangani nilai kosong dengan strategi yang sesuai konteks (median untuk numerik, "
            "modus untuk kategorikal, atau hapus baris bila jumlahnya kecil)."
        )
    if profile.row_count < 100:
        recommendations.append(
            "Tambahkan data bila memungkinkan; ukuran sampel kecil membatasi keandalan model."
        )
    if not profile.numeric_columns:
        recommendations.append(
            "Konversi kolom yang seharusnya numerik agar analisis korelasi dan regresi tersedia."
        )
    if model_result is not None:
        if model_result.task == "regression" and (model_result.metrics.get("r2") or 0) < 0.3:
            recommendations.append(
                "R² rendah: tinjau kembali pemilihan fitur, tambahkan variabel relevan, "
                "atau gunakan model yang lebih sesuai."
            )
        if model_result.task == "classification" and (model_result.metrics.get("accuracy") or 0) < 0.7:
            recommendations.append(
                "Akurasi klasifikasi rendah: periksa keseimbangan kelas, kualitas label, "
                "dan relevansi fitur."
            )
        if model_result.warnings:
            recommendations.extend(model_result.warnings)
    if not recommendations:
        recommendations.append("Data siap dianalisis lebih lanjut; validasi temuan dengan data baru.")

    payload = {
        "overview": overview,
        "quality": _describe_quality(profile),
        "cleaning": cleaning_actions or ["Tidak ada tindakan pembersihan yang tercatat."],
        "diagnostics": diagnostics,
        "model": _describe_model(model_result) if model_result else [
            "Model belum dilatih. Pilih target dan jenis masalah untuk memulai pemodelan."
        ],
        "recommendations": recommendations,
        "limitations": (model_result.limitations if model_result else [])
        + [
            "Interpretasi dihasilkan otomatis dari statistik dataset dan harus diperiksa manusia.",
            "Dataset yang tidak representatif dapat menghasilkan kesimpulan yang menyesatkan.",
            "Analitik diagnostik tidak membuktikan hubungan sebab-akibat.",
        ],
        "disclaimer": DISCLAIMER_ANALYTICS,
    }
    return payload


def interpretation_to_markdown(interpretation: dict[str, Any]) -> str:
    """Render the interpretation as readable Markdown."""
    lines: list[str] = ["# Ringkasan Interpretasi DataSense AI", ""]
    sections = [
        ("Gambaran Umum", "overview"),
        ("Kualitas Data", "quality"),
        ("Tindakan Pembersihan", "cleaning"),
        ("Analitik Diagnostik", "diagnostics"),
        ("Hasil Model", "model"),
        ("Rekomendasi", "recommendations"),
        ("Keterbatasan", "limitations"),
    ]
    for title, key in sections:
        lines.append(f"## {title}")
        for item in interpretation.get(key, []):
            lines.append(f"- {item}")
        lines.append("")
    lines.append(f"> {interpretation.get('disclaimer', DISCLAIMER_ANALYTICS)}")
    lines.append("")
    return "\n".join(lines)
