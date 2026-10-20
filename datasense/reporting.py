"""Downloadable PDF analysis report for DataSense AI (FR-07)."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from common.config import DISCLAIMER_ANALYTICS, DATASENSE_NAME
from datasense.eda import correlation_matrix, numeric_distribution, top_correlations
from datasense.interpretation import build_interpretation
from datasense.modeling import ModelResult
from datasense.profiling import ProfileReport


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontSize=20,
            textColor=colors.HexColor("#123B5B"),
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#5A6B7B"),
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1Custom",
            parent=base["Heading1"],
            fontSize=14,
            textColor=colors.HexColor("#123B5B"),
            spaceBefore=14,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2Custom",
            parent=base["Heading2"],
            fontSize=11,
            textColor=colors.HexColor("#1F6F8B"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle("BodyCustom", parent=base["BodyText"], fontSize=9, leading=13),
        "small": ParagraphStyle(
            "SmallCustom",
            parent=base["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#4A5568"),
        ),
        "warning": ParagraphStyle(
            "WarningCustom",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#8A4B08"),
            backColor=colors.HexColor("#FFF4E5"),
            borderPadding=6,
        ),
    }


def _table(data: list[list[Any]], widths: list[float] | None = None) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F6F8B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _figure_to_image(figure: plt.Figure, width_cm: float = 16.0) -> Image:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return Image(buffer, width=width_cm * cm, height=width_cm * cm * 0.5)


def _distribution_chart(df: pd.DataFrame, column: str) -> Image | None:
    try:
        distribution = numeric_distribution(df, column)
    except (KeyError, ValueError):
        return None
    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    edges = distribution["bins"]
    centers = [(edges[index] + edges[index + 1]) / 2 for index in range(len(edges) - 1)]
    width = (edges[-1] - edges[0]) / max(1, len(centers)) if edges[-1] > edges[0] else 0.8
    axis.bar(centers, distribution["counts"], width=width * 0.92, color="#2C7DA0", edgecolor="white")
    axis.set_title(f"Distribusi {column}", fontsize=10)
    axis.set_xlabel(column, fontsize=8)
    axis.set_ylabel("Frekuensi", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _figure_to_image(figure)


def _correlation_chart(df: pd.DataFrame, columns: list[str]) -> Image | None:
    if len(columns) < 2:
        return None
    try:
        matrix = correlation_matrix(df, columns[:10])
    except ValueError:
        return None
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    image = axis.imshow(matrix.to_numpy(dtype="float64"), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [str(col)[:16] for col in matrix.columns]
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=6)
    axis.set_title("Matriks Korelasi", fontsize=10)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return _figure_to_image(figure, width_cm=14.0)


def _predictions_chart(model: ModelResult) -> Image | None:
    frame = model.predictions
    if frame.empty:
        return None
    figure, axis = plt.subplots(figsize=(6.6, 3.6))
    if model.task == "regression":
        axis.scatter(frame["actual"], frame["predicted"], s=14, alpha=0.65, color="#1F6F8B")
        low = float(np.nanmin([frame["actual"].min(), frame["predicted"].min()]))
        high = float(np.nanmax([frame["actual"].max(), frame["predicted"].max()]))
        axis.plot([low, high], [low, high], linestyle="--", color="#C0392B", linewidth=1)
        axis.set_xlabel("Nilai aktual", fontsize=8)
        axis.set_ylabel("Nilai prediksi", fontsize=8)
        axis.set_title("Aktual vs Prediksi", fontsize=10)
    else:
        counts = frame["actual"].astype(str).value_counts().head(10)
        indices = np.arange(len(counts))
        axis.bar(indices - 0.2, counts.to_numpy(), width=0.4, label="Aktual", color="#2C7DA0")
        predicted_counts = frame["predicted"].astype(str).value_counts().reindex(counts.index, fill_value=0)
        axis.bar(indices + 0.2, predicted_counts.to_numpy(), width=0.4, label="Prediksi", color="#F4A261")
        axis.set_xticks(indices)
        axis.set_xticklabels([label[:14] for label in counts.index], rotation=30, ha="right", fontsize=6)
        axis.set_title("Perbandingan kelas aktual dan prediksi", fontsize=10)
        axis.legend(fontsize=7)
    axis.tick_params(labelsize=7)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _figure_to_image(figure)


def _add_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6B7C8F"))
    canvas.drawString(2 * cm, 1.2 * cm, f"{DATASENSE_NAME} - laporan analisis")
    canvas.drawRightString(
        A4[0] - 2 * cm, 1.2 * cm, f"Halaman {document.page}"
    )
    canvas.restoreState()


def build_report_pdf(
    df: pd.DataFrame,
    profile: ProfileReport,
    dataset_name: str,
    dataset_sha256: str,
    cleaning_actions: list[str] | None = None,
    model_result: ModelResult | None = None,
    interpretation: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Build a complete PDF report and return it as bytes."""
    styles = _styles()
    generated_at = generated_at or datetime.now()
    interpretation = interpretation or build_interpretation(
        df, profile, cleaning_actions, model_result
    )

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title=f"{DATASENSE_NAME} Report - {dataset_name}",
        author=DATASENSE_NAME,
    )

    story: list[Any] = [
        Paragraph("Laporan Analisis Data", styles["title"]),
        Paragraph(
            f"{dataset_name} &nbsp;|&nbsp; dibuat {generated_at.strftime('%d %B %Y %H:%M')} "
            f"&nbsp;|&nbsp; SHA-256: {dataset_sha256[:16]}...",
            styles["subtitle"],
        ),
        Paragraph(interpretation.get("disclaimer", DISCLAIMER_ANALYTICS), styles["warning"]),
        Spacer(1, 10),
        Paragraph("1. Informasi Dataset", styles["h1"]),
        _table(
            [
                ["Properti", "Nilai"],
                ["Nama file", dataset_name],
                ["Jumlah baris", f"{profile.row_count:,}"],
                ["Jumlah kolom", f"{profile.column_count:,}"],
                ["Sel kosong", f"{profile.missing_cells:,} ({profile.missing_pct:.1f}%)"],
                ["Baris duplikat", f"{profile.duplicate_rows:,} ({profile.duplicate_pct:.1f}%)"],
                ["Skor kualitas (heuristik)", f"{profile.quality_score}/100"],
                ["Ukuran memori", f"{profile.memory_bytes / 1024 / 1024:.2f} MB"],
                ["Hash dataset", dataset_sha256],
            ],
            widths=[5 * cm, 11.5 * cm],
        ),
        Paragraph("2. Struktur dan Statistik Kolom", styles["h1"]),
    ]

    column_rows: list[list[Any]] = [["Kolom", "Tipe", "Jenis", "Kosong", "Unik", "Ringkasan"]]
    for column in profile.columns[:40]:
        stats = column.stats or {}
        if column.kind == "numeric" and stats:
            summary = (
                f"mean {stats['mean']:.3g}, median {stats['median']:.3g}, "
                f"min {stats['min']:.3g}, max {stats['max']:.3g}"
            )
        elif stats and "top" in stats:
            summary = f"top '{stats['top']}' ({stats['top_pct']:.1f}%)"
        elif column.kind == "datetime" and stats:
            summary = f"{stats['min'][:10]} s.d. {stats['max'][:10]}"
        else:
            summary = "-"
        column_rows.append(
            [
                Paragraph(str(column.name)[:32], styles["small"]),
                str(column.dtype),
                column.kind,
                f"{column.null_count}",
                f"{column.unique_count}",
                Paragraph(summary, styles["small"]),
            ]
        )
    story.append(_table(column_rows, widths=[3.8 * cm, 2.2 * cm, 2.2 * cm, 1.6 * cm, 1.6 * cm, 5.1 * cm]))

    if profile.warnings:
        story.append(Paragraph("Peringatan Kualitas Data", styles["h2"]))
        for warning in profile.warnings:
            story.append(Paragraph(f"- {warning}", styles["body"]))

    story.append(Paragraph("3. Tindakan Pembersihan", styles["h1"]))
    actions = cleaning_actions or ["Tidak ada tindakan pembersihan yang diterapkan."]
    for index, action in enumerate(actions, start=1):
        story.append(Paragraph(f"{index}. {action}", styles["body"]))

    story.append(Paragraph("4. Analisis dan Visualisasi", styles["h1"]))
    chart_count = 0
    for column in profile.numeric_columns[:2]:
        chart = _distribution_chart(df, column)
        if chart is not None:
            story.append(KeepTogether([Paragraph(f"Distribusi {column}", styles["h2"]), chart]))
            chart_count += 1
    if len(profile.numeric_columns) >= 2:
        correlation_chart = _correlation_chart(df, profile.numeric_columns)
        if correlation_chart is not None:
            story.append(
                KeepTogether([Paragraph("Korelasi antar kolom numerik", styles["h2"]), correlation_chart])
            )
            chart_count += 1
    if chart_count == 0:
        story.append(Paragraph("Tidak ada grafik yang dapat dibuat dari kolom yang tersedia.", styles["body"]))

    try:
        matrix = correlation_matrix(df, profile.numeric_columns)
        pairs = top_correlations(matrix, limit=5)
    except ValueError:
        pairs = []
    if pairs:
        story.append(Paragraph("Korelasi terkuat (bukan bukti sebab-akibat)", styles["h2"]))
        story.append(
            _table(
                [["Kolom A", "Kolom B", "Koefisien"]]
                + [[pair["left"], pair["right"], f"{pair['correlation']:.3f}"] for pair in pairs],
                widths=[5.5 * cm, 5.5 * cm, 3 * cm],
            )
        )

    if model_result is not None:
        story.append(PageBreak())
        story.append(Paragraph("5. Hasil Model Machine Learning", styles["h1"]))
        story.append(
            _table(
                [
                    ["Konfigurasi", "Nilai"],
                    ["Jenis masalah", model_result.task],
                    ["Target", model_result.target],
                    ["Model", model_result.model_name],
                    ["Jumlah fitur", str(len(model_result.features))],
                    ["Baris latih / uji", f"{model_result.train_rows} / {model_result.test_rows}"],
                    ["Random state", str(model_result.config.get("random_state"))],
                ],
                widths=[5 * cm, 11.5 * cm],
            )
        )
        story.append(Paragraph("Metrik Evaluasi (data uji)", styles["h2"]))
        metric_rows = [["Metrik", "Nilai"]]
        for key, value in model_result.metrics.items():
            if value is None:
                continue
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
            metric_rows.append([key, formatted])
        if model_result.cv_metrics.get("mean") is not None:
            metric_rows.append(
                [
                    f"CV {model_result.cv_metrics['metric']} (mean ± std)",
                    f"{model_result.cv_metrics['mean']:.4f} ± {model_result.cv_metrics['std']:.4f}",
                ]
            )
        story.append(_table(metric_rows, widths=[6.5 * cm, 5 * cm]))

        predictions_chart = _predictions_chart(model_result)
        if predictions_chart is not None:
            story.append(
                KeepTogether([Paragraph("Perbandingan aktual dan prediksi", styles["h2"]), predictions_chart])
            )

        if model_result.feature_importance:
            story.append(Paragraph("Fitur Paling Berpengaruh", styles["h2"]))
            story.append(
                _table(
                    [["Fitur", "Kepentingan (%)"]]
                    + [
                        [item["feature"], f"{item.get('importance_pct', 0):.2f}"]
                        for item in model_result.feature_importance[:10]
                    ],
                    widths=[10 * cm, 4 * cm],
                )
            )

        if model_result.per_class:
            story.append(Paragraph("Metrik per Kelas", styles["h2"]))
            story.append(
                _table(
                    [["Kelas", "Precision", "Recall", "F1", "Support"]]
                    + [
                        [
                            item["label"],
                            f"{item['precision']:.3f}",
                            f"{item['recall']:.3f}",
                            f"{item['f1']:.3f}",
                            str(item["support"]),
                        ]
                        for item in model_result.per_class
                    ],
                    widths=[5 * cm, 3 * cm, 3 * cm, 3 * cm, 2.5 * cm],
                )
            )
    else:
        story.append(Paragraph("5. Model Machine Learning", styles["h1"]))
        story.append(Paragraph("Model belum dilatih pada sesi ini.", styles["body"]))

    story.append(PageBreak())
    story.append(Paragraph("6. Interpretasi", styles["h1"]))
    for title, key in (
        ("Gambaran Umum", "overview"),
        ("Kualitas Data", "quality"),
        ("Analitik Diagnostik", "diagnostics"),
        ("Hasil Model", "model"),
        ("Rekomendasi", "recommendations"),
    ):
        story.append(Paragraph(title, styles["h2"]))
        for item in interpretation.get(key, []):
            story.append(Paragraph(f"- {item}", styles["body"]))

    story.append(Paragraph("7. Keterbatasan dan Peringatan", styles["h1"]))
    limitations = list(dict.fromkeys(interpretation.get("limitations", [])))
    for item in limitations:
        story.append(Paragraph(f"- {item}", styles["body"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(interpretation.get("disclaimer", DISCLAIMER_ANALYTICS), styles["warning"]))

    document.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)
    return buffer.getvalue()
