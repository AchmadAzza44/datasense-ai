"""Tests for interpretation and report generation (FR-06, FR-07)."""

from __future__ import annotations

import pandas as pd
import pytest

from datasense.cleaning import CleaningPlan, ColumnCleaningRule, apply_cleaning_plan
from datasense.interpretation import build_interpretation, interpretation_to_markdown
from datasense.modeling import train_model
from datasense.profiling import profile_dataframe
from datasense.reporting import build_report_pdf


def test_interpretation_contains_required_sections(sales_frame):
    profile = profile_dataframe(sales_frame)
    interpretation = build_interpretation(sales_frame, profile)
    for key in (
        "overview",
        "quality",
        "cleaning",
        "diagnostics",
        "model",
        "recommendations",
        "limitations",
        "disclaimer",
    ):
        assert key in interpretation
        assert interpretation[key]
    assert "sebab-akibat" in " ".join(interpretation["limitations"])


def test_interpretation_includes_cleaning_actions(sales_frame):
    profile = profile_dataframe(sales_frame)
    interpretation = build_interpretation(
        sales_frame, profile, cleaning_actions=["Menghapus 1 baris duplikat."]
    )
    assert "Menghapus 1 baris duplikat." in interpretation["cleaning"]


def test_interpretation_includes_model_metrics(sales_frame):
    profile = profile_dataframe(sales_frame)
    result = train_model(sales_frame, target="penjualan", task="regression", model_choice="linear")
    interpretation = build_interpretation(sales_frame, profile, model_result=result)
    joined = " ".join(interpretation["model"])
    assert "R²" in joined
    assert result.model_name in joined


def test_interpretation_markdown_renders(sales_frame):
    profile = profile_dataframe(sales_frame)
    interpretation = build_interpretation(sales_frame, profile)
    markdown = interpretation_to_markdown(interpretation)
    assert markdown.startswith("# Ringkasan Interpretasi")
    assert "## Rekomendasi" in markdown


def test_report_pdf_bytes(sales_frame):
    profile = profile_dataframe(sales_frame)
    pdf = build_report_pdf(
        df=sales_frame,
        profile=profile,
        dataset_name="sales.csv",
        dataset_sha256="a" * 64,
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 10_000


def test_report_pdf_with_model(sales_frame):
    cleaned = apply_cleaning_plan(
        sales_frame,
        CleaningPlan(drop_duplicates=True, rules=[ColumnCleaningRule("biaya_iklan", strategy="fill_median")]),
    ).dataframe
    profile = profile_dataframe(cleaned)
    result = train_model(cleaned, target="penjualan", task="regression", model_choice="linear")
    interpretation = build_interpretation(cleaned, profile, model_result=result)
    pdf = build_report_pdf(
        df=cleaned,
        profile=profile,
        dataset_name="sales.csv",
        dataset_sha256="b" * 64,
        cleaning_actions=["Mengisi nilai kosong."],
        model_result=result,
        interpretation=interpretation,
    )
    assert pdf.startswith(b"%PDF")
    assert b"R" in pdf[:2000] or len(pdf) > 20_000


def test_report_pdf_classification(churn_frame):
    profile = profile_dataframe(churn_frame)
    result = train_model(churn_frame, target="churn", task="classification", model_choice="logistic")
    pdf = build_report_pdf(
        df=churn_frame,
        profile=profile,
        dataset_name="churn.csv",
        dataset_sha256="c" * 64,
        model_result=result,
    )
    assert pdf.startswith(b"%PDF")
