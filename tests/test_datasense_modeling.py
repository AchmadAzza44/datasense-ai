"""Tests for model training and evaluation (FR-05)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datasense.modeling import ModelResult, predict_new_rows, train_model


def test_regression_metrics_are_computed(sales_frame):
    result = train_model(sales_frame, target="penjualan", task="regression", model_choice="linear")
    assert result.task == "regression"
    assert set(result.metrics) >= {"mae", "mse", "rmse", "r2"}
    assert result.metrics["r2"] > 0.6
    assert result.train_rows + result.test_rows == len(sales_frame)
    assert len(result.feature_importance) > 0


def test_regression_auto_selects_best_model(sales_frame):
    result = train_model(sales_frame, target="penjualan", task="regression", model_choice="auto")
    assert result.model_name in {"Linear Regression", "Ridge Regression", "Random Forest Regressor"}
    assert result.cv_metrics.get("metric") == "r2"


def test_classification_metrics_are_computed(churn_frame):
    result = train_model(churn_frame, target="churn", task="classification", model_choice="auto")
    assert result.task == "classification"
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert 0.0 <= result.metrics["f1_weighted"] <= 1.0
    assert result.confusion is not None
    assert len(result.per_class) == churn_frame["churn"].nunique()


def test_predictions_frame_has_expected_columns(sales_frame):
    result = train_model(sales_frame, target="penjualan", task="regression", model_choice="ridge")
    assert {"actual", "predicted", "residual"} <= set(result.predictions.columns)


def test_classification_predictions_include_probabilities(churn_frame):
    result = train_model(churn_frame, target="churn", task="classification", model_choice="logistic")
    probability_columns = [col for col in result.predictions.columns if col.startswith("prob_")]
    assert probability_columns
    assert "correct" in result.predictions.columns


def test_missing_target_raises(sales_frame):
    with pytest.raises(KeyError):
        train_model(sales_frame, target="tidak_ada")


def test_target_only_dataset_raises():
    frame = pd.DataFrame({"target": [1.0, 2.0, 3.0] * 10})
    with pytest.raises(ValueError):
        train_model(frame, target="target")


def test_too_few_rows_raises():
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="minimal"):
        train_model(frame, target="y")


def test_single_class_classification_raises():
    frame = pd.DataFrame({"x": range(30), "label": ["ya"] * 30})
    with pytest.raises(ValueError, match="dua kelas"):
        train_model(frame, target="label", task="classification")


def test_invalid_model_for_task_raises(sales_frame):
    with pytest.raises(ValueError):
        train_model(sales_frame, target="penjualan", task="regression", model_choice="logistic")


def test_predict_new_rows(sales_frame):
    result = train_model(sales_frame, target="penjualan", task="regression", model_choice="ridge")
    new_rows = sales_frame.drop(columns=["penjualan"]).head(3)
    predictions = predict_new_rows(result, new_rows)
    assert "prediction" in predictions.columns
    assert len(predictions) == 3


def test_predict_new_rows_missing_feature(sales_frame):
    result = train_model(sales_frame, target="penjualan", task="regression", model_choice="ridge")
    with pytest.raises(ValueError, match="tidak ditemukan"):
        predict_new_rows(result, pd.DataFrame({"harga": [10.0]}))


def test_model_result_to_dict_is_serializable(sales_frame):
    import json

    result = train_model(sales_frame, target="penjualan", task="regression", model_choice="linear")
    payload = json.dumps(result.to_dict(), allow_nan=False)
    assert "metrics" in payload


def test_limitations_are_reported(sales_frame):
    result = train_model(sales_frame, target="penjualan", task="regression", model_choice="linear")
    assert result.limitations
    assert any("sebab-akibat" in item for item in result.limitations)
