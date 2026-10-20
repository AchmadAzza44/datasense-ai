"""Machine learning training and evaluation for DataSense AI (FR-05)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.utils import to_jsonable

TaskType = Literal["regression", "classification"]
ModelChoice = Literal["auto", "linear", "ridge", "random_forest", "logistic"]

RANDOM_STATE = 42
MIN_ROWS = 20


@dataclass
class ModelResult:
    """Trained model, metrics, and everything needed to reproduce the run."""

    task: str
    target: str
    features: list[str]
    model_name: str
    metrics: dict[str, Any]
    cv_metrics: dict[str, Any]
    train_rows: int
    test_rows: int
    feature_importance: list[dict]
    predictions: pd.DataFrame
    confusion: list[list[int]] | None
    per_class: list[dict]
    config: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    pipeline: Pipeline | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task": self.task,
            "target": self.target,
            "features": self.features,
            "model_name": self.model_name,
            "metrics": self.metrics,
            "cv_metrics": self.cv_metrics,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "feature_importance": self.feature_importance,
            "confusion": self.confusion,
            "per_class": self.per_class,
            "config": self.config,
            "warnings": self.warnings,
            "limitations": self.limitations,
        }
        return to_jsonable(payload)


def _detect_task(series: pd.Series) -> TaskType:
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        return "regression"
    return "classification"


def _build_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = [col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])]
    categorical_cols = [col for col in features.columns if col not in numeric_cols]
    transformers: list[tuple] = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("Tidak ada fitur yang dapat digunakan untuk pelatihan.")
    return ColumnTransformer(transformers=transformers), numeric_cols, categorical_cols


def _candidate_models(task: TaskType, choice: ModelChoice) -> list[tuple[str, Any]]:
    if task == "regression":
        registry = {
            "linear": ("Linear Regression", LinearRegression()),
            "ridge": ("Ridge Regression", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            "random_forest": (
                "Random Forest Regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        }
        if choice == "auto":
            return [registry["linear"], registry["ridge"], registry["random_forest"]]
        if choice == "logistic":
            raise ValueError("Model logistic hanya tersedia untuk klasifikasi.")
        return [registry[choice]]
    registry = {
        "logistic": (
            "Logistic Regression",
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "random_forest": (
            "Random Forest Classifier",
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }
    if choice == "auto":
        return [registry["logistic"], registry["random_forest"]]
    if choice in ("linear", "ridge"):
        raise ValueError("Model linear/ridge hanya tersedia untuk regresi.")
    return [registry[choice]]


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    mask = actual != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    mse = float(mean_squared_error(actual, predicted))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
        "mape_pct": _safe_mape(actual, predicted),
    }


def _classification_metrics(actual, predicted, probabilities, classes) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_weighted": float(precision_score(actual, predicted, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(actual, predicted, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
    }
    if probabilities is not None and len(classes) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(actual, probabilities[:, 1]))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def _feature_importance(pipeline: Pipeline, feature_names: list[str], limit: int = 15) -> list[dict]:
    try:
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        names = np.array(feature_names, dtype=object)
    model = pipeline.named_steps["model"]
    values: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype="float64")
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype="float64")
        values = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    if values is None or len(values) != len(names):
        return []
    cleaned = [
        {
            "feature": str(name).replace("numeric__", "").replace("categorical__", ""),
            "importance": float(value),
        }
        for name, value in zip(names, values)
    ]
    cleaned.sort(key=lambda item: abs(item["importance"]), reverse=True)
    total = sum(abs(item["importance"]) for item in cleaned) or 1.0
    for item in cleaned:
        item["importance_pct"] = round(abs(item["importance"]) / total * 100, 2)
    return cleaned[:limit]


def _limitations(task: str, rows: int, metrics: dict[str, Any], cv: dict[str, Any]) -> list[str]:
    items = [
        "Metrik dihitung pada data uji yang dipisahkan secara acak; performa pada data baru "
        "di masa depan dapat berbeda.",
        "Korelasi dan fitur penting tidak membuktikan hubungan sebab-akibat.",
    ]
    if rows < 100:
        items.append("Ukuran dataset kecil; hasil evaluasi memiliki variansi tinggi.")
    if task == "regression":
        r2 = metrics.get("r2")
        if r2 is not None and r2 < 0.3:
            items.append("Nilai R² rendah; model belum menjelaskan variasi target dengan baik.")
    else:
        accuracy = metrics.get("accuracy")
        if accuracy is not None and accuracy < 0.7:
            items.append("Akurasi klasifikasi di bawah 0.70; hasil belum layak untuk keputusan otomatis.")
    if cv.get("std", 0) > 0.1:
        items.append("Variasi metrik antar-fold cukup besar; model sensitif terhadap pembagian data.")
    return items


def train_model(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    task: TaskType | None = None,
    model_choice: ModelChoice = "auto",
    test_size: float = 0.2,
) -> ModelResult:
    """Train and evaluate a regression or classification model."""
    if df is None or df.empty:
        raise ValueError("Dataset kosong tidak dapat digunakan untuk pelatihan.")
    if target not in df.columns:
        raise KeyError(f"Kolom target '{target}' tidak ditemukan.")

    feature_columns = [col for col in (features or []) if col in df.columns and col != target]
    if not feature_columns:
        feature_columns = [col for col in df.columns if col != target]
    if not feature_columns:
        raise ValueError("Tidak ada kolom fitur selain target.")

    working = df[feature_columns + [target]].copy()
    working = working.dropna(subset=[target])
    if working.shape[0] < MIN_ROWS:
        raise ValueError(
            f"Data valid setelah menghapus target kosong hanya {working.shape[0]} baris; "
            f"minimal {MIN_ROWS} baris diperlukan."
        )

    warnings: list[str] = []
    detected = task or _detect_task(working[target])

    if detected == "regression":
        working[target] = pd.to_numeric(working[target], errors="coerce")
        working = working.dropna(subset=[target])
        if working.shape[0] < MIN_ROWS:
            raise ValueError("Nilai target numerik valid terlalu sedikit untuk regresi.")
        if working[target].nunique() < 5:
            warnings.append(
                "Target numerik memiliki kurang dari 5 nilai unik; pertimbangkan klasifikasi."
            )
    else:
        working[target] = working[target].astype(str)
        class_counts = working[target].value_counts()
        if class_counts.shape[0] < 2:
            raise ValueError("Klasifikasi membutuhkan minimal dua kelas pada kolom target.")
        if class_counts.min() < 2:
            warnings.append(
                "Terdapat kelas dengan kurang dari 2 contoh; stratifikasi mungkin gagal."
            )

    if working.shape[0] < MIN_ROWS:
        raise ValueError("Data valid terlalu sedikit setelah pembersihan target.")

    id_like = [
        col
        for col in feature_columns
        if working[col].nunique(dropna=True) == working.shape[0]
        and not pd.api.types.is_float_dtype(working[col])
    ]
    if id_like:
        warnings.append(
            "Kolom berikut tampak seperti identifier unik dan dapat menyebabkan overfitting: "
            + ", ".join(id_like)
        )

    X = working[feature_columns]
    y = working[target]
    stratify = y if detected == "classification" and y.value_counts().min() >= 2 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError as exc:
        warnings.append("Stratifikasi gagal; menggunakan pembagian acak tanpa stratifikasi.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )

    preprocessor, numeric_cols, categorical_cols = _build_preprocessor(X)
    candidates = _candidate_models(detected, model_choice)

    best: tuple[float, str, Pipeline, dict[str, Any], dict[str, Any]] | None = None
    for display_name, estimator in candidates:
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        if detected == "regression":
            metrics = _regression_metrics(y_test.to_numpy(dtype="float64"), predictions)
            primary = metrics.get("r2") or 0.0
        else:
            probabilities = None
            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                probabilities = pipeline.predict_proba(X_test)
            classes = pipeline.named_steps["model"].classes_
            metrics = _classification_metrics(y_test, predictions, probabilities, classes)
            primary = metrics.get("f1_weighted") or 0.0

        cv_metrics: dict[str, Any] = {}
        try:
            splits = 5 if working.shape[0] >= 100 else 3
            if detected == "regression":
                cv = KFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
                cv_metrics = {"metric": "r2", "mean": float(np.mean(scores)), "std": float(np.std(scores))}
            else:
                cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted")
                cv_metrics = {
                    "metric": "f1_weighted",
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                }
        except ValueError:
            warnings.append("Cross-validation tidak dapat dijalankan pada konfigurasi ini.")

        if best is None or primary > best[0]:
            best = (primary, display_name, pipeline, metrics, cv_metrics)

    assert best is not None
    _, model_name, pipeline, metrics, cv_metrics = best
    predictions = pipeline.predict(X_test)

    prediction_frame = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": predictions,
        }
    )
    if detected == "regression":
        prediction_frame["residual"] = prediction_frame["actual"] - prediction_frame["predicted"]
    else:
        model = pipeline.named_steps["model"]
        if hasattr(model, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)
            classes = list(model.classes_)
            for index, label in enumerate(classes):
                prediction_frame[f"prob_{label}"] = probabilities[:, index]
        prediction_frame["correct"] = prediction_frame["actual"] == prediction_frame["predicted"]

    confusion: list[list[int]] | None = None
    per_class: list[dict] = []
    if detected == "classification":
        labels = sorted(pd.unique(y).tolist())
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        confusion = matrix.astype(int).tolist()
        report = classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        )
        for label in labels:
            entry = report.get(str(label), {})
            per_class.append(
                {
                    "label": str(label),
                    "precision": float(entry.get("precision", 0.0)),
                    "recall": float(entry.get("recall", 0.0)),
                    "f1": float(entry.get("f1-score", 0.0)),
                    "support": int(entry.get("support", 0)),
                }
            )

    importance = _feature_importance(pipeline, feature_columns)
    limitations = _limitations(detected, working.shape[0], metrics, cv_metrics)

    config = {
        "task": detected,
        "target": target,
        "features": feature_columns,
        "model": model_name,
        "model_choice": model_choice,
        "test_size": test_size,
        "random_state": RANDOM_STATE,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "rows_used": int(working.shape[0]),
        "imputation": {"numeric": "median", "categorical": "most_frequent"},
        "scaling": "standard" if numeric_cols else "none",
        "encoding": "one-hot" if categorical_cols else "none",
    }

    return ModelResult(
        task=detected,
        target=target,
        features=feature_columns,
        model_name=model_name,
        metrics=metrics,
        cv_metrics=cv_metrics,
        train_rows=int(X_train.shape[0]),
        test_rows=int(X_test.shape[0]),
        feature_importance=importance,
        predictions=prediction_frame.head(500),
        confusion=confusion,
        per_class=per_class,
        config=config,
        warnings=warnings,
        limitations=limitations,
        pipeline=pipeline,
    )


def predict_new_rows(result: ModelResult, rows: pd.DataFrame) -> pd.DataFrame:
    """Apply a trained model to new rows containing the same feature columns."""
    if result.pipeline is None:
        raise ValueError("Model tidak tersedia pada hasil ini.")
    missing = [col for col in result.features if col not in rows.columns]
    if missing:
        raise ValueError("Kolom fitur berikut tidak ditemukan: " + ", ".join(missing))
    predictions = result.pipeline.predict(rows[result.features])
    output = rows.copy()
    output["prediction"] = predictions
    model = result.pipeline.named_steps["model"]
    if result.task == "classification" and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(rows[result.features])
        for index, label in enumerate(model.classes_):
            output[f"prob_{label}"] = probabilities[:, index]
    return output
