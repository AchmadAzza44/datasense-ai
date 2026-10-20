"""DataSense AI: analytics, machine learning, and reporting core."""

from .io import (
    DatasetValidationError,
    LoadedDataset,
    load_dataset,
    read_dataframe,
    validate_upload,
)
from .profiling import ProfileReport, profile_dataframe
from .cleaning import CleaningPlan, CleaningResult, apply_cleaning_plan
from .eda import correlation_matrix, numeric_distribution, categorical_distribution
from .modeling import ModelResult, train_model
from .interpretation import build_interpretation
from .reporting import build_report_pdf

__all__ = [
    "DatasetValidationError",
    "LoadedDataset",
    "load_dataset",
    "read_dataframe",
    "validate_upload",
    "ProfileReport",
    "profile_dataframe",
    "CleaningPlan",
    "CleaningResult",
    "apply_cleaning_plan",
    "correlation_matrix",
    "numeric_distribution",
    "categorical_distribution",
    "ModelResult",
    "train_model",
    "build_interpretation",
    "build_report_pdf",
]
