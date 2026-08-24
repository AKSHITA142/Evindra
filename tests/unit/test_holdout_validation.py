import pytest
import pandas as pd
import numpy as np

from backend.schemas.best_pipeline import BestPipelineResult
from backend.schemas.holdout_validation import FinalValidationReport
from backend.engine.holdout_validator import HoldoutValidator


def test_normal_dataset_holdout_validation():
    """Verify good generalization assessment on normal holdout test set."""
    X_tr = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    y_tr = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    X_ho = pd.DataFrame({"f1": [1.5, 2.5, 3.5, 6.5, 7.5, 8.5]})
    y_ho = pd.Series([0, 0, 0, 1, 1, 1])

    best_res = BestPipelineResult(
        winner_pipeline_id="pipe_normal",
        winner_pipeline_name="Normal Pipe",
        winner_model_family="LOGISTIC_REGRESSION",
        metric="accuracy",
        score=0.90,
    )

    validator = HoldoutValidator()
    report = validator.validate_holdout(best_res, X_tr, X_ho, y_tr, y_ho, problem_type="classification")

    assert isinstance(report, FinalValidationReport)
    assert report.generalization_assessment in ("GOOD", "MILD_OVERFITTING")
    assert report.holdout_score > 0.80
    assert len(report.confusion_matrix) == 2


def test_overfitting_detection():
    """Verify CV score 0.95 vs holdout score 0.50 triggers SEVERE_OVERFITTING and warning alert."""
    X_tr = pd.DataFrame({"f1": np.random.randn(20), "f2": np.random.randn(20)})
    y_tr = pd.Series([0] * 10 + [1] * 10)

    # Completely random holdout test set to force poor holdout performance
    X_ho = pd.DataFrame({"f1": np.random.randn(20), "f2": np.random.randn(20)})
    y_ho = pd.Series([1] * 10 + [0] * 10)  # Opposite labels

    best_res = BestPipelineResult(
        winner_pipeline_id="pipe_overfit",
        winner_pipeline_name="Overfitted Model",
        winner_model_family="RANDOM_FOREST",
        metric="accuracy",
        score=0.95,  # High CV score
    )

    validator = HoldoutValidator()
    report = validator.validate_holdout(best_res, X_tr, X_ho, y_tr, y_ho, problem_type="classification")

    assert report.generalization_assessment == "SEVERE_OVERFITTING"
    assert any("Severe overfitting detected" in w for w in report.warnings)


def test_regression_dataset_holdout_validation():
    """Verify residual analysis, RMSE, MAE, R² on regression holdout validation."""
    X_tr = pd.DataFrame({"sqft": [500.0, 1000.0, 1500.0, 2000.0, 2500.0]})
    y_tr = pd.Series([150000.0, 250000.0, 350000.0, 450000.0, 550000.0])

    X_ho = pd.DataFrame({"sqft": [800.0, 1200.0, 1800.0]})
    y_ho = pd.Series([210000.0, 290000.0, 410000.0])

    best_res = BestPipelineResult(
        winner_pipeline_id="pipe_reg",
        winner_pipeline_name="Ridge Regressor",
        winner_model_family="RIDGE_REGRESSION",
        metric="rmse",
        score=10000.0,
    )

    validator = HoldoutValidator()
    report = validator.validate_holdout(best_res, X_tr, X_ho, y_tr, y_ho, problem_type="regression")

    assert "rmse" in report.holdout_metrics
    assert "mae" in report.holdout_metrics
    assert "r2" in report.holdout_metrics
    assert "mean_residual" in report.residual_analysis


def test_confusion_matrix_generation():
    """Verify 2x2 confusion matrix generation for binary classification holdout."""
    X_tr = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6, 7, 8]})
    y_tr = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    X_ho = pd.DataFrame({"f1": [1, 2, 7, 8]})
    y_ho = pd.Series([0, 0, 1, 1])

    best_res = BestPipelineResult(winner_pipeline_id="pipe_cm", winner_pipeline_name="CM Pipe", winner_model_family="LOGISTIC_REGRESSION", metric="accuracy", score=1.0)
    validator = HoldoutValidator()
    report = validator.validate_holdout(best_res, X_tr, X_ho, y_tr, y_ho, problem_type="classification")

    assert report.confusion_matrix == [[2, 0], [0, 2]]
    assert report.leakage_checks["holdout_isolation_verified"] == "PASSED"


def test_suspicious_score_jump_flagging():
    """Verify holdout score suspiciously higher than CV score triggers SUSPICIOUS assessment."""
    X_tr = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    y_tr = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    X_ho = pd.DataFrame({"f1": [1.0, 2.0, 7.0, 8.0]})
    y_ho = pd.Series([0, 0, 1, 1])

    # Low CV score vs 1.0 holdout score
    best_res = BestPipelineResult(winner_pipeline_id="pipe_susp", winner_pipeline_name="Suspicious Pipe", winner_model_family="LOGISTIC_REGRESSION", metric="accuracy", score=0.60)
    validator = HoldoutValidator()
    report = validator.validate_holdout(best_res, X_tr, X_ho, y_tr, y_ho, problem_type="classification")

    assert report.generalization_assessment == "SUSPICIOUS"
    assert any("Suspicious performance jump" in w for w in report.warnings)
