import pytest
import pandas as pd
import numpy as np

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.pipeline import PipelineCandidateSet, PipelineCandidate
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan
from backend.schemas.decision import DecisionDomain, DecisionSource
from backend.schemas.experiment import ExperimentRunReport, PipelineEvaluationResult
from backend.engine.experiment_runner import ExperimentRunner


def test_normal_dataset_experiment():
    """Verify CV evaluation on standard classification dataset."""
    df_raw = pd.DataFrame({
        "age": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
        "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0, 100000.0, 110000.0, 120000.0, 130000.0, 140000.0],
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })

    step_scale = PreprocessingStep(step_number=1, stage="SCALING", domain=DecisionDomain.SCALING_TRANSFORMATION, action="STANDARD_SCALER", columns=["age", "income"], decision_id="d1", decision_source=DecisionSource.RULE, confidence=0.9)
    plan1 = PreprocessingPlan(dataset_name="normal_ds", target_column="target", steps=[step_scale])

    p1 = PipelineCandidate(name="Normal Pipe", description="Baseline", preprocessing_plan=plan1, model_spec={"model_family": "LOGISTIC_REGRESSION"})
    pset = PipelineCandidateSet(dataset_name="normal_ds", problem_type="classification", target_column="target", pipelines=[p1])

    runner = ExperimentRunner(n_folds=2)
    report = runner.run_experiment(pset, df_raw, target_column="target")

    assert isinstance(report, ExperimentRunReport)
    assert report.successful_evaluations == 1
    assert report.best_pipeline_id == p1.pipeline_id
    assert "accuracy" in report.evaluation_results[0].mean_metrics


def test_imbalanced_dataset_experiment():
    """Verify severe class imbalance prioritizes PR-AUC/F1/ROC-AUC metric over accuracy."""
    df_raw = pd.DataFrame({
        "feature1": np.random.randn(20),
        "target": [0] * 18 + [1] * 2,  # 9:1 imbalance ratio
    })

    step_pass = PreprocessingStep(step_number=1, stage="MISSING_VALUE_HANDLING", domain=DecisionDomain.MISSING_VALUE_STRATEGY, action="PASS_THROUGH", columns=["feature1"], decision_id="d1", decision_source=DecisionSource.RULE, confidence=0.9)
    plan = PreprocessingPlan(dataset_name="imb_ds", target_column="target", steps=[step_pass])
    p1 = PipelineCandidate(name="Imbalanced Pipe", description="Imbalance test", preprocessing_plan=plan, model_spec={"model_family": "LOGISTIC_REGRESSION"})
    pset = PipelineCandidateSet(dataset_name="imb_ds", problem_type="classification", target_column="target", pipelines=[p1])

    runner = ExperimentRunner(n_folds=2)
    report = runner.run_experiment(pset, df_raw, target_column="target")

    # Verify primary metric is set to PR-AUC/ROC-AUC instead of default accuracy
    assert report.primary_metric in ("pr_auc", "roc_auc", "f1")


def test_regression_dataset_experiment():
    """Verify RMSE, MAE, R² metrics on regression dataset."""
    df_raw = pd.DataFrame({
        "sqft": [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0],
        "price": [150000.0, 250000.0, 350000.0, 450000.0, 550000.0, 650000.0],
    })

    step_scale = PreprocessingStep(step_number=1, stage="SCALING", domain=DecisionDomain.SCALING_TRANSFORMATION, action="STANDARD_SCALER", columns=["sqft"], decision_id="d1", decision_source=DecisionSource.RULE, confidence=0.9)
    plan = PreprocessingPlan(dataset_name="reg_ds", target_column="price", steps=[step_scale])
    p1 = PipelineCandidate(name="Ridge Pipe", description="Ridge Regressor", preprocessing_plan=plan, model_spec={"model_family": "RIDGE_REGRESSION"})
    pset = PipelineCandidateSet(dataset_name="reg_ds", problem_type="regression", target_column="price", pipelines=[p1])

    runner = ExperimentRunner(n_folds=2)
    report = runner.run_experiment(pset, df_raw, target_column="price")

    assert report.primary_metric == "rmse"
    res = report.evaluation_results[0]
    assert "rmse" in res.mean_metrics
    assert "mae" in res.mean_metrics
    assert "r2" in res.mean_metrics


def test_temporal_dataset_experiment():
    """Verify TimeSeriesSplit cross-validation execution."""
    df_raw = pd.DataFrame({
        "val": np.sin(np.linspace(0, 10, 12)),
        "time_col": pd.date_range("2026-01-01", periods=12, freq="D"),
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    step_pass = PreprocessingStep(step_number=1, stage="MISSING_VALUE_HANDLING", domain=DecisionDomain.MISSING_VALUE_STRATEGY, action="PASS_THROUGH", columns=["val"], decision_id="d1", decision_source=DecisionSource.RULE, confidence=0.9)
    plan = PreprocessingPlan(dataset_name="time_ds", target_column="target", steps=[step_pass])
    p1 = PipelineCandidate(name="Time Pipe", description="Temporal CV", preprocessing_plan=plan, model_spec={"model_family": "LOGISTIC_REGRESSION"})
    pset = PipelineCandidateSet(dataset_name="time_ds", problem_type="classification", target_column="target", pipelines=[p1])

    runner = ExperimentRunner(n_folds=3)
    report = runner.run_experiment(pset, df_raw, target_column="target", time_column="time_col")

    assert report.successful_evaluations == 1



def test_grouped_dataset_experiment():
    """Verify GroupKFold cross-validation execution."""
    df_raw = pd.DataFrame({
        "feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "group_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    step_scale = PreprocessingStep(step_number=1, stage="SCALING", domain=DecisionDomain.SCALING_TRANSFORMATION, action="STANDARD_SCALER", columns=["feat"], decision_id="d1", decision_source=DecisionSource.RULE, confidence=0.9)
    plan = PreprocessingPlan(dataset_name="grp_ds", target_column="target", steps=[step_scale])
    p1 = PipelineCandidate(name="Group Pipe", description="Grouped CV", preprocessing_plan=plan, model_spec={"model_family": "LOGISTIC_REGRESSION"})
    pset = PipelineCandidateSet(dataset_name="grp_ds", problem_type="classification", target_column="target", pipelines=[p1])

    runner = ExperimentRunner(n_folds=2)
    report = runner.run_experiment(pset, df_raw, target_column="target", groups=df_raw["group_id"])

    assert report.successful_evaluations == 1


def test_pipeline_failure_fault_tolerance():
    """Verify single pipeline failure does NOT crash experiment and remaining candidates execute successfully."""
    df_raw = pd.DataFrame({
        "age": [20, 30, 40, 50, 60, 70],
        "target": [0, 1, 0, 1, 0, 1],
    })

    step_pass = PreprocessingStep(step_number=1, stage="MISSING_VALUE_HANDLING", domain=DecisionDomain.MISSING_VALUE_STRATEGY, action="PASS_THROUGH", columns=["age"], decision_id="d1", decision_source=DecisionSource.RULE, confidence=0.9)
    plan = PreprocessingPlan(dataset_name="fail_ds", target_column="target", steps=[step_pass])

    # p1 is valid
    p1 = PipelineCandidate(name="Valid Pipe", description="Good candidate", preprocessing_plan=plan, model_spec={"model_family": "LOGISTIC_REGRESSION"})
    # p2 is invalid model family designed to fail
    p2 = PipelineCandidate(name="Broken Pipe", description="Invalid model family", preprocessing_plan=plan, model_spec={"model_family": "INVALID_MAGIC_MODEL"})

    pset = PipelineCandidateSet(dataset_name="fail_ds", problem_type="classification", target_column="target", pipelines=[p1, p2])

    runner = ExperimentRunner(n_folds=2)
    report = runner.run_experiment(pset, df_raw, target_column="target")

    assert report.total_pipelines_evaluated == 2
    assert report.successful_evaluations == 1
    assert report.failed_evaluations == 1
    assert report.evaluation_results[1].status == "FAILED"
    assert "Unsupported or invalid model family" in report.evaluation_results[1].error_message
