import pytest

from backend.schemas.experiment import ExperimentRunReport, PipelineEvaluationResult
from backend.schemas.best_pipeline import BestPipelineResult
from backend.engine.pipeline_selector import BestPipelineSelector


def test_obvious_winner_selection():
    """Verify obvious winner with superior score is selected."""
    e1 = PipelineEvaluationResult(pipeline_id="p1", pipeline_name="Linear", model_family="LOGISTIC_REGRESSION", status="SUCCESS", primary_metric="roc_auc", primary_score=0.70, std_metrics={"roc_auc": 0.01})
    e2 = PipelineEvaluationResult(pipeline_id="p2", pipeline_name="Ensemble", model_family="RANDOM_FOREST", status="SUCCESS", primary_metric="roc_auc", primary_score=0.95, std_metrics={"roc_auc": 0.01})

    report = ExperimentRunReport(dataset_name="obvious_ds", problem_type="classification", primary_metric="roc_auc", evaluation_results=[e1, e2])
    selector = BestPipelineSelector(simplicity_threshold=0.01)
    res = selector.select_best_pipeline(report)

    assert res.winner_pipeline_id == "p2"
    assert res.winner_model_family == "RANDOM_FOREST"
    assert res.score == 0.95


def test_simpler_pipeline_wins_within_threshold():
    """Verify simpler pipeline wins over slightly higher ensemble within 1% threshold."""
    e_simple = PipelineEvaluationResult(pipeline_id="p_simple", pipeline_name="Logistic", model_family="LOGISTIC_REGRESSION", status="SUCCESS", primary_metric="roc_auc", primary_score=0.895, feature_count=5, std_metrics={"roc_auc": 0.005})
    e_complex = PipelineEvaluationResult(pipeline_id="p_complex", pipeline_name="Heavy Ensemble", model_family="RANDOM_FOREST", status="SUCCESS", primary_metric="roc_auc", primary_score=0.900, feature_count=25, std_metrics={"roc_auc": 0.020})

    report = ExperimentRunReport(dataset_name="simplicity_ds", problem_type="classification", primary_metric="roc_auc", evaluation_results=[e_simple, e_complex])
    selector = BestPipelineSelector(simplicity_threshold=0.01)
    res = selector.select_best_pipeline(report)

    # Simple pipeline is within 0.005 of 0.900 (<= 0.01 threshold), so simpler candidate wins!
    assert res.winner_pipeline_id == "p_simple"
    assert res.tradeoffs["is_simpler_pipeline_chosen"] is True
    assert "Selected simpler pipeline" in res.selection_reason


def test_tied_pipelines_selection():
    """Verify tied pipeline metrics select the simpler/faster pipeline with fewer features."""
    e1 = PipelineEvaluationResult(pipeline_id="p1", pipeline_name="Complex Tie", model_family="RANDOM_FOREST", status="SUCCESS", primary_metric="accuracy", primary_score=0.85, feature_count=30)
    e2 = PipelineEvaluationResult(pipeline_id="p2", pipeline_name="Simple Tie", model_family="LOGISTIC_REGRESSION", status="SUCCESS", primary_metric="accuracy", primary_score=0.85, feature_count=5)

    report = ExperimentRunReport(dataset_name="tied_ds", problem_type="classification", primary_metric="accuracy", evaluation_results=[e1, e2])
    selector = BestPipelineSelector()
    res = selector.select_best_pipeline(report)

    assert res.winner_pipeline_id == "p2"


def test_high_variance_pipeline_penalization():
    """Verify high std variance pipeline is penalized in favor of stable low std pipeline."""
    e_unstable = PipelineEvaluationResult(pipeline_id="p_unstable", pipeline_name="Unstable", model_family="LIGHTGBM", status="SUCCESS", primary_metric="roc_auc", primary_score=0.86, std_metrics={"roc_auc": 0.15})
    e_stable = PipelineEvaluationResult(pipeline_id="p_stable", pipeline_name="Stable", model_family="LOGISTIC_REGRESSION", status="SUCCESS", primary_metric="roc_auc", primary_score=0.85, std_metrics={"roc_auc": 0.01})

    report = ExperimentRunReport(dataset_name="variance_ds", problem_type="classification", primary_metric="roc_auc", evaluation_results=[e_unstable, e_stable])
    selector = BestPipelineSelector(std_penalty_weight=1.0)
    res = selector.select_best_pipeline(report)

    assert res.winner_pipeline_id == "p_stable"


def test_failed_pipeline_filtering():
    """Verify failed candidate pipelines are filtered out during selection."""
    e_failed = PipelineEvaluationResult(pipeline_id="p_failed", pipeline_name="Broken", model_family="UNKNOWN", status="FAILED", error_message="Runtime error")
    e_valid = PipelineEvaluationResult(pipeline_id="p_valid", pipeline_name="Valid", model_family="RIDGE_REGRESSION", status="SUCCESS", primary_metric="rmse", primary_score=10.5)

    report = ExperimentRunReport(dataset_name="failed_ds", problem_type="regression", primary_metric="rmse", evaluation_results=[e_failed, e_valid])
    selector = BestPipelineSelector()
    res = selector.select_best_pipeline(report)

    assert res.winner_pipeline_id == "p_valid"
    assert res.score == 10.5


def test_missing_metrics_handling():
    """Verify missing std/metrics in evaluation results are handled gracefully."""
    e1 = PipelineEvaluationResult(pipeline_id="p1", pipeline_name="No Std", model_family="LOGISTIC_REGRESSION", status="SUCCESS", primary_metric="accuracy", primary_score=0.80)
    report = ExperimentRunReport(dataset_name="missing_met", problem_type="classification", primary_metric="accuracy", evaluation_results=[e1])

    selector = BestPipelineSelector()
    res = selector.select_best_pipeline(report)

    assert res.winner_pipeline_id == "p1"
    assert res.confidence >= 0.50
