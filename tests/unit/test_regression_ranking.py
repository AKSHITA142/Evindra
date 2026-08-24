import pandas as pd
import numpy as np
from backend.ml_execution.metrics import MetricEngine
from backend.evaluation.metric_analyzer import MetricAnalyzer
from backend.schemas.experiment import ExperimentResult, MetricsResult, PipelineDefinition


def test_positive_rmse_calculation():
    y_true = np.array([50000, 60000, 70000, 80000])
    y_pred = np.array([52000, 58000, 71000, 79000])

    res = MetricEngine.compute_metrics(y_true, y_pred, task_type="regression")

    # Verify RMSE is strictly positive
    assert res.metrics["rmse"] > 0.0
    assert res.primary_metric > 0.0
    assert res.metrics["rmse"] == round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4)


def test_loss_metric_normalization_ranking():
    # Model A: RMSE 50.0 (Best, lowest error)
    # Model B: RMSE 100.0 (Worst, highest error)
    res_a = ExperimentResult(
        experiment_id="EXP_A",
        model="ModelA",
        pipeline=PipelineDefinition(operations=[], model_name="ModelA"),
        status="completed",
        metrics=MetricsResult(primary_metric=50.0, metrics={"rmse": 50.0})
    )

    res_b = ExperimentResult(
        experiment_id="EXP_B",
        model="ModelB",
        pipeline=PipelineDefinition(operations=[], model_name="ModelB"),
        status="completed",
        metrics=MetricsResult(primary_metric=100.0, metrics={"rmse": 100.0})
    )

    scores = MetricAnalyzer.normalize_scores([res_a, res_b])

    # Model A (smallest error 50.0) MUST get score 1.0 (Rank 1)
    assert scores["EXP_A"] == 1.0
    # Model B (largest error 100.0) MUST get score 0.0 (Rank 2)
    assert scores["EXP_B"] == 0.0
