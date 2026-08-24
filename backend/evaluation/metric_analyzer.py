from typing import List, Dict, Any, Optional
from backend.schemas.experiment import ExperimentResult


class MetricAnalyzer:
    """Analyzes and normalizes raw metric results across experiment runs."""

    @classmethod
    def extract_primary_metric_score(cls, result: ExperimentResult) -> float:
        """Extracts primary metric score from an experiment result."""
        if not result.metrics or result.status != "completed":
            return 0.0
        return float(result.metrics.primary_metric)

    @classmethod
    def normalize_scores(cls, results: List[ExperimentResult]) -> Dict[str, float]:
        """
        Normalizes primary metric scores into [0.0, 1.0] relative scale.
        Automatically handles loss metrics (RMSE, MAE, MSE) where lower values indicate better models.
        """
        scores = {r.experiment_id: cls.extract_primary_metric_score(r) for r in results if r.status == "completed"}
        if not scores:
            return {}

        # Check if the primary metric is a loss metric (e.g. RMSE, MAE, MSE)
        is_loss_metric = False
        first_completed = next((r for r in results if r.status == "completed" and r.metrics), None)
        if first_completed and first_completed.metrics:
            metrics_dict = first_completed.metrics.metrics or {}
            if ("rmse" in metrics_dict or "mae" in metrics_dict) and "accuracy" not in metrics_dict:
                is_loss_metric = True

        min_val = min(scores.values())
        max_val = max(scores.values())

        if min_val == max_val:
            return {exp_id: 1.0 for exp_id in scores}

        range_val = max_val - min_val

        if is_loss_metric:
            return {exp_id: round(1.0 - ((val - min_val) / range_val), 4) for exp_id, val in scores.items()}
        else:
            return {exp_id: round((val - min_val) / range_val, 4) for exp_id, val in scores.items()}
