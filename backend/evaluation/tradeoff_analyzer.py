from typing import Dict, List
from backend.schemas.experiment import ExperimentResult
from backend.evaluation.metric_analyzer import MetricAnalyzer


class TradeoffAnalyzer:
    """Evaluates metric efficiency vs runtime compute cost."""

    @classmethod
    def calculate_efficiency(cls, results: List[ExperimentResult]) -> Dict[str, float]:
        """Calculates normalized efficiency score (normalized_score / runtime) in [0.0, 1.0]."""
        efficiency_map: Dict[str, float] = {}
        completed = [r for r in results if r.status == "completed" and r.metrics]

        if not completed:
            return {r.experiment_id: 0.0 for r in results}

        norm_scores = MetricAnalyzer.normalize_scores(completed)

        raw_ratios = {}
        for r in completed:
            runtime = max(0.001, r.runtime or 1.0)
            # Use normalized relative score so higher score is always better (for both classification and regression)
            score = norm_scores.get(r.experiment_id, 0.5)
            raw_ratios[r.experiment_id] = score / runtime

        max_ratio = max(raw_ratios.values()) if raw_ratios else 1.0
        min_ratio = min(raw_ratios.values()) if raw_ratios else 0.0

        if max_ratio == min_ratio:
            return {r.experiment_id: 1.0 for r in completed}

        range_val = max_ratio - min_ratio
        for exp_id, val in raw_ratios.items():
            efficiency_map[exp_id] = round((val - min_ratio) / range_val, 4)

        return efficiency_map
