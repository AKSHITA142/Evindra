from typing import Dict, List
from backend.schemas.experiment import ExperimentResult


class GeneralizationAnalyzer:
    """Evaluates generalization gap and overfitting risk across experiment runs."""

    @classmethod
    def evaluate_generalization(cls, result: ExperimentResult) -> float:
        """Returns generalization score in [0.0, 1.0]. High score = low overfitting risk."""
        if not result.metrics or result.status != "completed":
            return 0.0

        # Measure overfitting risk using honest train_test_gap if available
        metrics_dict = result.metrics.metrics or {}
        if "train_test_gap" in metrics_dict:
            gap = abs(float(metrics_dict["train_test_gap"]))
            # Small gap (<0.05) indicates high generalization (low overfitting)
            gen_score = max(0.0, 1.0 - (gap / 0.25))
            return round(gen_score, 4)

        # Fallback to CV fold spread if train_test_gap absent
        cv_scores = result.metrics.cv_scores
        if cv_scores:
            cv_min = min(cv_scores)
            cv_max = max(cv_scores)
            gap = cv_max - cv_min
            gen_score = max(0.0, 1.0 - (gap / 0.15))
            return round(gen_score, 4)

        return 0.8  # Default moderate generalization score if fold details absent

    @classmethod
    def analyze_batch(cls, results: List[ExperimentResult]) -> Dict[str, float]:
        """Returns dictionary mapping experiment_id -> generalization score."""
        return {r.experiment_id: cls.evaluate_generalization(r) for r in results}
