from typing import List, Dict, Optional, Any
from backend.schemas.experiment import ExperimentResult
from backend.schemas.evaluation import RankingItem
from backend.schemas.mission_brief import MissionBrief
from backend.evaluation.metric_analyzer import MetricAnalyzer
from backend.evaluation.stability_analyzer import StabilityAnalyzer
from backend.evaluation.generalization_analyzer import GeneralizationAnalyzer
from backend.evaluation.constraint_validator import ConstraintValidator
from backend.evaluation.tradeoff_analyzer import TradeoffAnalyzer


class RankingEngine:
    """Computes multi-dimensional composite ranking scores for experiment batches."""

    DEFAULT_WEIGHTS = {
        "primary_metric": 0.35,
        "generalization": 0.25,
        "stability": 0.20,
        "runtime": 0.10,
        "interpretability": 0.10,
    }

    @classmethod
    def rank_experiments(
        cls,
        results: List[ExperimentResult],
        mission_brief: Optional[MissionBrief] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> List[RankingItem]:
        """Ranks completed experiments by composite score and returns List[RankingItem]."""
        weights = custom_weights or cls.DEFAULT_WEIGHTS

        norm_metrics = MetricAnalyzer.normalize_scores(results)
        stability_scores = StabilityAnalyzer.analyze_batch(results)
        gen_scores = GeneralizationAnalyzer.analyze_batch(results)
        compliance_map = ConstraintValidator.validate_batch(results, mission_brief)
        efficiency_map = TradeoffAnalyzer.calculate_efficiency(results)

        items: List[Dict[str, Any]] = []

        # Determine if we are sorting loss metrics (RMSE/MAE where lower is better)
        is_loss_metric = False
        first_exp = next((r for r in results if r.status == "completed" and r.metrics), None)
        if first_exp and first_exp.metrics:
            m_dict = first_exp.metrics.metrics or {}
            if ("rmse" in m_dict or "mae" in m_dict) and "accuracy" not in m_dict:
                is_loss_metric = True

        for r in results:
            exp_id = r.experiment_id
            is_valid = compliance_map.get(exp_id, False)

            if r.status != "completed" or not is_valid:
                composite = 0.0
                p_val = -999999999.0
            else:
                p_norm = norm_metrics.get(exp_id, 0.5)
                g_score = gen_scores.get(exp_id, 0.5)
                s_score = stability_scores.get(exp_id, 0.5)
                e_score = efficiency_map.get(exp_id, 0.5)

                composite = (
                    weights["primary_metric"] * p_norm
                    + weights["generalization"] * g_score
                    + weights["stability"] * s_score
                    + weights["runtime"] * e_score
                )
                p_val = -r.metrics.primary_metric if (is_loss_metric and r.metrics) else (r.metrics.primary_metric if r.metrics else 0.0)

            items.append({
                "experiment_id": exp_id,
                "composite_score": round(composite, 4),
                "primary_metric": r.metrics.primary_metric if r.metrics else 0.0,
                "sort_metric": p_val,
                "model": r.model,
            })

        # Sort descending by composite_score, then sort_metric
        items.sort(key=lambda x: (x["composite_score"], x["sort_metric"]), reverse=True)

        # Build final RankingItem instances with rank indices (1-indexed)
        ranked_list: List[RankingItem] = []
        for idx, item in enumerate(items, start=1):
            ranked_list.append(
                RankingItem(
                    rank=idx,
                    experiment_id=item["experiment_id"],
                    score=item["composite_score"],
                    model=item["model"],
                )
            )

        return ranked_list
