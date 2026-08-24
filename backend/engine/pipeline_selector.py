import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from backend.schemas.experiment import ExperimentRunReport, PipelineEvaluationResult
from backend.schemas.best_pipeline import BestPipelineResult

logger = logging.getLogger("datapilot.engine.pipeline_selector")


class BestPipelineSelector:
    """
    Best Pipeline Selector for Evindra Pipeline (Phase 15).
    Deterministically selects the optimal final pipeline using multi-criteria scoring
    (primary metric, fold std stability, feature count, inference latency, complexity penalty,
    and configurable simplicity threshold).
    """

    def __init__(
        self,
        simplicity_threshold: float = 0.01,  # 1% performance equivalence threshold
        std_penalty_weight: float = 1.0,
        feature_penalty_weight: float = 0.0005,
        latency_penalty_weight: float = 0.01,
    ):
        self.simplicity_threshold = simplicity_threshold
        self.std_penalty_weight = std_penalty_weight
        self.feature_penalty_weight = feature_penalty_weight
        self.latency_penalty_weight = latency_penalty_weight

    def select_best_pipeline(
        self,
        report: ExperimentRunReport,
        simplicity_threshold: Optional[float] = None,
    ) -> BestPipelineResult:
        """
        Deterministically selects the winner pipeline from an ExperimentRunReport.

        Args:
            report: ExperimentRunReport containing candidate evaluations.
            simplicity_threshold: Optional override for simplicity equivalence threshold.

        Returns:
            BestPipelineResult with complete winner details, confidence, and tradeoffs.
        """
        threshold = simplicity_threshold if simplicity_threshold is not None else self.simplicity_threshold
        metric = report.primary_metric
        is_lower_better = metric.lower() in ("rmse", "mae", "log_loss")

        # 1. Filter out failed or invalid pipeline evaluations
        successful_evals = [res for res in report.evaluation_results if res.status == "SUCCESS"]

        if not successful_evals:
            raise ValueError(f"Selection Error: No successful pipeline evaluations found in experiment run report '{report.run_id}'.")

        # 2. Compute composite scores and normalized metrics
        scored_candidates: List[Dict[str, Any]] = []

        for res in successful_evals:
            raw_score = res.primary_score
            # Normalize score so higher is always better
            norm_score = -raw_score if is_lower_better else raw_score
            std_val = res.std_metrics.get(metric, 0.0)

            # Model complexity penalty
            mf = res.model_family.upper()
            if any(k in mf for k in ("LOGISTIC", "RIDGE", "LINEAR", "PASSTHROUGH")):
                complexity_penalty = 0.0
                complexity_tier = "LOW"
            elif any(k in mf for k in ("ELASTICNET", "LASSO", "KNN")):
                complexity_penalty = 0.002
                complexity_tier = "MEDIUM_LOW"
            elif any(k in mf for k in ("LIGHTGBM", "HIST_GRADIENT", "BOOSTING", "TREE")):
                complexity_penalty = 0.005
                complexity_tier = "MEDIUM"
            else:
                complexity_penalty = 0.010
                complexity_tier = "HIGH"

            # Penalty calculations
            variance_penalty = std_val * self.std_penalty_weight
            feature_penalty = res.feature_count * self.feature_penalty_weight
            latency_penalty = res.prediction_time_seconds * self.latency_penalty_weight

            composite_score = norm_score - variance_penalty - feature_penalty - latency_penalty - complexity_penalty

            scored_candidates.append({
                "evaluation": res,
                "raw_score": raw_score,
                "norm_score": norm_score,
                "std_val": std_val,
                "complexity_tier": complexity_tier,
                "composite_score": composite_score,
            })

        # 3. Find top raw score candidate
        max_raw_cand = max(scored_candidates, key=lambda x: x["norm_score"])
        max_norm_score = max_raw_cand["norm_score"]

        # 4. Identify top-tier candidates within simplicity threshold
        eligible_candidates = [
            cand for cand in scored_candidates
            if round(max_norm_score - cand["norm_score"], 6) <= round(threshold, 6)
        ]

        # 5. Pick winner with highest composite score from eligible candidates
        winner_item = max(
            eligible_candidates,
            key=lambda x: (x["composite_score"], -x["evaluation"].feature_count, x["evaluation"].pipeline_id)
        )
        winner_eval: PipelineEvaluationResult = winner_item["evaluation"]

        # Sort remaining candidates as alternatives
        alternatives: List[Dict[str, Any]] = []
        for item in sorted(scored_candidates, key=lambda x: x["composite_score"], reverse=True):
            cand_eval: PipelineEvaluationResult = item["evaluation"]
            if cand_eval.pipeline_id != winner_eval.pipeline_id:
                alternatives.append({
                    "pipeline_id": cand_eval.pipeline_id,
                    "pipeline_name": cand_eval.pipeline_name,
                    "model_family": cand_eval.model_family,
                    "score": cand_eval.primary_score,
                    "std": item["std_val"],
                    "composite_score": item["composite_score"],
                    "feature_count": cand_eval.feature_count,
                })

        # 6. Construct Selection Reason & Tradeoff Analysis
        is_simpler_winner = winner_eval.pipeline_id != max_raw_cand["evaluation"].pipeline_id
        if is_simpler_winner:
            raw_diff = abs(winner_item["raw_score"] - max_raw_cand["raw_score"])
            reason = (
                f"Selected simpler pipeline '{winner_eval.pipeline_name}' ({winner_eval.model_family}) "
                f"because score difference ({raw_diff:.4f}) is within simplicity threshold ({threshold:.4f}) "
                f"while providing lower complexity/variance."
            )
        else:
            reason = (
                f"Selected pipeline '{winner_eval.pipeline_name}' ({winner_eval.model_family}) "
                f"as the clear winner with optimal composite score ({winner_item['composite_score']:.4f}) "
                f"and primary metric {metric}={winner_item['raw_score']:.4f}."
            )

        # 7. Confidence Score Calculation
        if len(scored_candidates) > 1:
            runner_up_comp = max(
                [c["composite_score"] for c in scored_candidates if c["evaluation"].pipeline_id != winner_eval.pipeline_id]
            )
            comp_margin = max(0.0, winner_item["composite_score"] - runner_up_comp)
            confidence = float(np.clip(0.70 + (comp_margin * 5.0) - (winner_item["std_val"] * 0.5), 0.50, 0.99))
        else:
            confidence = 0.90

        tradeoffs = {
            "is_simpler_pipeline_chosen": is_simpler_winner,
            "raw_metric_winner_id": max_raw_cand["evaluation"].pipeline_id,
            "score_difference_from_top": abs(winner_item["raw_score"] - max_raw_cand["raw_score"]),
            "winner_std": winner_item["std_val"],
            "winner_feature_count": winner_eval.feature_count,
            "winner_complexity_tier": winner_item["complexity_tier"],
            "simplicity_threshold_used": threshold,
        }

        result = BestPipelineResult(
            winner_pipeline_id=winner_eval.pipeline_id,
            winner_pipeline_name=winner_eval.pipeline_name,
            winner_model_family=winner_eval.model_family,
            metric=metric,
            score=winner_eval.primary_score,
            confidence=round(confidence, 4),
            selection_reason=reason,
            alternatives=alternatives,
            tradeoffs=tradeoffs,
            winner_evaluation=winner_eval,
            metadata={
                "dataset_name": report.dataset_name,
                "problem_type": report.problem_type,
                "total_candidates": len(report.evaluation_results),
            },
        )

        logger.info(f"Final pipeline selected: '{winner_eval.pipeline_id}' ({winner_eval.model_family}, {metric}={winner_eval.primary_score:.4f}, confidence={confidence:.2f}).")
        return result
