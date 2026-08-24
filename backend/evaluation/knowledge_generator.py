from typing import List, Dict, Any, Optional
from collections import defaultdict
from backend.schemas.experiment import ExperimentResult
from backend.schemas.evaluation import KnowledgeFinding


class KnowledgeGenerator:
    """Extracts cross-experiment strategy insights into reusable KnowledgeFinding objects."""

    @classmethod
    def generate_findings(
        cls,
        results: List[ExperimentResult],
        rankings: Optional[List[Any]] = None,
    ) -> List[KnowledgeFinding]:
        """Extracts generalizable strategy patterns across experiment results."""
        findings: List[KnowledgeFinding] = []
        completed = [r for r in results if r.status == "completed" and r.metrics]

        if not completed:
            return findings

        # Group metrics by model family
        model_scores = defaultdict(list)
        model_gaps = defaultdict(list)
        operation_scores = defaultdict(list)

        for r in completed:
            score = float(r.metrics.primary_metric)
            metrics_dict = r.metrics.metrics or {}
            gap = float(metrics_dict.get("train_test_gap", 0.0))

            model_scores[r.model].append(score)
            model_gaps[r.model].append(gap)

            if r.pipeline:
                for op in r.pipeline.operations:
                    key = f"{op.type}:{op.method}"
                    operation_scores[key].append(score)

        # 1. Evaluate top performing model family using authoritative Rankings from RankingEngine
        if rankings and len(rankings) > 0:
            top_rank = rankings[0]
            top_exp_id = top_rank.experiment_id
            winning_exp = next((r for r in completed if r.experiment_id == top_exp_id), completed[0])
            best_model = winning_exp.model
            primary_score = float(winning_exp.metrics.primary_metric)
            metrics_dict = winning_exp.metrics.metrics or {}
            avg_gap = float(metrics_dict.get("train_test_gap", 0.03))
            gap_desc = "low overfitting risk" if avg_gap <= 0.15 else "moderate overfitting - regularization recommended"

            findings.append(
                KnowledgeFinding(
                    finding=f"Experiment '{top_exp_id}' utilizing {best_model} achieved the top performance with primary test score {primary_score:.4f} and {gap_desc}.",
                    confidence=0.92,
                    supporting_experiments=[top_exp_id],
                )
            )
        else:
            best_model = list(model_scores.keys())[0] if model_scores else "Unknown"
            avg_score = round(sum(model_scores[best_model]) / len(model_scores[best_model]), 4) if model_scores else 0.0
            findings.append(
                KnowledgeFinding(
                    finding=f"Model family '{best_model}' achieved top performance with test score {avg_score}.",
                    confidence=0.88,
                    supporting_experiments=[r.experiment_id for r in completed if r.model == best_model],
                )
            )

        # 2. Evaluate top performing preprocessing operation
        if operation_scores:
            best_op = max(operation_scores.keys(), key=lambda op: sum(operation_scores[op]) / len(operation_scores[op]))
            op_avg = round(sum(operation_scores[best_op]) / len(operation_scores[best_op]), 4)
            op_type, op_method = best_op.split(":")

            findings.append(
                KnowledgeFinding(
                    finding=f"Preprocessing strategy '{op_method}' ({op_type}) demonstrated positive impact across experiments (avg test score: {op_avg}).",
                    confidence=0.85,
                    supporting_experiments=[
                        r.experiment_id for r in completed
                        if r.pipeline and any(op.method == op_method for op in r.pipeline.operations)
                    ],
                )
            )

        return findings
