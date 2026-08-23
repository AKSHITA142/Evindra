import logging
from typing import List, Dict, Any, Optional

from backend.services.rag.hybrid_retrieval_service import HybridRetrievalService

logger = logging.getLogger("datapilot.rag.reranker")


class ScenarioRerankerService:
    """
    Scenario Reranker Service for Evindra RAG System (Phase B).
    Applies multi-signal reranking combining semantic similarity, structured metadata match,
    and historical scenario validation/quality scores. Produces transparent rank explanations.

    Treats existing scenarios and embeddings as READ-ONLY.
    """

    def __init__(
        self,
        hybrid_retrieval_service: Optional[HybridRetrievalService] = None,
        default_w_semantic: float = 0.50,
        default_w_structured: float = 0.35,
        default_w_validation: float = 0.15,
    ):
        self.hybrid_service = hybrid_retrieval_service or HybridRetrievalService()
        self.w_semantic = default_w_semantic
        self.w_structured = default_w_structured
        self.w_validation = default_w_validation

    def rerank_scenarios(
        self,
        candidates: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved candidate scenarios using multi-factor scoring.

        Args:
            candidates: List of candidate scenario dictionaries from HybridRetrievalService or VectorRetrievalService.
            weights: Optional custom dictionary of weights:
                - semantic: float (default: 0.50)
                - structured: float (default: 0.35)
                - validation: float (default: 0.15)

        Returns:
            List of reranked candidate scenarios containing:
            - scenario_id
            - domain
            - scenario_type
            - semantic_score
            - structured_score
            - validation_score
            - final_score
            - rank_explanation
            - retrieval_text
            - metadata
        """
        if not candidates:
            return []

        w_sem = self.w_semantic
        w_struct = self.w_structured
        w_val = self.w_validation

        if weights:
            w_sem = weights.get("semantic", w_sem)
            w_struct = weights.get("structured", w_struct)
            w_val = weights.get("validation", w_val)

        # Normalize weights to sum to 1.0
        total_w = w_sem + w_struct + w_val
        if total_w > 0:
            w_sem = w_sem / total_w
            w_struct = w_struct / total_w
            w_val = w_val / total_w

        reranked: List[Dict[str, Any]] = []
        for cand in candidates:
            sem_score = float(cand.get("semantic_score", cand.get("similarity_score", 0.0)))
            struct_score = float(cand.get("structured_score", 0.5))
            val_score = self._compute_validation_score(cand)

            final_score = (w_sem * sem_score) + (w_struct * struct_score) + (w_val * val_score)

            explanation = self._generate_rank_explanation(sem_score, struct_score, val_score, cand)

            reranked.append({
                "scenario_id": cand.get("scenario_id"),
                "domain": cand.get("domain"),
                "scenario_type": cand.get("scenario_type"),
                "semantic_score": round(sem_score, 6),
                "structured_score": round(struct_score, 6),
                "validation_score": round(val_score, 6),
                "final_score": round(final_score, 6),
                "rank_explanation": explanation,
                "retrieval_text": cand.get("retrieval_text"),
                "metadata": cand.get("metadata", {}),
            })

        # Sort descending by final score
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked

    def retrieve_and_rerank(
        self,
        query_text: str,
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method: runs hybrid retrieval with oversampling, then applies multi-factor reranking.
        """
        # Oversample pool for reranking
        pool_size = max(top_k * 4, 20)
        hybrid_candidates = self.hybrid_service.retrieve_relevant_scenarios(
            query_text=query_text,
            context=context,
            top_k=pool_size,
        )
        reranked = self.rerank_scenarios(hybrid_candidates, weights=weights)
        return reranked[:top_k]

    def _compute_validation_score(self, candidate: Dict[str, Any]) -> float:
        """
        Computes a normalized validation & quality score (0.0 to 1.0) based on historical
        candidate quality metrics and validation status.
        """
        metadata = candidate.get("metadata", {}) or {}
        val_info = metadata.get("validation", {}) or {}
        ans_key = metadata.get("answer_key", {}) or {}

        score = 0.5  # Baseline

        # Signal 1: Raw candidate_quality_score (typically 1.0 to 5.0)
        quality_raw = metadata.get("candidate_quality_score")
        if quality_raw is not None:
            try:
                score = min(float(quality_raw) / 5.0, 1.0)
            except (ValueError, TypeError):
                pass

        # Signal 2: Validation status bonus
        val_status = str(val_info.get("status") or "").upper()
        if val_status in ("RULE_VALIDATED", "EXPERIMENT_VALIDATED", "BENCHMARK_VALIDATED"):
            score = min(score + 0.15, 1.0)
        elif val_status == "FAILED":
            score = max(score - 0.3, 0.0)

        # Signal 3: Deterministic answer key bonus
        ans_source = str(ans_key.get("answer_source") or "").upper()
        if ans_source == "DETERMINISTIC":
            score = min(score + 0.1, 1.0)

        return round(min(max(score, 0.0), 1.0), 4)

    def _generate_rank_explanation(
        self, sem_score: float, struct_score: float, val_score: float, candidate: Dict[str, Any]
    ) -> str:
        """Generates a concise, human-readable rank explanation for transparency."""
        sem_desc = "Strong" if sem_score >= 0.75 else ("Moderate" if sem_score >= 0.65 else "Low")
        struct_desc = "perfect" if struct_score >= 0.95 else ("strong" if struct_score >= 0.70 else "partial")
        val_desc = "high validated quality" if val_score >= 0.80 else "standard quality"

        scenario_id = candidate.get("scenario_id", "Scenario")
        return (
            f"{sem_desc} semantic similarity ({sem_score:.4f}), {struct_desc} structured context match "
            f"({struct_score:.4f}), and {val_desc} ({val_score:.4f}) for {scenario_id}."
        )


# Convenience function for direct module usage
def rerank_scenarios(
    candidates: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper around ScenarioRerankerService.rerank_scenarios."""
    service = ScenarioRerankerService()
    return service.rerank_scenarios(candidates, weights=weights)


def retrieve_and_rerank_scenarios(
    query_text: str,
    context: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper around ScenarioRerankerService.retrieve_and_rerank."""
    service = ScenarioRerankerService()
    return service.retrieve_and_rerank(query_text, context=context, top_k=top_k, weights=weights)
