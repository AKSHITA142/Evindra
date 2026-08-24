import logging
from typing import List, Dict, Any, Optional

from backend.services.rag.retrieval_service import VectorRetrievalService

logger = logging.getLogger("datapilot.rag.hybrid_retrieval")


class HybridRetrievalService:
    """
    Hybrid Retrieval Service for Evindra RAG System (Phase A).
    Combines semantic vector search with structured metadata matching (domain, scenario_type,
    problem_type, column_type, severity) to retrieve optimal context scenarios.
    
    Treats existing scenarios and embeddings as READ-ONLY.
    """

    def __init__(
        self,
        vector_retrieval_service: Optional[VectorRetrievalService] = None,
        default_vector_weight: float = 0.6,
        default_structured_weight: float = 0.4,
    ):
        self.vector_service = vector_retrieval_service or VectorRetrievalService()
        self.default_vector_weight = default_vector_weight
        self.default_structured_weight = default_structured_weight

    def retrieve_relevant_scenarios(
        self,
        query_text: str,
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        vector_weight: Optional[float] = None,
        structured_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-K scenarios using a hybrid scoring model (semantic vector + structured match).

        Args:
            query_text: Natural language query describing ML problem/preprocessing situation.
            context: Optional dictionary containing structured dataset/problem characteristics:
                - domain: e.g. "missing_value_strategy", "encoding_strategy", "column_intelligence"
                - scenario_type: e.g. "missing_value", "categorical_encoding", "column_role"
                - problem_type: e.g. "regression", "binary_classification", "multiclass_classification"
                - column_type: e.g. "numeric", "categorical", "text_or_categorical"
                - severity: e.g. "high", "medium", "low"
            top_k: Final number of ranked candidate scenarios to return.
            vector_weight: Weight assigned to semantic vector similarity (default: 0.6).
            structured_weight: Weight assigned to structured metadata match (default: 0.4).

        Returns:
            List of ranked scenario dictionaries containing:
            - scenario_id
            - domain
            - scenario_type
            - semantic_score
            - structured_score
            - final_score (combined hybrid score)
            - retrieval_text
            - metadata
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty for hybrid retrieval.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k}")

        w_vec = vector_weight if vector_weight is not None else self.default_vector_weight
        w_struct = structured_weight if structured_weight is not None else self.default_structured_weight

        # Normalize weights to sum to 1.0
        total_w = w_vec + w_struct
        if total_w > 0:
            w_vec = w_vec / total_w
            w_struct = w_struct / total_w

        # Step 1: Fetch initial candidate pool via vector search (oversampling for hybrid filtering)
        pool_size = max(top_k * 4, 20)
        logger.info(
            f"Fetching initial candidate pool (size={pool_size}) via vector search..."
        )
        vector_candidates = self.vector_service.search_similar_scenarios(query_text, top_k=pool_size)

        # Step 2: Compute structured match score for each candidate
        ranked_results: List[Dict[str, Any]] = []
        for cand in vector_candidates:
            semantic_score = cand.get("similarity_score", 0.0)
            structured_score = self._compute_structured_score(cand, context)

            final_score = (w_vec * semantic_score) + (w_struct * structured_score)

            ranked_results.append({
                "scenario_id": cand.get("scenario_id"),
                "domain": cand.get("domain"),
                "scenario_type": cand.get("scenario_type"),
                "semantic_score": round(semantic_score, 6),
                "structured_score": round(structured_score, 6),
                "final_score": round(final_score, 6),
                "retrieval_text": cand.get("retrieval_text"),
                "metadata": cand.get("metadata", {}),
            })

        # Step 3: Sort candidates descending by final hybrid score
        ranked_results.sort(key=lambda x: x["final_score"], reverse=True)

        # Step 4: Return Top-K candidates
        top_results = ranked_results[:top_k]
        logger.info(
            f"Hybrid retrieval complete. Returning top {len(top_results)} scenarios (weights: vec={w_vec:.2f}, struct={w_struct:.2f})."
        )
        return top_results

    def _compute_structured_score(
        self, candidate: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Computes a normalized structured match score (0.0 to 1.0) by matching context keys
        against candidate domain, scenario_type, and metadata properties.
        """
        if not context:
            return 0.5  # Neutral score when no structured context is supplied

        metadata = candidate.get("metadata", {}) or {}
        routing = metadata.get("routing", {}) or {}
        
        matches = 0.0
        total_criteria = 0.0

        # Criterion 1: Domain Match
        target_domain = context.get("domain")
        if target_domain:
            total_criteria += 1.0
            cand_domain = (candidate.get("domain") or "").lower()
            routing_domain = (routing.get("primary_domain") or "").lower()
            if target_domain.lower() in (cand_domain, routing_domain):
                matches += 1.0
            elif target_domain.lower() in cand_domain or cand_domain in target_domain.lower():
                matches += 0.5

        # Criterion 2: Scenario Type / Family Match
        target_type = context.get("scenario_type")
        if target_type:
            total_criteria += 1.0
            cand_type = (candidate.get("scenario_type") or "").lower()
            family_type = (metadata.get("scenario_family") or metadata.get("source_family") or "").lower()
            if target_type.lower() in (cand_type, family_type):
                matches += 1.0
            elif target_type.lower() in cand_type or cand_type in target_type.lower():
                matches += 0.5

        # Criterion 3: Problem Type Match (e.g. regression vs classification)
        target_problem = context.get("problem_type")
        if target_problem:
            total_criteria += 1.0
            cand_problem = (metadata.get("problem_type") or "").lower()
            if target_problem.lower() == cand_problem:
                matches += 1.0
            elif target_problem.lower() in cand_problem or cand_problem in target_problem.lower():
                matches += 0.5

        # Criterion 4: Column Type / Dtype Match
        target_col_type = context.get("column_type")
        if target_col_type:
            total_criteria += 1.0
            cand_col_type = str(metadata.get("column_type") or "").lower()
            ret_text = (candidate.get("retrieval_text") or "").lower()
            if target_col_type.lower() in cand_col_type or f"dtype: {target_col_type.lower()}" in ret_text:
                matches += 1.0

        # Criterion 5: Severity Match
        target_severity = context.get("severity")
        if target_severity:
            total_criteria += 1.0
            cand_severity = str(metadata.get("severity") or "").lower()
            if target_severity.lower() == cand_severity:
                matches += 1.0

        if total_criteria == 0.0:
            return 0.5  # Neutral fallback if context dict contains no recognized keys

        return matches / total_criteria


# Convenience function for direct module usage
def retrieve_relevant_scenarios(
    query_text: str,
    context: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    vector_weight: float = 0.6,
    structured_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """Convenience wrapper around HybridRetrievalService.retrieve_relevant_scenarios."""
    service = HybridRetrievalService()
    return service.retrieve_relevant_scenarios(
        query_text=query_text,
        context=context,
        top_k=top_k,
        vector_weight=vector_weight,
        structured_weight=structured_weight,
    )
