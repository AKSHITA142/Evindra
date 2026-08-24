"""
RAG service package — all imports are lazy so that optional cloud dependencies
(google-genai, supabase pgvector) do not break the engine when unavailable.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.services.rag.embedding_service import EmbeddingService
    from backend.services.rag.retrieval_service import VectorRetrievalService, search_similar_scenarios
    from backend.services.rag.hybrid_retrieval_service import HybridRetrievalService, retrieve_relevant_scenarios
    from backend.services.rag.reranker_service import (
        ScenarioRerankerService,
        rerank_scenarios,
        retrieve_and_rerank_scenarios,
    )
    from backend.services.rag.context_builder import (
        RAGContextBuilder,
        RAGEvidencePackage,
        build_rag_evidence_package,
    )
    from backend.services.rag.decision_service import (
        LLMDecisionService,
        PreprocessingRecommendation,
        generate_preprocessing_recommendation,
    )
    from backend.services.rag.recommendation_validator import (
        RecommendationValidatorService,
        RecommendationValidationReport,
        validate_recommendation,
    )
    from backend.services.rag.pipeline_planner import (
        EvindraPipelinePlanner,
        EvindraPreprocessingPlan,
        PreprocessingPlanStep,
        generate_evindra_preprocessing_plan,
    )

_MODULE_MAP = {
    "EmbeddingService": ("backend.services.rag.embedding_service", "EmbeddingService"),
    "VectorRetrievalService": ("backend.services.rag.retrieval_service", "VectorRetrievalService"),
    "search_similar_scenarios": ("backend.services.rag.retrieval_service", "search_similar_scenarios"),
    "HybridRetrievalService": ("backend.services.rag.hybrid_retrieval_service", "HybridRetrievalService"),
    "retrieve_relevant_scenarios": ("backend.services.rag.hybrid_retrieval_service", "retrieve_relevant_scenarios"),
    "ScenarioRerankerService": ("backend.services.rag.reranker_service", "ScenarioRerankerService"),
    "rerank_scenarios": ("backend.services.rag.reranker_service", "rerank_scenarios"),
    "retrieve_and_rerank_scenarios": ("backend.services.rag.reranker_service", "retrieve_and_rerank_scenarios"),
    "RAGContextBuilder": ("backend.services.rag.context_builder", "RAGContextBuilder"),
    "RAGEvidencePackage": ("backend.services.rag.context_builder", "RAGEvidencePackage"),
    "build_rag_evidence_package": ("backend.services.rag.context_builder", "build_rag_evidence_package"),
    "LLMDecisionService": ("backend.services.rag.decision_service", "LLMDecisionService"),
    "PreprocessingRecommendation": ("backend.services.rag.decision_service", "PreprocessingRecommendation"),
    "generate_preprocessing_recommendation": ("backend.services.rag.decision_service", "generate_preprocessing_recommendation"),
    "RecommendationValidatorService": ("backend.services.rag.recommendation_validator", "RecommendationValidatorService"),
    "RecommendationValidationReport": ("backend.services.rag.recommendation_validator", "RecommendationValidationReport"),
    "validate_recommendation": ("backend.services.rag.recommendation_validator", "validate_recommendation"),
    "EvindraPipelinePlanner": ("backend.services.rag.pipeline_planner", "EvindraPipelinePlanner"),
    "EvindraPreprocessingPlan": ("backend.services.rag.pipeline_planner", "EvindraPreprocessingPlan"),
    "PreprocessingPlanStep": ("backend.services.rag.pipeline_planner", "PreprocessingPlanStep"),
    "generate_evindra_preprocessing_plan": ("backend.services.rag.pipeline_planner", "generate_evindra_preprocessing_plan"),
}

__all__ = list(_MODULE_MAP.keys())


def __getattr__(name: str):
    if name in _MODULE_MAP:
        module_path, attr = _MODULE_MAP[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
