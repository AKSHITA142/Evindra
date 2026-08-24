import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.services.rag.hybrid_retrieval_service import HybridRetrievalService
from backend.services.rag.reranker_service import ScenarioRerankerService
from backend.services.rag.context_builder import RAGContextBuilder
from backend.services.rag.decision_service import LLMDecisionService
from backend.services.rag.recommendation_validator import RecommendationValidatorService

logger = logging.getLogger("datapilot.rag.pipeline_planner")


class PreprocessingPlanStep(BaseModel):
    """
    Individual execution step within an Evindra Preprocessing Plan (Phase F).
    """
    step_id: int = Field(..., description="1-indexed step identifier in execution order")
    target_column: str = Field(..., description="Target feature column name to transform")
    issue_type: str = Field(..., description="Type of data issue (e.g. missing_value, categorical_encoding)")
    recommended_action: str = Field(..., description="Primary recommended transformation action (e.g. IMPUTE_MEDIAN, ONE_HOT_ENCODING)")
    confidence_score: float = Field(..., description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(..., description="Step reasoning explicitly citing evidence scenario IDs")
    evidence_scenario_ids: List[str] = Field(default_factory=list, description="List of cited evidence scenario_ids")
    validation_status: str = Field(..., description="Validation status: PASSED, WARNING, or FAILED")
    transformation_spec: Dict[str, Any] = Field(default_factory=dict, description="Executable transformation specification")


class EvindraPreprocessingPlan(BaseModel):
    """
    Complete machine-readable, execution-ready Preprocessing Plan output by the Evindra RAG system.
    """
    plan_id: str = Field(..., description="Unique plan identifier")
    dataset_name: str = Field(..., description="Target dataset filename/identifier")
    problem_type: str = Field(..., description="ML problem type (e.g. regression, binary_classification)")
    created_at: str = Field(..., description="Plan creation ISO 8601 timestamp")
    steps: List[PreprocessingPlanStep] = Field(default_factory=list, description="Ordered list of transformation steps")
    pipeline_validation: Dict[str, Any] = Field(default_factory=dict, description="Aggregated pipeline validation status & estimated improvement")

    def to_dict(self) -> Dict[str, Any]:
        """Converts plan to dictionary representation."""
        return self.model_dump()

    def to_json_str(self, indent: int = 2) -> str:
        """Converts plan to formatted JSON string."""
        return json.dumps(self.model_dump(), indent=indent)


class EvindraPipelinePlanner:
    """
    End-to-End Pipeline Planner for Evindra RAG System (Phase F).
    Orchestrates Vector Search -> Hybrid Retrieval -> Multi-Factor Reranking -> Context Builder ->
    LLM Decision Layer -> Recommendation Validation into a single execution-ready Preprocessing Plan.

    Treats existing scenarios and embeddings as READ-ONLY.
    """

    def __init__(
        self,
        hybrid_service: Optional[HybridRetrievalService] = None,
        reranker_service: Optional[ScenarioRerankerService] = None,
        context_builder: Optional[RAGContextBuilder] = None,
        decision_service: Optional[LLMDecisionService] = None,
        validator_service: Optional[RecommendationValidatorService] = None,
    ):
        self.hybrid_service = hybrid_service or HybridRetrievalService()
        self.reranker_service = reranker_service or ScenarioRerankerService(hybrid_retrieval_service=self.hybrid_service)
        self.context_builder = context_builder or RAGContextBuilder()
        self.decision_service = decision_service or LLMDecisionService()
        self.validator_service = validator_service or RecommendationValidatorService()

    def generate_evindra_preprocessing_plan(
        self,
        dataset_profile: Dict[str, Any],
        column_issues: List[Dict[str, Any]],
        top_k_evidence: int = 3,
    ) -> EvindraPreprocessingPlan:
        """
        Generates an execution-ready Preprocessing Plan for a dataset profile across multiple column issues.

        Args:
            dataset_profile: Dictionary containing global dataset facts:
                - dataset_name: str
                - problem_type: str (e.g. "regression", "binary_classification")
                - target_column: Optional[str]
            column_issues: List of column-level issue dictionaries:
                - column: str (feature name)
                - issue_type: str (e.g. "missing_value", "categorical_encoding")
                - domain: str (e.g. "missing_value_strategy", "encoding_strategy")
                - column_type: str (e.g. "numeric", "categorical")
                - severity: Optional[str] (e.g. "high", "medium")
                - query_text: Optional[str] (natural language query)
            top_k_evidence: Number of top evidence scenarios to include per step.

        Returns:
            EvindraPreprocessingPlan object containing ordered execution steps and validation summary.
        """
        dataset_name = dataset_profile.get("dataset_name", "dataset.csv")
        problem_type = dataset_profile.get("problem_type", "general_tabular")

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sanitized_name = dataset_name.replace(".csv", "").replace(" ", "_").lower()
        plan_id = f"plan_{sanitized_name}_{timestamp_str}"

        steps: List[PreprocessingPlanStep] = []
        total_improvement = 0.0
        passed_steps_count = 0
        failed_steps_count = 0

        logger.info(f"Generating Evindra Preprocessing Plan '{plan_id}' for {len(column_issues)} column issues...")

        for idx, issue in enumerate(column_issues, 1):
            col_name = issue.get("column", f"column_{idx}")
            issue_type = issue.get("issue_type", "preprocessing")
            domain = issue.get("domain", f"{issue_type}_strategy")
            col_type = issue.get("column_type", "numeric")
            severity = issue.get("severity", "medium")

            query_text = issue.get("query_text") or (
                f"{col_type.capitalize()} feature '{col_name}' with {issue_type} in a {problem_type} dataset. "
                f"Need an appropriate {domain} preprocessing transformation strategy."
            )

            context_dict = {
                "domain": domain,
                "scenario_type": issue_type,
                "problem_type": problem_type,
                "column_type": col_type,
                "severity": severity,
            }

            issue_profile = {
                "dataset_name": dataset_name,
                "problem_type": problem_type,
                "target_column": dataset_profile.get("target_column"),
                "target_feature": col_name,
                "feature_dtype": col_type,
                "issue_description": f"{issue_type} on column '{col_name}'.",
            }

            # Phase 1 & A & B: Retrieval + Hybrid Search + Multi-Factor Reranking
            reranked_scenarios = self.reranker_service.retrieve_and_rerank(
                query_text=query_text,
                context=context_dict,
                top_k=top_k_evidence,
            )

            # Phase C: RAG Context Builder
            evidence_package = self.context_builder.build_evidence_package(
                dataset_profile=issue_profile,
                retrieved_scenarios=reranked_scenarios,
                max_evidence_count=top_k_evidence,
            )

            # Phase D: LLM Decision Layer
            recommendation = self.decision_service.generate_preprocessing_recommendation(
                dataset_profile=issue_profile,
                evidence_package=evidence_package,
            )

            # Phase E: Recommendation Validation Layer
            val_report = self.validator_service.validate_recommendation(
                dataset_profile=issue_profile,
                recommendation=recommendation,
                evidence_package=evidence_package,
            )

            # Track metrics
            if val_report.validation_status in ("PASSED", "WARNING"):
                passed_steps_count += 1
            else:
                failed_steps_count += 1

            imp = val_report.experimental_validation.get("improvement", 0.0)
            total_improvement += float(imp)

            step = PreprocessingPlanStep(
                step_id=idx,
                target_column=col_name,
                issue_type=issue_type,
                recommended_action=recommendation.primary_recommendation,
                confidence_score=recommendation.confidence_score,
                reasoning=recommendation.reasoning,
                evidence_scenario_ids=recommendation.evidence_scenarios,
                validation_status=val_report.validation_status,
                transformation_spec=val_report.executable_pipeline_spec,
            )
            steps.append(step)

        # Aggregate Pipeline Validation
        overall_status = "PASSED" if failed_steps_count == 0 else "FAILED"
        avg_improvement = round(total_improvement / len(column_issues), 3) if column_issues else 0.0

        pipeline_val = {
            "overall_status": overall_status,
            "total_steps": len(steps),
            "passed_steps": passed_steps_count,
            "failed_steps": failed_steps_count,
            "estimated_overall_improvement": avg_improvement,
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"Evindra Preprocessing Plan '{plan_id}' successfully generated with status '{overall_status}'.")

        return EvindraPreprocessingPlan(
            plan_id=plan_id,
            dataset_name=dataset_name,
            problem_type=problem_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            steps=steps,
            pipeline_validation=pipeline_val,
        )


# Convenience function for direct module usage
def generate_evindra_preprocessing_plan(
    dataset_profile: Dict[str, Any],
    column_issues: List[Dict[str, Any]],
    top_k_evidence: int = 3,
) -> EvindraPreprocessingPlan:
    """Convenience wrapper around EvindraPipelinePlanner.generate_evindra_preprocessing_plan."""
    planner = EvindraPipelinePlanner()
    return planner.generate_evindra_preprocessing_plan(dataset_profile, column_issues, top_k_evidence=top_k_evidence)
