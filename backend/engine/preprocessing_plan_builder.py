import logging
from typing import Optional, Dict, Any, List

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.decision import DecisionDomain, DecisionResult, DecisionSource
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan
from backend.engine.decision_orchestrator import DecisionOrchestrator

logger = logging.getLogger("datapilot.engine.plan_builder")


class PreprocessingPlanBuilder:
    """
    Preprocessing Plan Builder for Evindra Preprocessing Pipeline (Phase 8).
    Aggregates column-level and dataset-level decisions produced by the DecisionOrchestrator
    into a structured, ordered, DAG-aligned PreprocessingPlan.
    """

    def __init__(self, orchestrator: Optional[DecisionOrchestrator] = None):
        self.orchestrator = orchestrator or DecisionOrchestrator()

    def build_plan(
        self,
        dataset_profile: DatasetProfile,
        model_family: str = "general",
        user_mission: str = "",
        user_selections: Optional[Dict[str, str]] = None,
        auto_approve_default: bool = True,
    ) -> PreprocessingPlan:
        """
        Builds a complete PreprocessingPlan from a DatasetProfile using the DecisionOrchestrator.

        Args:
            dataset_profile: Unified DatasetProfile containing column and dataset statistics.
            model_family: Model family hint (e.g., "linear", "tree", "neural_network", "general").
            user_mission: Free-text user mission brief.
            user_selections: Optional dict mapping column/domain keys to user override selections.
            auto_approve_default: Whether to auto-approve safe defaults if LLM/RAG/Rule fall through.

        Returns:
            Structured PreprocessingPlan object.
        """
        user_selections = user_selections or {}
        steps: List[PreprocessingStep] = []
        step_counter = 1
        decision_results: List[DecisionResult] = []

        cols = dataset_profile.detailed_column_profiles or dataset_profile.column_profiles or []
        col_map = {col.name: col for col in cols}

        # -------------------------------------------------------------
        # Phase 8.1: Column-Level Decisions (Missing, Encoding, Scaling, Outliers)
        # -------------------------------------------------------------
        for col_name, col_prof in col_map.items():
            # Coerce col_prof to ColumnProfileExtended if standard ColumnProfile
            if not isinstance(col_prof, ColumnProfileExtended):
                col_prof = ColumnProfileExtended(
                    name=col_prof.name,
                    normalized_dtype=col_prof.type.value if hasattr(col_prof.type, "value") else str(col_prof.type),
                    missing_ratio=col_prof.missing_pct / 100.0 if hasattr(col_prof, "missing_pct") else 0.0,
                    distinct_count=col_prof.distinct_count if hasattr(col_prof, "distinct_count") else 0,
                )
            # 1. Missing Value Strategy
            if col_prof.missing_ratio > 0:
                user_sel = user_selections.get(f"{col_name}:missing_value_strategy")
                res = self.orchestrator.evaluate_decision(
                    domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                    col_profile=col_prof,
                    dataset_profile=dataset_profile,
                    model_family=model_family,
                    user_mission=user_mission,
                    user_selection=user_sel,
                    auto_approve_default=auto_approve_default,
                )
                decision_results.append(res)
                if res.decision not in ("PASS_THROUGH", "NONE"):
                    steps.append(
                        PreprocessingStep(
                            step_number=step_counter,
                            domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                            action=res.decision,
                            columns=[col_name],
                            params={"missing_ratio": col_prof.missing_ratio},
                            decision_id=res.decision_id,
                            decision_source=res.source,
                            confidence=res.confidence,
                            reasoning=res.reasoning,
                            requires_validation=res.requires_validation,
                            metadata=res.metadata,
                        )
                    )
                    step_counter += 1

            # 2. Encoding Strategy (Categorical / Text)
            if col_prof.normalized_dtype in ("categorical", "text") or col_prof.distinct_count < 20:
                user_sel = user_selections.get(f"{col_name}:encoding_strategy")
                res = self.orchestrator.evaluate_decision(
                    domain=DecisionDomain.ENCODING_STRATEGY,
                    col_profile=col_prof,
                    dataset_profile=dataset_profile,
                    model_family=model_family,
                    user_mission=user_mission,
                    user_selection=user_sel,
                    auto_approve_default=auto_approve_default,
                )
                decision_results.append(res)
                if res.decision not in ("PASS_THROUGH", "NONE"):
                    steps.append(
                        PreprocessingStep(
                            step_number=step_counter,
                            domain=DecisionDomain.ENCODING_STRATEGY,
                            action=res.decision,
                            columns=[col_name],
                            params={"distinct_count": col_prof.distinct_count},
                            decision_id=res.decision_id,
                            decision_source=res.source,
                            confidence=res.confidence,
                            reasoning=res.reasoning,
                            requires_validation=res.requires_validation,
                            metadata=res.metadata,
                        )
                    )
                    step_counter += 1

            # 3. Scaling & Transformation (Numeric)
            if col_prof.normalized_dtype == "numeric":
                user_sel = user_selections.get(f"{col_name}:scaling_transformation")
                res = self.orchestrator.evaluate_decision(
                    domain=DecisionDomain.SCALING_TRANSFORMATION,
                    col_profile=col_prof,
                    dataset_profile=dataset_profile,
                    model_family=model_family,
                    user_mission=user_mission,
                    user_selection=user_sel,
                    auto_approve_default=auto_approve_default,
                )
                decision_results.append(res)
                if res.decision not in ("PASS_THROUGH", "NONE"):
                    steps.append(
                        PreprocessingStep(
                            step_number=step_counter,
                            domain=DecisionDomain.SCALING_TRANSFORMATION,
                            action=res.decision,
                            columns=[col_name],
                            params={"skewness": col_prof.skewness},
                            decision_id=res.decision_id,
                            decision_source=res.source,
                            confidence=res.confidence,
                            reasoning=res.reasoning,
                            requires_validation=res.requires_validation,
                            metadata=res.metadata,
                        )
                    )
                    step_counter += 1

            # 4. Outlier Handling (Numeric with outliers)
            if col_prof.normalized_dtype == "numeric" and col_prof.outlier_ratio > 0.01:
                user_sel = user_selections.get(f"{col_name}:outlier_handling")
                res = self.orchestrator.evaluate_decision(
                    domain=DecisionDomain.OUTLIER_HANDLING,
                    col_profile=col_prof,
                    dataset_profile=dataset_profile,
                    model_family=model_family,
                    user_mission=user_mission,
                    user_selection=user_sel,
                    auto_approve_default=auto_approve_default,
                )
                decision_results.append(res)
                if res.decision not in ("PASS_THROUGH", "NONE"):
                    steps.append(
                        PreprocessingStep(
                            step_number=step_counter,
                            domain=DecisionDomain.OUTLIER_HANDLING,
                            action=res.decision,
                            columns=[col_name],
                            params={"outlier_ratio": col_prof.outlier_ratio},
                            decision_id=res.decision_id,
                            decision_source=res.source,
                            confidence=res.confidence,
                            reasoning=res.reasoning,
                            requires_validation=res.requires_validation,
                            metadata=res.metadata,
                        )
                    )
                    step_counter += 1

        # -------------------------------------------------------------
        # Phase 8.2: Dataset-Level Decisions (Feature Selection)
        # -------------------------------------------------------------
        user_sel = user_selections.get("feature_selection")
        res_fs = self.orchestrator.evaluate_decision(
            domain=DecisionDomain.FEATURE_SELECTION,
            dataset_profile=dataset_profile,
            model_family=model_family,
            user_mission=user_mission,
            user_selection=user_sel,
            auto_approve_default=auto_approve_default,
        )
        decision_results.append(res_fs)

        if res_fs.decision not in ("PASS_THROUGH", "NONE"):
            steps.append(
                PreprocessingStep(
                    step_number=step_counter,
                    domain=DecisionDomain.FEATURE_SELECTION,
                    action=res_fs.decision,
                    columns=list(col_map.keys()),
                    params={"duplicate_columns": getattr(dataset_profile, "duplicate_column_pairs", [])},
                    decision_id=res_fs.decision_id,
                    decision_source=res_fs.source,
                    confidence=res_fs.confidence,
                    reasoning=res_fs.reasoning,
                    requires_validation=res_fs.requires_validation,
                    metadata=res_fs.metadata,
                )
            )

        # Calculate overall confidence
        conf_scores = [d.confidence for d in decision_results if d.confidence is not None]
        overall_conf = round(sum(conf_scores) / len(conf_scores), 4) if conf_scores else 1.0

        target_col = getattr(dataset_profile, "target_column", None) or dataset_profile.dataset_summary.get("target", {}).get("target_column")
        prob_type = getattr(dataset_profile, "problem_type", None) or dataset_profile.dataset_summary.get("target", {}).get("task_type", "general_tabular")

        plan = PreprocessingPlan(
            dataset_name=dataset_profile.dataset_name,
            target_column=target_col,
            problem_type=prob_type,
            model_family=model_family,
            steps=steps,
            decisions_summary={
                "total_decisions": len(decision_results),
                "total_steps": len(steps),
                "sources_breakdown": {
                    "rule": len([d for d in decision_results if d.source == DecisionSource.RULE]),
                    "rag": len([d for d in decision_results if d.source == DecisionSource.RAG]),
                    "llm": len([d for d in decision_results if d.source == DecisionSource.LLM]),
                    "user": len([d for d in decision_results if d.source == DecisionSource.USER]),
                },
            },
            overall_confidence=overall_conf,
        )

        logger.info(f"Built PreprocessingPlan ({plan.plan_id}) with {len(steps)} steps for dataset '{dataset_profile.dataset_name}'.")
        return plan
