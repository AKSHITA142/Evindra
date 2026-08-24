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
    Aggregates decisions produced by the DecisionOrchestrator into a structured,
    ordered, DAG-aligned PreprocessingPlan with explicit dependency tracking.
    """

    STAGE_ORDER = {
        "DATA_INGESTION": 1,
        "TARGET_SEPARATION": 2,
        "LEAKAGE_REMOVAL": 3,
        "TRAIN_TEST_SPLIT": 4,
        "MISSING_VALUE_HANDLING": 5,
        "ENCODING": 6,
        "SCALING": 7,
        "OUTLIER_TRANSFORMATION": 8,
        "FEATURE_ENGINEERING": 9,
        "FEATURE_SELECTION": 10,
        "MODEL": 11,
    }

    DOMAIN_STAGE_MAP = {
        DecisionDomain.COLUMN_INTELLIGENCE: "DATA_INGESTION",
        DecisionDomain.TARGET_DETECTION: "TARGET_SEPARATION",
        DecisionDomain.LEAKAGE_DETECTION: "LEAKAGE_REMOVAL",
        DecisionDomain.MISSING_VALUE_STRATEGY: "MISSING_VALUE_HANDLING",
        DecisionDomain.ENCODING_STRATEGY: "ENCODING",
        DecisionDomain.SCALING_TRANSFORMATION: "SCALING",
        DecisionDomain.OUTLIER_HANDLING: "OUTLIER_TRANSFORMATION",
        DecisionDomain.FEATURE_ENGINEERING: "FEATURE_ENGINEERING",
        DecisionDomain.FEATURE_SELECTION: "FEATURE_SELECTION",
        DecisionDomain.PIPELINE_STRATEGY: "TRAIN_TEST_SPLIT",
    }

    def __init__(self, orchestrator: Optional[DecisionOrchestrator] = None):
        self.orchestrator = orchestrator or DecisionOrchestrator()

    def build_plan(
        self,
        dataset_profile: DatasetProfile,
        model_family: str = "general",
        user_mission: str = "",
        user_selections: Optional[Dict[str, str]] = None,
        auto_approve_default: bool = True,
        decisions: Optional[List[DecisionResult]] = None,
    ) -> PreprocessingPlan:
        """
        Builds a complete, DAG-ordered PreprocessingPlan from a DatasetProfile.
        """
        user_selections = user_selections or {}
        raw_steps: List[PreprocessingStep] = []
        decision_results: List[DecisionResult] = []

        cols = dataset_profile.detailed_column_profiles or dataset_profile.column_profiles or []
        col_map = {col.name: col for col in cols}
        target_col = getattr(dataset_profile, "target_column", None) or dataset_profile.dataset_summary.get("target", {}).get("target_column")
        prob_type = getattr(dataset_profile, "problem_type", None) or dataset_profile.dataset_summary.get("target", {}).get("task_type", "general_tabular")

        # Input Schema
        input_schema = {c_name: c_prof.normalized_dtype for c_name, c_prof in col_map.items()}

        if decisions:
            # Use precomputed decisions list directly
            # Precompute column sets by dtype for proper action-column matching
            numeric_cols = [c for c in col_map.keys() if c != target_col and getattr(col_map[c], 'normalized_dtype', '') == 'numeric']
            categorical_cols = [c for c in col_map.keys() if c != target_col and getattr(col_map[c], 'normalized_dtype', '') in ('categorical', 'text')]
            all_feature_cols = [c for c in col_map.keys() if c != target_col]

            NUMERIC_ACTIONS = {"STANDARD_SCALER", "MINMAX_SCALER", "ROBUST_SCALER", "LOG_TRANSFORM",
                               "POWER_TRANSFORM", "CLIP_IQR", "WINZORIZE", "IMPUTE_MEAN", "IMPUTE_MEDIAN",
                               "IMPUTE_KNN", "IMPUTE_ZERO"}
            CATEGORICAL_ACTIONS = {"ONE_HOT_ENCODING", "TARGET_ENCODING", "TARGET_ENCODING_OUT_OF_FOLD",
                                   "ORDINAL_ENCODING", "FREQUENCY_ENCODING", "IMPUTE_MODE",
                                   "IMPUTE_EXPLICIT_CATEGORY"}

            step_num = 1
            for d in decisions:
                stage = self.DOMAIN_STAGE_MAP.get(d.domain, "MISSING_VALUE_HANDLING")
                action_upper = d.decision.upper()

                # Select columns based on action type
                if action_upper in NUMERIC_ACTIONS:
                    target_cols_list = numeric_cols[:5] if numeric_cols else all_feature_cols[:5]
                elif action_upper in CATEGORICAL_ACTIONS:
                    target_cols_list = categorical_cols[:5] if categorical_cols else all_feature_cols[:5]
                else:
                    target_cols_list = all_feature_cols[:5] if all_feature_cols else []

                if not target_cols_list and col_map:
                    target_cols_list = [list(col_map.keys())[0]]

                step = PreprocessingStep(
                    step_number=step_num,
                    stage=stage,
                    domain=d.domain,
                    action=d.decision,
                    columns=target_cols_list,
                    decision_id=d.decision_id,
                    decision_source=d.source,
                    confidence=d.confidence,
                    reasoning=d.reasoning,
                )
                raw_steps.append(step)
                step_num += 1
                decision_results.append(d)

            ordered_steps = sorted(raw_steps, key=lambda s: (self.STAGE_ORDER.get(s.stage, 99), s.step_number))
            for idx, s in enumerate(ordered_steps, 1):
                s.step_number = idx

            return PreprocessingPlan(
                dataset_id=getattr(dataset_profile, "dataset_id", "dataset_001"),
                dataset_name=dataset_profile.dataset_name,
                target_column=target_col or "target",
                steps=ordered_steps,
            )

        # Step 1: Data Ingestion Step
        ingest_res = DecisionResult(
            domain=DecisionDomain.COLUMN_INTELLIGENCE,
            decision="VERIFY_DATASET_SCHEMA",
            confidence=1.0,
            reasoning="Validate input dataset schema and fingerprint.",
            source=DecisionSource.RULE,
        )
        decision_results.append(ingest_res)
        raw_steps.append(
            PreprocessingStep(
                step_number=1,
                stage="DATA_INGESTION",
                domain=DecisionDomain.COLUMN_INTELLIGENCE,
                action="VERIFY_DATASET_SCHEMA",
                columns=list(col_map.keys()),
                decision_id=ingest_res.decision_id,
                decision_source=ingest_res.source,
                confidence=1.0,
                reasoning="Ingest raw dataset schema.",
            )
        )

        # Step 2: Target Separation Step
        if target_col and target_col in col_map:
            target_res = self.orchestrator.evaluate_decision(
                domain=DecisionDomain.TARGET_DETECTION,
                dataset_profile=dataset_profile,
                model_family=model_family,
                user_mission=user_mission,
                user_selection=user_selections.get("target_detection"),
                auto_approve_default=auto_approve_default,
            )
            decision_results.append(target_res)
            raw_steps.append(
                PreprocessingStep(
                    step_number=2,
                    stage="TARGET_SEPARATION",
                    domain=DecisionDomain.TARGET_DETECTION,
                    action=f"SEPARATE_TARGET:{target_col}",
                    columns=[target_col],
                    decision_id=target_res.decision_id,
                    decision_source=target_res.source,
                    confidence=target_res.confidence,
                    reasoning=f"Separate target column '{target_col}' from feature matrix.",
                )
            )

        # Step 3: Leakage Detection & Removal Step
        leakage_res = self.orchestrator.evaluate_decision(
            domain=DecisionDomain.LEAKAGE_DETECTION,
            dataset_profile=dataset_profile,
            model_family=model_family,
            user_mission=user_mission,
            user_selection=user_selections.get("leakage_detection"),
            auto_approve_default=auto_approve_default,
        )
        decision_results.append(leakage_res)
        if "FLAG_LEAKAGE" in leakage_res.decision or "DROP" in leakage_res.decision:
            leak_cols = [c for c in col_map.keys() if c != target_col and ("leak" in c.lower() or c in leakage_res.decision)]
            if leak_cols:
                raw_steps.append(
                    PreprocessingStep(
                        step_number=3,
                        stage="LEAKAGE_REMOVAL",
                        domain=DecisionDomain.LEAKAGE_DETECTION,
                        action="DROP_LEAKAGE_COLUMNS",
                        columns=leak_cols,
                        decision_id=leakage_res.decision_id,
                        decision_source=leakage_res.source,
                        confidence=leakage_res.confidence,
                        reasoning=leakage_res.reasoning,
                    )
                )

        # Step 4: Train/Test Split Step
        split_res = self.orchestrator.evaluate_decision(
            domain=DecisionDomain.PIPELINE_STRATEGY,
            dataset_profile=dataset_profile,
            model_family=model_family,
            user_mission=user_mission,
            user_selection=user_selections.get("pipeline_strategy"),
            auto_approve_default=auto_approve_default,
        )
        decision_results.append(split_res)
        raw_steps.append(
            PreprocessingStep(
                step_number=4,
                stage="TRAIN_TEST_SPLIT",
                domain=DecisionDomain.PIPELINE_STRATEGY,
                action="STRATIFIED_TRAIN_TEST_SPLIT",
                columns=list(col_map.keys()),
                params={"test_size": 0.2, "random_state": 42},
                decision_id=split_res.decision_id,
                decision_source=split_res.source,
                confidence=split_res.confidence,
                reasoning=split_res.reasoning,
            )
        )

        # Step 5: Column-Level Feature Steps (Target column excluded!)
        feature_cols = [c_name for c_name in col_map.keys() if c_name != target_col]
        for col_name in feature_cols:
            col_prof = col_map[col_name]
            if not isinstance(col_prof, ColumnProfileExtended):
                col_prof = ColumnProfileExtended(
                    name=col_prof.name,
                    normalized_dtype=col_prof.type.value if hasattr(col_prof.type, "value") else str(col_prof.type),
                    missing_ratio=col_prof.missing_pct / 100.0 if hasattr(col_prof, "missing_pct") else 0.0,
                    distinct_count=col_prof.distinct_count if hasattr(col_prof, "distinct_count") else 0,
                )

            # Missing Value Handling
            if col_prof.missing_ratio > 0:
                res = self.orchestrator.evaluate_decision(
                    domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                    col_profile=col_prof,
                    dataset_profile=dataset_profile,
                    model_family=model_family,
                    user_mission=user_mission,
                    user_selection=user_selections.get(f"{col_name}:missing_value_strategy"),
                    auto_approve_default=auto_approve_default,
                )
                decision_results.append(res)
                if res.decision not in ("PASS_THROUGH", "NONE"):
                    raw_steps.append(
                        PreprocessingStep(
                            step_number=5,
                            stage="MISSING_VALUE_HANDLING",
                            domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                            action=res.decision,
                            columns=[col_name],
                            params={"missing_ratio": col_prof.missing_ratio},
                            decision_id=res.decision_id,
                            decision_source=res.source,
                            confidence=res.confidence,
                            reasoning=res.reasoning,
                        )
                    )

            # Encoding Strategy
            if col_prof.normalized_dtype in ("categorical", "text") or col_prof.distinct_count < 20:
                res = self.orchestrator.evaluate_decision(
                    domain=DecisionDomain.ENCODING_STRATEGY,
                    col_profile=col_prof,
                    dataset_profile=dataset_profile,
                    model_family=model_family,
                    user_mission=user_mission,
                    user_selection=user_selections.get(f"{col_name}:encoding_strategy"),
                    auto_approve_default=auto_approve_default,
                )
                decision_results.append(res)
                if res.decision not in ("PASS_THROUGH", "NONE"):
                    raw_steps.append(
                        PreprocessingStep(
                            step_number=6,
                            stage="ENCODING",
                            domain=DecisionDomain.ENCODING_STRATEGY,
                            action=res.decision,
                            columns=[col_name],
                            params={"distinct_count": col_prof.distinct_count},
                            decision_id=res.decision_id,
                            decision_source=res.source,
                            confidence=res.confidence,
                            reasoning=res.reasoning,
                        )
                    )

            # Scaling & Transformation
            if col_prof.normalized_dtype == "numeric":
                res = self.orchestrator.evaluate_decision(
                    domain=DecisionDomain.SCALING_TRANSFORMATION,
                    col_profile=col_prof,
                    dataset_profile=dataset_profile,
                    model_family=model_family,
                    user_mission=user_mission,
                    user_selection=user_selections.get(f"{col_name}:scaling_transformation"),
                    auto_approve_default=auto_approve_default,
                )
                decision_results.append(res)
                if res.decision not in ("PASS_THROUGH", "NONE"):
                    raw_steps.append(
                        PreprocessingStep(
                            step_number=7,
                            stage="SCALING",
                            domain=DecisionDomain.SCALING_TRANSFORMATION,
                            action=res.decision,
                            columns=[col_name],
                            params={"skewness": col_prof.skewness},
                            decision_id=res.decision_id,
                            decision_source=res.source,
                            confidence=res.confidence,
                            reasoning=res.reasoning,
                        )
                    )

            # Outlier Handling
            if col_prof.normalized_dtype == "numeric" and col_prof.outlier_ratio > 0.01:
                res = self.orchestrator.evaluate_decision(
                    domain=DecisionDomain.OUTLIER_HANDLING,
                    col_profile=col_prof,
                    dataset_profile=dataset_profile,
                    model_family=model_family,
                    user_mission=user_mission,
                    user_selection=user_selections.get(f"{col_name}:outlier_handling"),
                    auto_approve_default=auto_approve_default,
                )
                decision_results.append(res)
                if res.decision not in ("PASS_THROUGH", "NONE"):
                    raw_steps.append(
                        PreprocessingStep(
                            step_number=8,
                            stage="OUTLIER_TRANSFORMATION",
                            domain=DecisionDomain.OUTLIER_HANDLING,
                            action=res.decision,
                            columns=[col_name],
                            params={"outlier_ratio": col_prof.outlier_ratio},
                            decision_id=res.decision_id,
                            decision_source=res.source,
                            confidence=res.confidence,
                            reasoning=res.reasoning,
                        )
                    )

        # Step 6: Feature Selection
        res_fs = self.orchestrator.evaluate_decision(
            domain=DecisionDomain.FEATURE_SELECTION,
            dataset_profile=dataset_profile,
            model_family=model_family,
            user_mission=user_mission,
            user_selection=user_selections.get("feature_selection"),
            auto_approve_default=auto_approve_default,
        )
        decision_results.append(res_fs)
        if res_fs.decision not in ("PASS_THROUGH", "NONE"):
            raw_steps.append(
                PreprocessingStep(
                    step_number=10,
                    stage="FEATURE_SELECTION",
                    domain=DecisionDomain.FEATURE_SELECTION,
                    action=res_fs.decision,
                    columns=feature_cols,
                    decision_id=res_fs.decision_id,
                    decision_source=res_fs.source,
                    confidence=res_fs.confidence,
                    reasoning=res_fs.reasoning,
                )
            )

        # Deduplicate, resolve conflicts, and sort DAG
        cleaned_steps = self._deduplicate_and_resolve_conflicts(raw_steps)
        ordered_steps = self._sort_dag(cleaned_steps)

        # Assign step numbers & build explicit dependency map
        dep_map: Dict[str, List[str]] = {}
        for idx, step in enumerate(ordered_steps, start=1):
            step.step_number = idx
            stage_order = self.STAGE_ORDER.get(step.stage, 99)
            preceding_deps = [
                s.step_id for s in ordered_steps[: idx - 1]
                if self.STAGE_ORDER.get(s.stage, 99) < stage_order
                and (set(s.columns).intersection(set(step.columns)) or not s.columns)
            ]
            step.dependencies = preceding_deps
            dep_map[step.step_id] = preceding_deps

        # Compute output schema
        output_schema = dict(input_schema)
        for s in ordered_steps:
            if "DROP" in s.action:
                for c in s.columns:
                    output_schema.pop(c, None)
            elif "ENCODING" in s.action or "ONE_HOT" in s.action:
                for c in s.columns:
                    output_schema[c] = "numeric"

        ops_list = [s.model_dump(mode="json") for s in ordered_steps]
        conf_scores = [d.confidence for d in decision_results if d.confidence is not None]
        overall_conf = round(sum(conf_scores) / len(conf_scores), 4) if conf_scores else 1.0

        plan = PreprocessingPlan(
            dataset_id=dataset_profile.dataset_name,
            dataset_name=dataset_profile.dataset_name,
            target=target_col,
            target_column=target_col,
            problem_type=prob_type,
            model_family=model_family,
            steps=ordered_steps,
            operations=ops_list,
            dependencies=dep_map,
            expected_input_schema=input_schema,
            expected_output_schema=output_schema,
            decisions_summary={
                "total_decisions": len(decision_results),
                "total_steps": len(ordered_steps),
                "sources_breakdown": {
                    "rule": len([d for d in decision_results if d.source == DecisionSource.RULE]),
                    "rag": len([d for d in decision_results if d.source == DecisionSource.RAG]),
                    "llm": len([d for d in decision_results if d.source == DecisionSource.LLM]),
                    "user": len([d for d in decision_results if d.source in (DecisionSource.USER, DecisionSource.SAFETY_DEFAULT)]),
                },
            },
            overall_confidence=overall_conf,
            reasoning=f"Generated DAG preprocessing plan with {len(ordered_steps)} steps across {len(set(s.stage for s in ordered_steps))} stages.",
        )

        logger.info(f"Built PreprocessingPlan ({plan.plan_id}) with {len(ordered_steps)} DAG steps for dataset '{dataset_profile.dataset_name}'.")
        return plan

    def _deduplicate_and_resolve_conflicts(self, steps: List[PreprocessingStep]) -> List[PreprocessingStep]:
        """Removes duplicate steps and resolves conflicting column transformations."""
        seen_keys = set()
        dropped_cols = set()
        cleaned = []

        # Find dropped columns first
        for s in steps:
            if "DROP" in s.action or "CLASSIFY_IDENTIFIER_AND_DROP" in s.action:
                dropped_cols.update(s.columns)

        for s in steps:
            # Skip transformations on columns marked for dropping
            if not ("DROP" in s.action or "CLASSIFY_IDENTIFIER_AND_DROP" in s.action):
                if any(c in dropped_cols for c in s.columns):
                    logger.warning(f"Removing conflicting step '{s.action}' on column(s) {s.columns} marked for deletion.")
                    continue

            key = (s.stage, s.domain.value if hasattr(s.domain, "value") else str(s.domain), s.action, tuple(sorted(s.columns)))
            if key in seen_keys:
                logger.info(f"Removing duplicate preprocessing step: {key}")
                continue
            seen_keys.add(key)
            cleaned.append(s)

        return cleaned

    def _sort_dag(self, steps: List[PreprocessingStep]) -> List[PreprocessingStep]:
        """Topologically sorts steps according to STAGE_ORDER."""
        return sorted(steps, key=lambda s: self.STAGE_ORDER.get(s.stage, 99))

