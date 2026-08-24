import logging
from typing import Optional, Dict, Any, List
import pandas as pd

from backend.schemas.dataset_profile import DatasetProfile
from backend.schemas.decision import DecisionDomain, DecisionSource
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan
from backend.schemas.pipeline import PipelineCandidate, PipelineCandidateSet
from backend.engine.preprocessing_plan_builder import PreprocessingPlanBuilder

logger = logging.getLogger("datapilot.engine.pipeline_generator")


class PipelineGenerator:
    """
    Pipeline Generator for Evindra Preprocessing Pipeline (Phase 13).
    Generates tailored, deterministic preprocessing + feature engineering + feature selection + model candidates
    based on DatasetProfile characteristics (problem type, dataset size, cardinality, missingness).
    """

    def __init__(self, max_pipelines: int = 4):
        self.max_pipelines = max_pipelines
        self.builder = PreprocessingPlanBuilder()

    def generate_candidate_pipelines(
        self,
        dataset_profile: DatasetProfile,
        problem_type: Optional[str] = None,
        max_pipelines: Optional[int] = None,
    ) -> PipelineCandidateSet:
        """
        Generates deterministic candidate pipelines tailored to DatasetProfile.

        Args:
            dataset_profile: Canonical DatasetProfile object.
            problem_type: "classification" or "regression".
            max_pipelines: Optional override for max candidate pipeline count.

        Returns:
            PipelineCandidateSet containing candidate pipelines.
        """
        limit = max_pipelines if max_pipelines is not None else self.max_pipelines
        prob_type = (
            problem_type
            or getattr(dataset_profile, "problem_type", None)
            or dataset_profile.dataset_summary.get("target", {}).get("problem_type")
            or "classification"
        )
        ds_name = dataset_profile.dataset_name
        target_col = dataset_profile.target_column

        num_cols = dataset_profile.numeric_columns or []
        cat_cols = dataset_profile.categorical_columns or []
        dt_cols = dataset_profile.datetime_columns or []
        text_cols = dataset_profile.text_columns or []

        rows = dataset_profile.rows
        has_categoricals = len(cat_cols) > 0
        is_numeric_only = len(cat_cols) == 0 and len(dt_cols) == 0
        is_categorical_heavy = len(cat_cols) > len(num_cols)
        is_large = rows > 50000

        candidates: List[PipelineCandidate] = []

        # -------------------------------------------------------------
        # CANDIDATE 1: Baseline Minimal (Fast, Simple, High Reliability)
        # -------------------------------------------------------------
        p1_steps = []
        has_missing = getattr(dataset_profile, "missing_columns", None) or getattr(dataset_profile, "missing_count", 0) > 0 or True
        if has_missing:
            p1_steps.append(
                PreprocessingStep(
                    step_number=1,
                    stage="MISSING_VALUE_HANDLING",
                    domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                    action="IMPUTE_MEDIAN" if num_cols else "IMPUTE_MODE",
                    columns=num_cols + cat_cols,
                    decision_id="p1_imp",
                    decision_source=DecisionSource.RULE,
                    confidence=0.95,
                )
            )
        if has_categoricals:
            p1_steps.append(
                PreprocessingStep(
                    step_number=len(p1_steps) + 1,
                    stage="ENCODING",
                    domain=DecisionDomain.ENCODING_STRATEGY,
                    action="ONE_HOT_ENCODING" if not is_large else "ORDINAL_ENCODING",
                    columns=cat_cols,
                    decision_id="p1_enc",
                    decision_source=DecisionSource.RULE,
                    confidence=0.90,
                )
            )
        if num_cols:
            p1_steps.append(
                PreprocessingStep(
                    step_number=len(p1_steps) + 1,
                    stage="SCALING",
                    domain=DecisionDomain.SCALING_TRANSFORMATION,
                    action="STANDARD_SCALER",
                    columns=num_cols,
                    decision_id="p1_scale",
                    decision_source=DecisionSource.RULE,
                    confidence=0.90,
                )
            )

        plan1 = PreprocessingPlan(dataset_name=ds_name, target_column=target_col, steps=p1_steps)
        candidates.append(
            PipelineCandidate(
                name="Baseline Minimal Pipeline",
                description="Fast baseline with simple median/mode imputation, standard scaling, and linear/logistic baseline model.",
                preprocessing_plan=plan1,
                feature_engineering_plan={"enabled": False, "strategies": []},
                feature_selection_plan={"enabled": False, "method": "NONE"},
                model_spec={
                    "model_family": "LOGISTIC_REGRESSION" if "class" in prob_type.lower() else "RIDGE_REGRESSION",
                    "hyperparameters": {"C": 1.0, "alpha": 1.0},
                },
                estimated_cost="LOW",
                rank_score=0.75,
            )
        )

        # -------------------------------------------------------------
        # CANDIDATE 2: Robust Scaled Regularized Model (Outlier Resistant)
        # -------------------------------------------------------------
        p2_steps = []
        if num_cols:
            p2_steps.append(
                PreprocessingStep(
                    step_number=1,
                    stage="MISSING_VALUE_HANDLING",
                    domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                    action="IMPUTE_MEDIAN",
                    columns=num_cols,
                    decision_id="p2_imp",
                    decision_source=DecisionSource.RULE,
                    confidence=0.95,
                )
            )
            p2_steps.append(
                PreprocessingStep(
                    step_number=len(p2_steps) + 1,
                    stage="OUTLIER_TRANSFORMATION",
                    domain=DecisionDomain.OUTLIER_HANDLING,
                    action="CLIP_IQR",
                    columns=num_cols,
                    decision_id="p2_outlier",
                    decision_source=DecisionSource.RULE,
                    confidence=0.85,
                )
            )
            p2_steps.append(
                PreprocessingStep(
                    step_number=len(p2_steps) + 1,
                    stage="SCALING",
                    domain=DecisionDomain.SCALING_TRANSFORMATION,
                    action="ROBUST_SCALER",
                    columns=num_cols,
                    decision_id="p2_scale",
                    decision_source=DecisionSource.RULE,
                    confidence=0.90,
                )
            )
        if has_categoricals:
            p2_steps.append(
                PreprocessingStep(
                    step_number=len(p2_steps) + 1,
                    stage="ENCODING",
                    domain=DecisionDomain.ENCODING_STRATEGY,
                    action="TARGET_ENCODING_OUT_OF_FOLD",
                    columns=cat_cols,
                    decision_id="p2_enc",
                    decision_source=DecisionSource.RULE,
                    confidence=0.85,
                )
            )

        plan2 = PreprocessingPlan(dataset_name=ds_name, target_column=target_col, steps=p2_steps)
        candidates.append(
            PipelineCandidate(
                name="Robust Scaled Regularized Pipeline",
                description="Outlier-resistant pipeline with median imputation, IQR clipping, robust scaling, and out-of-fold target encoding.",
                preprocessing_plan=plan2,
                feature_engineering_plan={"enabled": True, "strategies": ["log1p", "ratios"]},
                feature_selection_plan={"enabled": True, "method": "CORRELATION_FILTER"},
                model_spec={
                    "model_family": "ELASTICNET" if "reg" in prob_type.lower() else "LOGISTIC_REGRESSION_L1",
                    "hyperparameters": {"l1_ratio": 0.5},
                },
                estimated_cost="MEDIUM",
                rank_score=0.85,
            )
        )

        # -------------------------------------------------------------
        # CANDIDATE 3: Tree-Based Model (Gradient Boosting / LightGBM)
        # -------------------------------------------------------------
        p3_steps = []
        if has_categoricals:
            p3_steps.append(
                PreprocessingStep(
                    step_number=1,
                    stage="ENCODING",
                    domain=DecisionDomain.ENCODING_STRATEGY,
                    action="FREQUENCY_ENCODING" if is_categorical_heavy else "ORDINAL_ENCODING",
                    columns=cat_cols,
                    decision_id="p3_enc",
                    decision_source=DecisionSource.RULE,
                    confidence=0.90,
                )
            )

        plan3 = PreprocessingPlan(dataset_name=ds_name, target_column=target_col, steps=p3_steps)
        candidates.append(
            PipelineCandidate(
                name="Tree-Based Gradient Boosting Pipeline",
                description="Tree-optimized pipeline using frequency/ordinal encoding and tree models without scaling requirement.",
                preprocessing_plan=plan3,
                feature_engineering_plan={"enabled": True, "strategies": ["datetime_cyclical", "text_length"]},
                feature_selection_plan={"enabled": True, "method": "RF_IMPORTANCE"},
                model_spec={
                    "model_family": "LIGHTGBM" if "class" in prob_type.lower() else "HIST_GRADIENT_BOOSTING",
                    "hyperparameters": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6},
                },
                estimated_cost="MEDIUM",
                rank_score=0.90,
            )
        )

        # -------------------------------------------------------------
        # CANDIDATE 4: Feature-Engineered Ensemble Pipeline
        # -------------------------------------------------------------
        p4_plan = self.builder.build_plan(dataset_profile)
        candidates.append(
            PipelineCandidate(
                name="Feature-Engineered Ensemble Pipeline",
                description="High-performance pipeline combining automated feature engineering, CV feature selection, and ensemble modeling.",
                preprocessing_plan=p4_plan,
                feature_engineering_plan={"enabled": True, "strategies": ["numeric_ratios", "datetime_cyclical", "categorical_combinations"]},
                feature_selection_plan={"enabled": True, "method": "CV_FOLD_STABILITY"},
                model_spec={
                    "model_family": "RANDOM_FOREST",
                    "hyperparameters": {"n_estimators": 200, "max_depth": 10},
                },
                estimated_cost="HIGH",
                rank_score=0.95,
            )
        )

        # Filter down to max_pipelines limit
        selected_candidates = candidates[:limit]

        pipeline_set = PipelineCandidateSet(
            dataset_name=ds_name,
            problem_type=prob_type,
            target_column=target_col,
            total_candidates=len(selected_candidates),
            pipelines=selected_candidates,
            metadata={"max_pipelines_requested": limit, "generated_total": len(candidates)},
        )

        logger.info(f"Generated {len(selected_candidates)} candidate pipelines for dataset '{ds_name}'.")
        return pipeline_set
