import logging
from typing import Optional, Dict, Any, List, Set

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.decision import DecisionDomain
from backend.schemas.preprocessing_plan import PreprocessingPlan, PreprocessingStep, PlanValidationResult

logger = logging.getLogger("datapilot.engine.plan_validator")


class PlanValidator:
    """
    Plan Validator for Evindra Preprocessing Pipeline (Phase 9).
    Deterministically validates a PreprocessingPlan against data integrity rules,
    target leakage constraints, operation validity, step order safety, and dtype compatibility.
    """

    SUPPORTED_ACTIONS = {
        # Missing value actions
        "IMPUTE_MEAN", "IMPUTE_MEDIAN", "IMPUTE_MODE", "IMPUTE_KNN",
        "IMPUTE_ZERO", "IMPUTE_EXPLICIT_CATEGORY", "PASS_THROUGH", "NONE", "NO_ACTION",
        # Encoding actions
        "ONE_HOT_ENCODING", "TARGET_ENCODING", "ORDINAL_ENCODING", "FREQUENCY_ENCODING",
        # Scaling actions
        "STANDARD_SCALER", "MINMAX_SCALER", "ROBUST_SCALER", "LOG_TRANSFORM", "POWER_TRANSFORM",
        # Outlier actions
        "CLIP_IQR", "REMOVE_OUTLIERS", "WINZORIZE",
        # Feature selection & column removal actions
        "DROP_COLUMNS", "REMOVE_DUPLICATE_COLUMNS", "REMOVE_HIGH_MISSING", "FEATURE_SELECTION",
    }

    NUMERIC_ONLY_ACTIONS = {
        "STANDARD_SCALER", "MINMAX_SCALER", "ROBUST_SCALER", "LOG_TRANSFORM",
        "POWER_TRANSFORM", "CLIP_IQR", "WINZORIZE",
    }

    CATEGORICAL_ONLY_ACTIONS = {
        "ONE_HOT_ENCODING", "TARGET_ENCODING", "ORDINAL_ENCODING", "FREQUENCY_ENCODING",
        "IMPUTE_EXPLICIT_CATEGORY",
    }

    def validate_plan(
        self,
        plan: PreprocessingPlan,
        dataset_profile: Optional[DatasetProfile] = None,
    ) -> PlanValidationResult:
        """
        Deterministically validates a PreprocessingPlan.

        Args:
            plan: The PreprocessingPlan object to validate.
            dataset_profile: Optional DatasetProfile for column existence and dtype validation.

        Returns:
            PlanValidationResult object with is_valid boolean, errors, and warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []
        dropped_columns: Set[str] = set()
        scaled_columns: Set[str] = set()
        encoded_columns: Set[str] = set()

        target_col = plan.target_column
        if not target_col and dataset_profile:
            target_col = getattr(dataset_profile, "target_column", None) or dataset_profile.dataset_summary.get("target", {}).get("target_column")

        # Map dataset profile column metadata
        col_prof_map: Dict[str, Any] = {}
        if dataset_profile:
            cols = dataset_profile.detailed_column_profiles or dataset_profile.column_profiles or []
            col_prof_map = {c.name: c for c in cols}

        step_count = 0
        for step in plan.steps:
            step_count += 1
            action_upper = step.action.upper()
            step_cols = step.columns or []

            # 1. Action Validity Check
            if action_upper not in self.SUPPORTED_ACTIONS:
                errors.append(f"Step #{step.step_number} ({step.domain.value if hasattr(step.domain, 'value') else str(step.domain)}): Unsupported or invalid action '{step.action}'.")

            # 2. Target Leakage Validation
            if target_col:
                for c in step_cols:
                    if c == target_col:
                        if step.domain in (DecisionDomain.MISSING_VALUE_STRATEGY, DecisionDomain.SCALING_TRANSFORMATION, DecisionDomain.OUTLIER_HANDLING):
                            errors.append(f"Target Leakage Error in Step #{step.step_number}: Target column '{target_col}' cannot undergo feature transformation '{step.action}'.")
                        elif step.domain == DecisionDomain.ENCODING_STRATEGY:
                            errors.append(f"Target Leakage Error in Step #{step.step_number}: Target column '{target_col}' cannot be encoded as a feature.")
                        elif action_upper in ("DROP_COLUMNS", "FEATURE_SELECTION") and c in step.params.get("columns_to_drop", []):
                            errors.append(f"Target Deletion Error in Step #{step.step_number}: Target column '{target_col}' is marked for deletion.")

            # 3. Column Existence & References
            if col_prof_map:
                for c in step_cols:
                    if c not in col_prof_map and c != target_col and step.domain != DecisionDomain.FEATURE_SELECTION:
                        warnings.append(f"Step #{step.step_number}: Column '{c}' not found in dataset profile metadata.")

            # 4. Step Order & Contradiction Validation
            for c in step_cols:
                if c in dropped_columns:
                    errors.append(f"Contradiction Error in Step #{step.step_number}: Column '{c}' was dropped in an earlier step but referenced in '{step.action}'.")

                if action_upper in self.NUMERIC_ONLY_ACTIONS:
                    if c in col_prof_map:
                        dtype = getattr(col_prof_map[c], "normalized_dtype", getattr(col_prof_map[c], "type", ""))
                        dtype_str = dtype.value if hasattr(dtype, "value") else str(dtype)
                        if dtype_str.lower() in ("categorical", "text", "string"):
                            errors.append(f"Dtype Mismatch Error in Step #{step.step_number}: Cannot apply numeric transformation '{step.action}' to non-numeric column '{c}'.")

                if action_upper in self.CATEGORICAL_ONLY_ACTIONS:
                    if c in encoded_columns:
                        warnings.append(f"Step #{step.step_number}: Double encoding applied to column '{c}'.")

            # Track state changes
            if action_upper == "DROP_COLUMNS":
                dropped_columns.update(step_cols)
            if action_upper in self.NUMERIC_ONLY_ACTIONS:
                scaled_columns.update(step_cols)
            if action_upper in self.CATEGORICAL_ONLY_ACTIONS:
                encoded_columns.update(step_cols)

            # 5. Parameter Constraints Validation
            if "missing_ratio" in step.params:
                mr = step.params["missing_ratio"]
                if not (0.0 <= mr <= 1.0):
                    errors.append(f"Invalid Parameter in Step #{step.step_number}: missing_ratio must be between 0.0 and 1.0 (got {mr}).")

        is_valid = len(errors) == 0
        if is_valid:
            logger.info(f"PreprocessingPlan '{plan.plan_id}' successfully validated ({step_count} steps, 0 errors).")
        else:
            logger.warning(f"PreprocessingPlan '{plan.plan_id}' validation failed with {len(errors)} errors.")

        return PlanValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validated_step_count=step_count,
            metadata={"plan_id": plan.plan_id, "target_column": target_col},
        )
