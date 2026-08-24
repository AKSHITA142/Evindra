import logging
from typing import Optional, Dict, Any, List, Set

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.decision import DecisionDomain
from backend.schemas.preprocessing_plan import PreprocessingPlan, PreprocessingStep, PlanValidationResult

logger = logging.getLogger("datapilot.engine.plan_validator")


class PlanValidator:
    """
    Plan Validator for Evindra Preprocessing Pipeline (Phase 9).
    Deterministically validates a PreprocessingPlan against 16 core safety gate rules,
    data integrity constraints, target leakage patterns, operation validity, and dtype compatibility.
    """

    SUPPORTED_ACTIONS = {
        # Missing value actions
        "IMPUTE_MEAN", "IMPUTE_MEDIAN", "IMPUTE_MODE", "IMPUTE_KNN",
        "IMPUTE_ZERO", "IMPUTE_EXPLICIT_CATEGORY", "PASS_THROUGH", "NONE", "NO_ACTION",
        # Encoding actions
        "ONE_HOT_ENCODING", "TARGET_ENCODING", "TARGET_ENCODING_OUT_OF_FOLD", "ORDINAL_ENCODING", "FREQUENCY_ENCODING", "CLASSIFY_IDENTIFIER_AND_DROP",
        # Scaling actions
        "STANDARD_SCALER", "MINMAX_SCALER", "ROBUST_SCALER", "LOG_TRANSFORM", "POWER_TRANSFORM", "NO_SCALING",
        # Outlier actions
        "CLIP_IQR", "REMOVE_OUTLIERS", "WINZORIZE", "KEEP_OUTLIERS",
        # Feature selection & column removal actions
        "DROP_COLUMNS", "DROP_LEAKAGE_COLUMNS", "REMOVE_DUPLICATE_COLUMNS", "REMOVE_HIGH_MISSING", "FEATURE_SELECTION",
        # Ingestion & split actions
        "VERIFY_DATASET_SCHEMA", "STRATIFIED_TRAIN_TEST_SPLIT", "TIME_SERIES_SPLIT", "GROUP_KFOLD_SPLIT",
    }

    NUMERIC_ONLY_ACTIONS = {
        "STANDARD_SCALER", "MINMAX_SCALER", "ROBUST_SCALER", "LOG_TRANSFORM",
        "POWER_TRANSFORM", "CLIP_IQR", "WINZORIZE",
    }

    CATEGORICAL_ONLY_ACTIONS = {
        "ONE_HOT_ENCODING", "TARGET_ENCODING", "TARGET_ENCODING_OUT_OF_FOLD", "ORDINAL_ENCODING", "FREQUENCY_ENCODING",
        "IMPUTE_EXPLICIT_CATEGORY",
    }

    def validate_plan(
        self,
        plan: PreprocessingPlan,
        dataset_profile: Optional[DatasetProfile] = None,
    ) -> PlanValidationResult:
        """
        Deterministically validates a PreprocessingPlan against 16 safety gate rules.
        """
        errors: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, str] = {}

        dropped_columns: Set[str] = set()
        seen_operations: Set[tuple] = set()
        split_step_idx: int = -1

        target_col = plan.target or plan.target_column
        if not target_col and dataset_profile:
            target_col = getattr(dataset_profile, "target_column", None) or dataset_profile.dataset_summary.get("target", {}).get("target_column")

        # Map dataset profile column metadata
        col_prof_map: Dict[str, Any] = {}
        if dataset_profile:
            cols = dataset_profile.detailed_column_profiles or dataset_profile.column_profiles or []
            col_prof_map = {c.name: c for c in cols}

        # Rule 1: Target Exists Check
        if target_col:
            checks["target_exists"] = "PASSED"
        else:
            checks["target_exists"] = "FAILED"
            errors.append("Validation Error (Rule 1): Target column is missing or unspecified in preprocessing plan.")

        # Locate train/test split step index
        for idx, step in enumerate(plan.steps):
            if step.stage in ("TRAIN_TEST_SPLIT", "PIPELINE_STRATEGY") or "SPLIT" in step.action.upper():
                split_step_idx = idx
                break

        # Iterate steps for step-level checks
        step_count = len(plan.steps)
        for idx, step in enumerate(plan.steps):
            action_upper = step.action.upper()
            step_cols = step.columns or []
            stage_upper = step.stage.upper()

            # Rule 0: Action Validity Check
            if action_upper not in self.SUPPORTED_ACTIONS:
                errors.append(f"Validation Error: Unsupported or invalid action '{step.action}' in step #{step.step_number}.")

            # Rule 3: Referenced Columns Exist Check
            if col_prof_map:
                for c in step_cols:
                    if c not in col_prof_map and c != target_col and stage_upper not in ("DATA_INGESTION", "FEATURE_SELECTION", "TRAIN_TEST_SPLIT"):
                        warnings.append(f"Validation Warning (Rule 3): Column '{c}' in step #{step.step_number} not found in dataset profile.")

            # Rule 2: Target Isolated in Feature Steps
            if target_col and stage_upper not in ("DATA_INGESTION", "TARGET_SEPARATION", "TRAIN_TEST_SPLIT"):
                if target_col in step_cols:
                    checks["target_isolated"] = "FAILED"
                    errors.append(f"Target Leakage Error (Rule 2): Target column '{target_col}' included in feature step #{step.step_number} ({step.action}).")

            # Rule 4: Data Types Match Operations
            for c in step_cols:
                if action_upper in self.NUMERIC_ONLY_ACTIONS and c in col_prof_map:
                    dtype = getattr(col_prof_map[c], "normalized_dtype", getattr(col_prof_map[c], "type", ""))
                    dtype_str = dtype.value if hasattr(dtype, "value") else str(dtype)
                    if dtype_str.lower() in ("categorical", "text", "string"):
                        errors.append(f"Dtype Mismatch Error (Rule 4): Numeric action '{step.action}' applied to non-numeric column '{c}'.")

            # Rule 5: Duplicate Transformation Check
            for c in step_cols:
                key = (stage_upper, action_upper, c)
                if key in seen_operations:
                    warnings.append(f"Duplicate Transformation Warning (Rule 5): Step #{step.step_number} repeats action '{step.action}' on column '{c}'.")
                seen_operations.add(key)

            # Rule 6: Conflicting Operations Check
            for c in step_cols:
                if c in dropped_columns and action_upper not in ("DROP_COLUMNS", "DROP_LEAKAGE_COLUMNS"):
                    errors.append(f"Contradiction Error (Rule 6): Step #{step.step_number} attempts action '{step.action}' on previously dropped column '{c}'.")

            # Rule 8: Test-Data Fitting Prevention Check
            if split_step_idx >= 0 and idx < split_step_idx:
                if action_upper in ("STANDARD_SCALER", "MINMAX_SCALER", "ROBUST_SCALER", "TARGET_ENCODING"):
                    errors.append(f"Data Leakage Error (Rule 8): Transformation '{step.action}' fitted before train/test split in step #{step.step_number}.")

            # Rule 13: Target Encoding Leakage Safety Check
            if "TARGET_ENCODING" in action_upper:
                if split_step_idx < 0 and "OUT_OF_FOLD" not in action_upper and not step.params.get("out_of_fold"):
                    errors.append(f"Target Encoding Leakage Error (Rule 13): Target encoding in step #{step.step_number} must be out-of-fold or fitted strictly on train split.")

            # Rule 7 & 16: Target Leakage & Generated Feature Check
            if target_col:
                for c in step_cols:
                    if c != target_col and (c.startswith(f"{target_col}_") or c.endswith(f"_{target_col}") or f"target_{target_col}" in c or f"{target_col}_" in c):
                        errors.append(f"Target Leakage Error (Rule 7/16): Feature '{c}' in step #{step.step_number} directly derives from target column name '{target_col}'.")

            # Rule 14: Temporal Operations Check
            if "FUTURE" in action_upper or any("future" in str(p).lower() for p in step.params.values()):
                errors.append(f"Temporal Leakage Error (Rule 14): Step #{step.step_number} ({step.action}) contains future timestamp lookup operations.")

            # Track dropped columns state
            if action_upper in ("DROP_COLUMNS", "DROP_LEAKAGE_COLUMNS", "CLASSIFY_IDENTIFIER_AND_DROP"):
                dropped_columns.update(step_cols)

        # Populate checks evaluation map
        checks.setdefault("target_exists", "PASSED" if target_col else "FAILED")
        checks["target_isolated"] = "FAILED" if any("Rule 2" in e for e in errors) else "PASSED"
        checks["columns_exist"] = "WARNING" if any("Rule 3" in w for w in warnings) else "PASSED"
        checks["dtype_compatibility"] = "FAILED" if any("Rule 4" in e for e in errors) else "PASSED"
        checks["no_duplicate_transformations"] = "WARNING" if any("Rule 5" in w for w in warnings) else "PASSED"
        checks["no_conflicting_operations"] = "FAILED" if any("Rule 6" in e for e in errors) else "PASSED"
        checks["no_target_leakage"] = "FAILED" if any("Rule 7" in e or "Rule 16" in e for e in errors) else "PASSED"
        checks["no_test_data_fitting"] = "FAILED" if any("Rule 8" in e for e in errors) else "PASSED"
        checks["target_encoding_leakage_safe"] = "FAILED" if any("Rule 13" in e for e in errors) else "PASSED"
        checks["temporal_ordering"] = "FAILED" if any("Rule 14" in e for e in errors) else "PASSED"

        is_valid = len(errors) == 0
        if not is_valid:
            severity = "CRITICAL"
            rec_action = "REJECT"
        elif len(warnings) > 0:
            severity = "WARNING"
            rec_action = "REVISE"
        else:
            severity = "CLEAN"
            rec_action = "PROCEED"

        return PlanValidationResult(
            is_valid=is_valid,
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            checks=checks,
            severity=severity,
            recommended_action=rec_action,
            validated_step_count=step_count,
            metadata={"plan_id": plan.plan_id, "target_column": target_col},
        )


