import time
import logging
from typing import Tuple, Dict, Any, List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer

from backend.schemas.preprocessing_plan import PreprocessingPlan, PreprocessingStep, PlanExecutionResult
from backend.engine.plan_validator import PlanValidator

logger = logging.getLogger("datapilot.engine.plan_executor")


class PlanExecutor:
    """
    Plan Executor for Evindra Preprocessing Pipeline (Phase 10 — Safe Execution).
    Safely executes a validated PreprocessingPlan step-by-step on a Pandas DataFrame
    without data leakage, without in-place side effects on input data, and with runtime safety checks.
    """

    def __init__(self, validator: Optional[PlanValidator] = None):
        self.validator = validator or PlanValidator()

    def execute_plan(
        self,
        plan: PreprocessingPlan,
        df: pd.DataFrame,
        validate_first: bool = True,
    ) -> Tuple[pd.DataFrame, PlanExecutionResult]:
        """
        Safely executes a PreprocessingPlan on a DataFrame.

        Args:
            plan: PreprocessingPlan to execute.
            df: Input pandas DataFrame.
            validate_first: If True, validates plan with PlanValidator before execution.

        Returns:
            Tuple of (transformed_df, PlanExecutionResult)
        """
        start_time = time.time()
        initial_shape = [len(df), len(df.columns)]

        # 1. Plan Validation Check
        if validate_first:
            val_res = self.validator.validate_plan(plan)
            if not val_res.is_valid:
                err_msg = f"Plan validation failed with {len(val_res.errors)} errors: {'; '.join(val_res.errors)}"
                logger.error(err_msg)
                return df.copy(), PlanExecutionResult(
                    plan_id=plan.plan_id,
                    status="FAILED",
                    initial_shape=initial_shape,
                    final_shape=initial_shape,
                    executed_steps_count=0,
                    execution_time_seconds=round(time.time() - start_time, 4),
                    error_message=err_msg,
                )

        # Create defensive isolated copy of input data
        transformed_df = df.copy()
        target_col = plan.target_column

        step_logs: List[Dict[str, Any]] = []
        executed_count = 0

        try:
            for step in plan.steps:
                step_start = time.time()
                action_upper = step.action.upper()
                cols_to_process = [c for c in (step.columns or []) if c in transformed_df.columns and c != target_col]

                cols_before = list(transformed_df.columns)
                rows_before = len(transformed_df)

                # --- Execute Step Actions ---
                if action_upper in ("NO_ACTION", "PASS_THROUGH", "NONE"):
                    pass  # No-op

                elif action_upper == "IMPUTE_MEDIAN":
                    for c in cols_to_process:
                        if pd.api.types.is_numeric_dtype(transformed_df[c]):
                            med_val = transformed_df[c].median()
                            if pd.isna(med_val):
                                med_val = 0.0
                            transformed_df[c] = transformed_df[c].fillna(med_val)

                elif action_upper == "IMPUTE_MEAN":
                    for c in cols_to_process:
                        if pd.api.types.is_numeric_dtype(transformed_df[c]):
                            mean_val = transformed_df[c].mean()
                            if pd.isna(mean_val):
                                mean_val = 0.0
                            transformed_df[c] = transformed_df[c].fillna(mean_val)

                elif action_upper == "IMPUTE_MODE":
                    for c in cols_to_process:
                        mode_res = transformed_df[c].mode()
                        fill_val = mode_res.iloc[0] if not mode_res.empty else "Missing"
                        transformed_df[c] = transformed_df[c].fillna(fill_val)

                elif action_upper in ("IMPUTE_ZERO", "IMPUTE_CONSTANT"):
                    fill_val = step.params.get("fill_value", 0)
                    for c in cols_to_process:
                        transformed_df[c] = transformed_df[c].fillna(fill_val)

                elif action_upper == "IMPUTE_EXPLICIT_CATEGORY":
                    fill_val = step.params.get("fill_value", "Missing")
                    for c in cols_to_process:
                        transformed_df[c] = transformed_df[c].fillna(fill_val)

                elif action_upper == "IMPUTE_KNN":
                    if cols_to_process:
                        num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(transformed_df[c])]
                        if num_cols:
                            knn = KNNImputer(n_neighbors=step.params.get("n_neighbors", 5))
                            transformed_df[num_cols] = knn.fit_transform(transformed_df[num_cols])

                elif action_upper == "ONE_HOT_ENCODING":
                    if cols_to_process:
                        # Use pd.get_dummies with prefixing
                        ohe_cols = [c for c in cols_to_process if str(transformed_df[c].dtype) in ("object", "category", "string")]
                        if ohe_cols:
                            dummies = pd.get_dummies(transformed_df[ohe_cols], prefix=ohe_cols, drop_first=step.params.get("drop_first", False), dtype=float)
                            transformed_df = transformed_df.drop(columns=ohe_cols).join(dummies)

                elif action_upper == "ORDINAL_ENCODING":
                    if cols_to_process:
                        cat_cols = [c for c in cols_to_process if str(transformed_df[c].dtype) in ("object", "category", "string")]
                        if cat_cols:
                            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                            # Fill NaNs before encoding to prevent errors
                            df_cats = transformed_df[cat_cols].fillna("Unknown").astype(str)
                            transformed_df[cat_cols] = enc.fit_transform(df_cats)

                elif action_upper == "STANDARD_SCALER":
                    num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(transformed_df[c])]
                    if num_cols:
                        scaler = StandardScaler()
                        transformed_df[num_cols] = scaler.fit_transform(transformed_df[num_cols].fillna(0.0))

                elif action_upper == "MINMAX_SCALER":
                    num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(transformed_df[c])]
                    if num_cols:
                        scaler = MinMaxScaler()
                        transformed_df[num_cols] = scaler.fit_transform(transformed_df[num_cols].fillna(0.0))

                elif action_upper == "ROBUST_SCALER":
                    num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(transformed_df[c])]
                    if num_cols:
                        scaler = RobustScaler()
                        transformed_df[num_cols] = scaler.fit_transform(transformed_df[num_cols].fillna(0.0))

                elif action_upper == "LOG_TRANSFORM":
                    num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(transformed_df[c])]
                    for c in num_cols:
                        transformed_df[c] = np.log1p(transformed_df[c].clip(lower=0))

                elif action_upper == "CLIP_IQR":
                    num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(transformed_df[c])]
                    for c in num_cols:
                        q1 = transformed_df[c].quantile(0.25)
                        q3 = transformed_df[c].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        transformed_df[c] = transformed_df[c].clip(lower=lower_bound, upper=upper_bound)

                elif action_upper in ("DROP_COLUMNS", "REMOVE_HIGH_MISSING", "REMOVE_DUPLICATE_COLUMNS"):
                    to_drop = [c for c in (step.columns or step.params.get("columns_to_drop", [])) if c in transformed_df.columns and c != target_col]
                    if to_drop:
                        transformed_df = transformed_df.drop(columns=to_drop)

                # Runtime Safety Checks after step
                if len(transformed_df) == 0:
                    raise ValueError(f"Execution Error in Step #{step.step_number} ({step.action}): DataFrame became empty!")

                step_duration = round(time.time() - step_start, 4)
                executed_count += 1
                step_logs.append({
                    "step_number": step.step_number,
                    "domain": step.domain.value if hasattr(step.domain, "value") else str(step.domain),
                    "action": step.action,
                    "columns_processed": cols_to_process,
                    "rows_before": rows_before,
                    "rows_after": len(transformed_df),
                    "cols_before": len(cols_before),
                    "cols_after": len(transformed_df.columns),
                    "duration_seconds": step_duration,
                })

        except Exception as e:
            err_msg = f"Runtime execution failure in step #{executed_count + 1}: {str(e)}"
            logger.error(err_msg, exc_info=True)
            return df.copy(), PlanExecutionResult(
                plan_id=plan.plan_id,
                status="FAILED",
                initial_shape=initial_shape,
                final_shape=[len(transformed_df), len(transformed_df.columns)],
                executed_steps_count=executed_count,
                step_logs=step_logs,
                execution_time_seconds=round(time.time() - start_time, 4),
                error_message=err_msg,
            )

        final_shape = [len(transformed_df), len(transformed_df.columns)]
        total_time = round(time.time() - start_time, 4)
        logger.info(f"Successfully executed PreprocessingPlan '{plan.plan_id}' ({executed_count} steps in {total_time}s). Initial shape: {initial_shape}, Final shape: {final_shape}")

        return transformed_df, PlanExecutionResult(
            plan_id=plan.plan_id,
            status="SUCCESS",
            initial_shape=initial_shape,
            final_shape=final_shape,
            executed_steps_count=executed_count,
            step_logs=step_logs,
            execution_time_seconds=total_time,
            metadata={"target_column": target_col},
        )
