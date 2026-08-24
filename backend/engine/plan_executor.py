import time
import logging
from typing import Tuple, Dict, Any, List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer

from backend.schemas.preprocessing_plan import PreprocessingPlan, PreprocessingStep, PlanExecutionResult
from backend.schemas.dataset_profile import DatasetProfile
from backend.engine.plan_validator import PlanValidator

logger = logging.getLogger("datapilot.engine.plan_executor")


class PlanExecutor:
    """
    Plan Executor for Evindra Preprocessing Pipeline (Phase 10 — Safe Execution).
    Safely executes a validated PreprocessingPlan step-by-step on a Pandas DataFrame
    enforcing the strict invariant:
    TRAIN/TEST SPLIT -> FIT preprocessing ONLY on TRAIN -> TRANSFORM TEST.
    """

    def __init__(self, validator: Optional[PlanValidator] = None):
        self.validator = validator or PlanValidator()

    def execute_plan(
        self,
        plan: PreprocessingPlan,
        df: pd.DataFrame,
        dataset_profile: Optional[DatasetProfile] = None,
        validate_first: bool = True,
    ) -> Tuple[pd.DataFrame, PlanExecutionResult]:
        """
        Executes plan on a single DataFrame or delegates to train/test split execution.
        """
        X_train, X_test, y_train, y_test, fitted_transformers, result = self.execute_train_test_pipeline(
            plan=plan, df=df, dataset_profile=dataset_profile, validate_first=validate_first
        )

        if result.status == "FAILED":
            return df.copy(), result

        # Recombine X_train and y_train (or full transformed df) for return compatibility
        transformed_df = X_train.copy()
        if y_train is not None and isinstance(y_train, pd.Series):
            transformed_df[y_train.name or "target"] = y_train.values

        return transformed_df, result

    def execute_train_test_pipeline(
        self,
        plan: PreprocessingPlan,
        df: pd.DataFrame,
        dataset_profile: Optional[DatasetProfile] = None,
        validate_first: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series], Dict[str, Any], PlanExecutionResult]:
        """
        Executes plan with strict train/test fit isolation:
        FIT transformers ONLY on X_train; TRANSFORM X_test.
        """
        start_time = time.time()
        initial_shape = [len(df), len(df.columns)]
        execution_trace: List[Dict[str, Any]] = []
        transformation_mapping: Dict[str, List[str]] = {}
        fitted_transformers: Dict[str, Any] = {}

        # 1. Phase 9 Validation Gate
        if validate_first:
            val_res = self.validator.validate_plan(plan, dataset_profile=dataset_profile)
            if not val_res.is_valid:
                err_msg = f"Phase 9 Plan Validation Failed: {'; '.join(val_res.errors)}"
                logger.error(err_msg)
                return (
                    df.copy(),
                    pd.DataFrame(),
                    None,
                    None,
                    {},
                    PlanExecutionResult(
                        plan_id=plan.plan_id,
                        status="FAILED",
                        initial_shape=initial_shape,
                        final_shape=initial_shape,
                        executed_steps_count=0,
                        execution_time_seconds=round(time.time() - start_time, 4),
                        error_message=err_msg,
                        warnings=val_res.warnings,
                    ),
                )

        target_col = plan.target or plan.target_column
        df_clean = df.copy()

        # Separate target column if present
        y: Optional[pd.Series] = None
        if target_col and target_col in df_clean.columns:
            y = df_clean[target_col].copy()
            X = df_clean.drop(columns=[target_col]).copy()
        else:
            X = df_clean.copy()

        # Perform Train/Test Split BEFORE feature preprocessing fit
        if y is not None and len(X) > 1 and test_size is not None and test_size > 0.0:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, shuffle=False
                )
        else:
            X_train = X.copy()
            X_test = pd.DataFrame(columns=X.columns)
            y_train = y
            y_test = None

        for col in X_train.columns:
            transformation_mapping[col] = [col]

        executed_count = 0
        try:
            for step in plan.steps:
                step_start = time.time()
                action_upper = step.action.upper()
                stage_upper = step.stage.upper()

                if stage_upper in ("DATA_INGESTION", "TARGET_SEPARATION", "TRAIN_TEST_SPLIT"):
                    continue

                cols_to_process = [c for c in (step.columns or []) if c in X_train.columns]
                rows_train_before = len(X_train)
                cols_train_before = len(X_train.columns)

                # --- 1. MISSING VALUE HANDLING ---
                if action_upper == "IMPUTE_MEDIAN":
                    for c in cols_to_process:
                        if pd.api.types.is_numeric_dtype(X_train[c]):
                            imp = SimpleImputer(strategy="median")
                            X_train[c] = imp.fit_transform(X_train[[c]]).ravel()
                            if not X_test.empty and c in X_test.columns:
                                X_test[c] = imp.transform(X_test[[c]]).ravel()
                            fitted_transformers[f"{c}:imputer"] = imp

                elif action_upper == "IMPUTE_MEAN":
                    for c in cols_to_process:
                        if pd.api.types.is_numeric_dtype(X_train[c]):
                            imp = SimpleImputer(strategy="mean")
                            X_train[c] = imp.fit_transform(X_train[[c]]).ravel()
                            if not X_test.empty and c in X_test.columns:
                                X_test[c] = imp.transform(X_test[[c]]).ravel()
                            fitted_transformers[f"{c}:imputer"] = imp

                elif action_upper == "IMPUTE_MODE":
                    for c in cols_to_process:
                        imp = SimpleImputer(strategy="most_frequent")
                        X_train[c] = imp.fit_transform(X_train[[c]]).ravel()
                        if not X_test.empty and c in X_test.columns:
                            X_test[c] = imp.transform(X_test[[c]]).ravel()
                        fitted_transformers[f"{c}:imputer"] = imp

                elif action_upper in ("IMPUTE_ZERO", "IMPUTE_CONSTANT"):
                    fill_val = step.params.get("fill_value", 0)
                    for c in cols_to_process:
                        imp = SimpleImputer(strategy="constant", fill_value=fill_val)
                        X_train[c] = imp.fit_transform(X_train[[c]]).ravel()
                        if not X_test.empty and c in X_test.columns:
                            X_test[c] = imp.transform(X_test[[c]]).ravel()
                        fitted_transformers[f"{c}:imputer"] = imp

                elif action_upper == "IMPUTE_KNN":
                    num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(X_train[c])]
                    if num_cols:
                        imp = KNNImputer(n_neighbors=step.params.get("n_neighbors", 5))
                        X_train[num_cols] = imp.fit_transform(X_train[num_cols])
                        if not X_test.empty and all(c in X_test.columns for c in num_cols):
                            X_test[num_cols] = imp.transform(X_test[num_cols])
                        fitted_transformers["knn_imputer"] = imp

                # --- 2. ENCODING STRATEGY ---
                elif action_upper == "ONE_HOT_ENCODING":
                    cat_cols = [c for c in cols_to_process if str(X_train[c].dtype) in ("object", "category", "string")]
                    if cat_cols:
                        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                        train_encoded = ohe.fit_transform(X_train[cat_cols].astype(str))
                        feature_names = list(ohe.get_feature_names_out(cat_cols))
                        
                        df_enc_train = pd.DataFrame(train_encoded, columns=feature_names, index=X_train.index)
                        X_train = X_train.drop(columns=cat_cols).join(df_enc_train)

                        if not X_test.empty and all(c in X_test.columns for c in cat_cols):
                            test_encoded = ohe.transform(X_test[cat_cols].astype(str))
                            df_enc_test = pd.DataFrame(test_encoded, columns=feature_names, index=X_test.index)
                            X_test = X_test.drop(columns=cat_cols).join(df_enc_test)

                        for c in cat_cols:
                            transformation_mapping[c] = [f for f in feature_names if f.startswith(f"{c}_")]
                        fitted_transformers["one_hot_encoder"] = ohe

                elif action_upper == "ORDINAL_ENCODING":
                    cat_cols = [c for c in cols_to_process if str(X_train[c].dtype) in ("object", "category", "string")]
                    if cat_cols:
                        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                        X_train[cat_cols] = ord_enc.fit_transform(X_train[cat_cols].astype(str))
                        if not X_test.empty and all(c in X_test.columns for c in cat_cols):
                            X_test[cat_cols] = ord_enc.transform(X_test[cat_cols].astype(str))
                        fitted_transformers["ordinal_encoder"] = ord_enc

                elif action_upper == "FREQUENCY_ENCODING":
                    cat_cols = [c for c in cols_to_process if c in X_train.columns and str(X_train[c].dtype) in ("object", "category", "string")]
                    for c in cat_cols:
                        freq_map = X_train[c].value_counts(normalize=True).to_dict()
                        X_train[c] = X_train[c].map(freq_map).fillna(0.0)
                        if not X_test.empty and c in X_test.columns:
                            X_test[c] = X_test[c].map(freq_map).fillna(0.0)
                        fitted_transformers[f"{c}:frequency_encoder"] = freq_map

                elif "TARGET_ENCODING" in action_upper:
                    cat_cols = [c for c in cols_to_process if c in X_train.columns]
                    if cat_cols and y_train is not None:
                        for c in cat_cols:
                            means = y_train.groupby(X_train[c]).mean().to_dict()
                            global_mean = float(y_train.mean())
                            X_train[c] = X_train[c].map(means).fillna(global_mean)
                            if not X_test.empty and c in X_test.columns:
                                X_test[c] = X_test[c].map(means).fillna(global_mean)
                            fitted_transformers[f"{c}:target_encoder"] = {"means": means, "global_mean": global_mean}

                elif action_upper == "IMPUTE_EXPLICIT_CATEGORY":
                    for c in cols_to_process:
                        fill_val = step.params.get("fill_value", "MISSING")
                        X_train[c] = X_train[c].fillna(fill_val)
                        if not X_test.empty and c in X_test.columns:
                            X_test[c] = X_test[c].fillna(fill_val)

                # --- 3. SCALING & TRANSFORMATION ---
                elif action_upper == "STANDARD_SCALER":
                    num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(X_train[c])]
                    if num_cols:
                        scaler = StandardScaler()
                        X_train[num_cols] = scaler.fit_transform(X_train[num_cols].fillna(0.0))
                        if not X_test.empty and all(c in X_test.columns for c in num_cols):
                            X_test[num_cols] = scaler.transform(X_test[num_cols].fillna(0.0))
                        fitted_transformers["standard_scaler"] = scaler

                elif action_upper == "MINMAX_SCALER":
                    num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(X_train[c])]
                    if num_cols:
                        scaler = MinMaxScaler()
                        X_train[num_cols] = scaler.fit_transform(X_train[num_cols].fillna(0.0))
                        if not X_test.empty and all(c in X_test.columns for c in num_cols):
                            X_test[num_cols] = scaler.transform(X_test[num_cols].fillna(0.0))
                        fitted_transformers["minmax_scaler"] = scaler

                elif action_upper == "ROBUST_SCALER":
                    num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(X_train[c])]
                    if num_cols:
                        scaler = RobustScaler()
                        X_train[num_cols] = scaler.fit_transform(X_train[num_cols].fillna(0.0))
                        if not X_test.empty and all(c in X_test.columns for c in num_cols):
                            X_test[num_cols] = scaler.transform(X_test[num_cols].fillna(0.0))
                        fitted_transformers["robust_scaler"] = scaler

                elif action_upper == "LOG_TRANSFORM":
                    num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(X_train[c])]
                    for c in num_cols:
                        X_train[c] = np.log1p(X_train[c].clip(lower=0))
                        if not X_test.empty and c in X_test.columns:
                            X_test[c] = np.log1p(X_test[c].clip(lower=0))

                elif action_upper in ("CLIP_IQR", "WINSORIZE", "WINZORIZE", "WINSORIZE_CLIPPING"):
                    num_cols = [c for c in cols_to_process if pd.api.types.is_numeric_dtype(X_train[c])]
                    for c in num_cols:
                        q1 = X_train[c].quantile(0.25)
                        q3 = X_train[c].quantile(0.75)
                        iqr = q3 - q1
                        lower_b = q1 - 1.5 * iqr
                        upper_b = q3 + 1.5 * iqr
                        X_train[c] = X_train[c].clip(lower=lower_b, upper=upper_b)
                        if not X_test.empty and c in X_test.columns:
                            X_test[c] = X_test[c].clip(lower=lower_b, upper=upper_b)
                        fitted_transformers[f"{c}:iqr_bounds"] = {"lower": lower_b, "upper": upper_b}

                # --- 4. DROPPING COLUMNS ---
                elif action_upper in ("DROP_COLUMNS", "DROP_LEAKAGE_COLUMNS", "REMOVE_HIGH_MISSING", "CLASSIFY_IDENTIFIER_AND_DROP"):
                    to_drop = [c for c in (step.columns or []) if c in X_train.columns]
                    if to_drop:
                        X_train = X_train.drop(columns=to_drop)
                        if not X_test.empty:
                            X_test = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])
                        for c in to_drop:
                            transformation_mapping[c] = []

                step_duration = round(time.time() - step_start, 4)
                executed_count += 1
                trace_entry = {
                    "step_number": step.step_number,
                    "stage": step.stage,
                    "action": step.action,
                    "columns_processed": cols_to_process,
                    "train_rows": len(X_train),
                    "train_cols": len(X_train.columns),
                    "duration_seconds": step_duration,
                }
                execution_trace.append(trace_entry)

        except Exception as e:
            err_msg = f"Runtime Execution Failure in Step #{executed_count + 1} ({plan.steps[executed_count].action if executed_count < len(plan.steps) else 'Unknown'}): {str(e)}"
            logger.error(err_msg, exc_info=True)
            return (
                X_train,
                X_test,
                y_train,
                y_test,
                fitted_transformers,
                PlanExecutionResult(
                    plan_id=plan.plan_id,
                    status="FAILED",
                    initial_shape=initial_shape,
                    final_shape=[len(X_train), len(X_train.columns)],
                    executed_steps_count=executed_count,
                    execution_trace=execution_trace,
                    transformation_mapping=transformation_mapping,
                    execution_time_seconds=round(time.time() - start_time, 4),
                    error_message=err_msg,
                ),
            )

        final_shape = [len(X_train) + len(X_test), len(X_train.columns)]
        total_time = round(time.time() - start_time, 4)

        result = PlanExecutionResult(
            plan_id=plan.plan_id,
            status="SUCCESS",
            initial_shape=initial_shape,
            final_shape=final_shape,
            train_shape=[len(X_train), len(X_train.columns)],
            test_shape=[len(X_test), len(X_test.columns)],
            executed_steps_count=executed_count,
            step_logs=execution_trace,
            execution_trace=execution_trace,
            transformation_mapping=transformation_mapping,
            fitted_pipeline_info={"transformers_count": len(fitted_transformers), "keys": list(fitted_transformers.keys())},
            execution_time_seconds=total_time,
            metadata={"target_column": target_col},
        )

        return X_train, X_test, y_train, y_test, fitted_transformers, result

