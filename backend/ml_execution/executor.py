from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union, Dict, Any, Tuple
import logging

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


from backend.schemas.experiment import (
    ExperimentPlan,
    ExperimentSpec,
    ExperimentResult,
    PipelineDefinition,
    Artifacts,
)
from backend.schemas.mission_brief import MissionBrief
from backend.ml_execution.validator import ExperimentValidator
from backend.ml_execution.pipeline_builder import PipelineBuilder
from backend.ml_execution.cross_validation import CrossValidationRunner
from backend.ml_execution.metrics import MetricEngine
from backend.ml_execution.logger import ExperimentLogger
from backend.ml_execution.pre_cleaner import DataPreCleaner

logger = logging.getLogger("datapilot.ml_execution.executor")


class MLExecutionEngine:
    """Deterministic ML Execution Engine that executes experiment batches starting from original data."""

    def __init__(self, max_workers: int = 4, n_splits: int = 5, random_state: int = 42):
        self.max_workers = max_workers
        self.n_splits = n_splits
        self.random_state = random_state
        self.validator = ExperimentValidator()
        self.pipeline_builder = PipelineBuilder()
        self.cv_runner = CrossValidationRunner(n_splits=self.n_splits, random_state=self.random_state)

    @staticmethod
    def _is_subtoken_metadata(col_name: str) -> bool:
        """
        Splits column header by delimiters (_, -, space, dot) to check if any token
        matches metadata keywords (name, patient_name, doctor_id, client_email, etc.).
        """
        import re
        col_str = str(col_name).lower().strip()
        meta_root_tokens = {
            "id", "name", "email", "ssn", "token", "hash", "uuid", "address",
            "phone", "code", "index", "rowid", "guid", "number"
        }
        tokens = [t for t in re.split(r"[_\-\s\.]+", col_str) if t]
        for token in tokens:
            if token in meta_root_tokens or (len(token) > 2 and (token.endswith("_id") or token.startswith("id_"))):
                return True
        return False

    @staticmethod
    def _extract_meta_and_features(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separates non-predictive identifier/metadata columns (id, patient_name, doctor_id, email, etc.)
        from ML feature columns X to prevent target memorization, feature explosion, and preserve metadata in export.
        """
        meta_cols = []
        n_rows = len(df)
        for col in df.columns:
            if col == target_column:
                continue

            # 1. Sub-token pattern matching (patient_name, doctor_id, client_email, etc.)
            if MLExecutionEngine._is_subtoken_metadata(col):
                meta_cols.append(col)
                continue

            # 2. High-cardinality string/object text columns (unique_ratio >= 0.70)
            if n_rows > 3 and (df[col].dtype == object or str(df[col].dtype) in ("category", "string")):
                unique_ratio = df[col].nunique() / float(n_rows)
                if unique_ratio >= 0.70:
                    meta_cols.append(col)

        meta_df = df[meta_cols].copy()
        feature_cols = [c for c in df.columns if c not in meta_cols and c != target_column]
        features_df = df[feature_cols].copy()
        return meta_df, features_df

    def execute_single_experiment(
        self,
        spec: ExperimentSpec,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        mission_brief: Optional[MissionBrief] = None,
    ) -> ExperimentResult:
        """Executes a single experiment spec starting from the original, unmodified dataset."""
        logger_inst = ExperimentLogger(spec.experiment_id)
        logger_inst.start()

        # Create fresh copy of original dataset for complete isolation
        df_copy = df.copy()

        try:
            # 0. Leakage-Safe Data Pre-Cleaning (Deduplication, target nulls, >75% missing cols, >50% sparse rows)
            df_cleaned, cleaning_audit = DataPreCleaner.clean_raw_dataset(df_copy, target_column)

            # Separate metadata (id, name), features X, and target y from pre-cleaned data
            meta_df, X = self._extract_meta_and_features(df_cleaned, target_column)
            y = df_cleaned[target_column]

            # Single LabelEncoder for classification: encode ONCE before splitting
            # This guarantees consistent integer labels [0..K-1] across train and test.
            if task_type == "classification":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

            # 1b. Perform 80/20 train/test split BEFORE fitting or CV
            from sklearn.model_selection import train_test_split

            stratify_target = None
            if task_type == "classification" and len(y) >= 10:
                class_counts = pd.Series(y).value_counts()
                if len(class_counts) > 1 and class_counts.min() >= 2:
                    stratify_target = y

            try:
                X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
                    X, y, meta_df,
                    test_size=0.2,
                    random_state=self.random_state,
                    stratify=stratify_target,
                )
            except Exception:
                # Fallback to non-stratified split if sample size or class distribution prevents stratification
                X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
                    X, y, meta_df,
                    test_size=0.2,
                    random_state=self.random_state,
                    stratify=None,
                )


            # 2. Build Pipeline
            pipeline = self.pipeline_builder.build_pipeline(
                spec=spec,
                task_type=task_type,
                random_state=self.random_state,
            )

            # 3. Cross-Validation on training split X_train, y_train only
            cv_scores, fitted_pipeline = self.cv_runner.run_cv(
                pipeline=pipeline,
                X=X_train,
                y=y_train,
                task_type=task_type,
            )

            # Refit pipeline on full training set
            fitted_pipeline.fit(X_train, y_train)

            # 4. Predict & evaluate ONCE on held-out test split (X_test, y_test)
            y_pred_test = fitted_pipeline.predict(X_test)
            y_proba_test = None
            if task_type == "classification" and hasattr(fitted_pipeline, "predict_proba"):
                try:
                    y_proba_test = fitted_pipeline.predict_proba(X_test)
                except Exception:
                    pass

            # Compute actual training score matching task type for honest train_test_gap
            if task_type == "classification":
                train_score = float(fitted_pipeline.score(X_train, y_train))
            else:
                from sklearn.metrics import mean_squared_error
                y_pred_train = fitted_pipeline.predict(X_train)
                train_score = abs(float(np.sqrt(mean_squared_error(y_train, y_pred_train))))

            # 5. Compute Metrics on held-out test split with CV scores
            user_goal_text = mission_brief.user_goal if (mission_brief and hasattr(mission_brief, "user_goal")) else ""
            metrics_result = MetricEngine.compute_metrics(
                y_true=y_test,
                y_pred=y_pred_test,
                y_proba=y_proba_test,
                task_type=task_type,
                cv_scores=cv_scores,
                train_score=train_score,
                user_goal=user_goal_text,
                target_column=target_column,
                column_names=list(X.columns) if hasattr(X, "columns") else None,
            )

            # 6. Extract Feature Importances with real column names
            feature_importance: Optional[Dict[str, float]] = None
            try:
                # Get real feature names from the preprocessing pipeline output
                feature_names = None
                try:
                    preproc_steps = Pipeline(fitted_pipeline.steps[:-1])
                    X_sample = preproc_steps.transform(X_train.iloc[:1])
                    if isinstance(X_sample, pd.DataFrame):
                        feature_names = list(X_sample.columns)
                except Exception:
                    pass

                model_step = fitted_pipeline.named_steps.get("model")
                if hasattr(model_step, "feature_importances_"):
                    importances = model_step.feature_importances_
                    if feature_names and len(feature_names) == len(importances):
                        feature_importance = {str(name): float(val) for name, val in zip(feature_names, importances)}
                    else:
                        feature_importance = {f"feature_{i}": float(val) for i, val in enumerate(importances)}
                elif hasattr(model_step, "coef_"):
                    coefs = np.abs(model_step.coef_).flatten()
                    if feature_names and len(feature_names) == len(coefs):
                        feature_importance = {str(name): float(val) for name, val in zip(feature_names, coefs)}
                    else:
                        feature_importance = {f"feature_{i}": float(val) for i, val in enumerate(coefs)}
            except Exception:
                pass

            # 7. Generate and save preprocessed dataset CSV artifact
            processed_csv_path: Optional[str] = None
            try:
                import os
                if len(fitted_pipeline.steps) > 1:
                    preproc_pipe = Pipeline(fitted_pipeline.steps[:-1])
                    X_trans = preproc_pipe.transform(X)
                    if isinstance(X_trans, pd.DataFrame):
                        clean_features_df = X_trans.copy()
                    else:
                        if hasattr(X_trans, "toarray"):
                            X_trans = X_trans.toarray()
                        clean_features_df = pd.DataFrame(X_trans, index=X.index)
                else:
                    clean_features_df = X.copy()

                os.makedirs("storage/artifacts", exist_ok=True)

                # Format 1: Business Action CSV (Original raw columns + Predictions, unscaled & unencoded for human readability)
                business_df = df_cleaned.copy().reset_index(drop=True)
                if 'y_pred_test' in locals() and y_pred_test is not None and len(y_pred_test) == len(business_df):
                    if 'le' in locals() and hasattr(le, 'inverse_transform'):
                        try:
                            business_df["predicted_" + str(target_column)] = le.inverse_transform(y_pred_test)
                        except Exception:
                            business_df["predicted_" + str(target_column)] = y_pred_test
                    else:
                        business_df["predicted_" + str(target_column)] = y_pred_test

                business_csv_path = f"storage/artifacts/{spec.experiment_id}_business_action.csv"
                business_df.to_csv(business_csv_path, index=False)

                # Format 2: ML-Ready Feature Matrix CSV (Pure engineered numeric X and target y)
                ml_df = clean_features_df.copy().reset_index(drop=True)
                ml_df[target_column] = y.values

                ml_ready_csv_path = f"storage/artifacts/{spec.experiment_id}_ml_ready.csv"
                ml_df.to_csv(ml_ready_csv_path, index=False)

                # Legacy fallback compatibility
                legacy_csv_path = f"storage/artifacts/{spec.experiment_id}_cleaned.csv"
                business_df.to_csv(legacy_csv_path, index=False)

                processed_csv_path = ml_ready_csv_path
            except Exception as pe:
                logger.warning(f"Could not export preprocessed dataset CSV: {pe}")

            artifacts = Artifacts(
                processed_dataset_path=processed_csv_path,
                feature_importance=feature_importance,
                cleaning_audit=cleaning_audit.to_dict() if 'cleaning_audit' in locals() else None,
            )
            runtime = logger_inst.finish(status="completed", metrics=metrics_result.metrics)


            pipeline_def = PipelineDefinition(
                operations=spec.operations,
                model_name=spec.model_name,
            )

            return ExperimentResult(
                experiment_id=spec.experiment_id,
                pipeline=pipeline_def,
                model=spec.model_name,
                metrics=metrics_result,
                runtime=runtime,
                status="completed",
                artifacts=artifacts,
            )

        except Exception as e:
            runtime = logger_inst.finish(status="failed")
            logger_inst.log_error(e)

            pipeline_def = PipelineDefinition(
                operations=spec.operations,
                model_name=spec.model_name,
            )

            return ExperimentResult(
                experiment_id=spec.experiment_id,
                pipeline=pipeline_def,
                model=spec.model_name,
                metrics=MetricEngine.compute_metrics([], [], task_type=task_type),
                runtime=runtime,
                status="failed",
                error_message=str(e),
            )

    def execute_plan(
        self,
        plan: ExperimentPlan,
        dataset: Union[pd.DataFrame, str],
        target_column: str,
        task_type: str = "classification",
        mission_brief: Optional[MissionBrief] = None,
    ) -> List[ExperimentResult]:
        """Executes a batch ExperimentPlan in parallel starting from the original dataset."""
        # Load dataset if CSV file path provided
        if isinstance(dataset, str):
            df = pd.read_csv(dataset)
        else:
            df = dataset.copy()

        # Validate entire plan
        self.validator.validate_plan(plan, df, target_column, mission_brief)

        results: List[ExperimentResult] = []

        # Run experiments in parallel
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(plan.experiments))) as executor:
            future_to_spec = {
                executor.submit(
                    self.execute_single_experiment,
                    spec,
                    df,
                    target_column,
                    task_type,
                    mission_brief,
                ): spec
                for spec in plan.experiments
            }

            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    logger.error(f"Experiment {spec.experiment_id} generated an exception: {exc}")
                    pipeline_def = PipelineDefinition(operations=spec.operations, model_name=spec.model_name)
                    results.append(
                        ExperimentResult(
                            experiment_id=spec.experiment_id,
                            pipeline=pipeline_def,
                            model=spec.model_name,
                            metrics=MetricEngine.compute_metrics([], [], task_type=task_type),
                            runtime=0.0,
                            status="failed",
                            error_message=str(exc),
                        )
                    )

        # Sort results by experiment priority / ID
        results.sort(key=lambda r: r.experiment_id)
        return results
