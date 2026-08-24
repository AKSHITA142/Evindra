from typing import Any, List, Optional
import pandas as pd

from backend.schemas.experiment import ExperimentPlan, ExperimentSpec
from backend.schemas.mission_brief import MissionBrief
from backend.core.exceptions import ValidationException


class ExperimentValidator:
    """Validates experiment plans and specs against target dataset schema and mission constraints."""

    ALLOWED_OPERATION_TYPES = {
        "imputation", "impute",
        "encoding", "encode",
        "scaling", "scale",
        "feature_engineering", "engineer",
        "feature_selection", "select",
        "model", "modeling", "estimator", "classification", "regression",
    }

    ALLOWED_IMPUTATION_METHODS = {"mean", "median", "mode", "constant", "knn"}
    ALLOWED_ENCODING_METHODS = {"onehot", "ordinal", "frequency", "target"}
    ALLOWED_SCALING_METHODS = {"standard", "minmax", "robust"}

    def validate_spec(
        self,
        spec: ExperimentSpec,
        df: pd.DataFrame,
        target_column: str,
        mission_brief: Optional[MissionBrief] = None,
    ) -> None:
        """Validates a single ExperimentSpec against dataframe and constraints."""
        if target_column not in df.columns:
            raise ValidationException(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

        # Validate operations
        for op in spec.operations:
            if op.type not in self.ALLOWED_OPERATION_TYPES:
                raise ValidationException(f"Invalid operation type '{op.type}' in experiment '{spec.experiment_id}'")

            if op.type == "imputation" and op.method not in self.ALLOWED_IMPUTATION_METHODS:
                raise ValidationException(f"Invalid imputation method '{op.method}'")

            if op.type == "encoding" and op.method not in self.ALLOWED_ENCODING_METHODS:
                raise ValidationException(f"Invalid encoding method '{op.method}'")

            if op.type == "scaling" and op.method not in self.ALLOWED_SCALING_METHODS:
                raise ValidationException(f"Invalid scaling method '{op.method}'")

        # Validate mission constraints if present
        if mission_brief and mission_brief.constraints:
            forbidden = set(mission_brief.constraints.forbidden_operations)
            for op in spec.operations:
                if op.method in forbidden or op.type in forbidden:
                    raise ValidationException(
                        f"Operation '{op.method}' ({op.type}) is forbidden by mission constraints"
                    )

        # Validate Model & Dataset Compatibility
        model_clean = spec.model_name.lower().replace(" ", "").replace("_", "").replace("-", "")

        # 1. MultinomialNB requires non-negative values
        if model_clean in ("multinomialnb", "multinomial"):
            num_cols = df.select_dtypes(include=["number"]).columns
            num_cols = [c for c in num_cols if c != target_column]
            if len(num_cols) > 0 and (df[num_cols] < 0).any().any():
                raise ValidationException("MultinomialNB model requires non-negative feature values.")

        # 2. GaussianProcess dataset row limit (O(N^3) memory/compute bound)
        if "gaussianprocess" in model_clean and len(df) > 2000:
            raise ValidationException(
                f"GaussianProcess models are computationally prohibitive for datasets with >2000 rows (dataset has {len(df)} rows)."
            )

    def validate_plan(
        self,
        plan: ExperimentPlan,
        df: pd.DataFrame,
        target_column: str,
        mission_brief: Optional[MissionBrief] = None,
    ) -> None:
        """Validates an entire ExperimentPlan batch."""
        if not plan.experiments:
            raise ValidationException("ExperimentPlan contains no experiments to execute.")

        for spec in plan.experiments:
            self.validate_spec(spec, df, target_column, mission_brief)
