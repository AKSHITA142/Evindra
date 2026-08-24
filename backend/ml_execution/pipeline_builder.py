from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.pipeline import Pipeline

from backend.schemas.experiment import ExperimentOperation, ExperimentSpec
from backend.ml_execution.transformers import (
    ImputerTransformer,
    CategoricalEncoderTransformer,
    FeatureScalerTransformer,
)
from backend.ml_execution.feature_engineering import (
    LogTransformTransformer,
    InteractionFeaturesTransformer,
    PolynomialFeaturesTransformer,
    DatetimeDecompositionTransformer,
)
from backend.ml_execution.trainer import ModelTrainerFactory


class PipelineBuilder:
    """Constructs executable scikit-learn Pipelines in strict structural order."""

    STRUCTURAL_ORDER = [
        "datetime_decomposition",
        "imputation",
        "encoding",
        "scaling",
        "feature_engineering",
    ]

    def build_pipeline(
        self,
        spec: ExperimentSpec,
        task_type: str = "classification",
        random_state: int = 42,
    ) -> Pipeline:
        """Assembles a scikit-learn Pipeline from ExperimentSpec operations and model choice."""
        steps: List[tuple] = []

        # Always decompose datetime columns first if present
        steps.append(("datetime_decomp", DatetimeDecompositionTransformer()))

        # Categorize operations by structural type (normalizing aliases)
        ops_by_type: Dict[str, ExperimentOperation] = {}
        for op in spec.operations:
            op_type = op.type.lower().strip()
            if op_type in ("impute", "imputation"):
                ops_by_type["imputation"] = op
            elif op_type in ("encode", "encoding"):
                ops_by_type["encoding"] = op
            elif op_type in ("scale", "scaling"):
                ops_by_type["scaling"] = op
            elif op_type in ("engineer", "feature_engineering"):
                ops_by_type["feature_engineering"] = op
            elif op_type in ("model", "modeling", "estimator", "classification", "regression"):
                pass
            else:
                ops_by_type[op_type] = op

        # 1. Imputation step
        if "imputation" in ops_by_type:
            imp_op = ops_by_type["imputation"]
            steps.append(("imputer", ImputerTransformer(strategy=imp_op.method)))
        else:
            steps.append(("default_imputer", ImputerTransformer(strategy="median")))

        # 2. Categorical Encoding step
        col_encs = {}
        if "encoding" in ops_by_type:
            enc_op = ops_by_type["encoding"]
            col_encs = enc_op.params.get("column_encodings", {}) if enc_op.params else {}
            steps.append(("encoder", CategoricalEncoderTransformer(method=enc_op.method, column_encodings=col_encs)))
        else:
            steps.append(("default_encoder", CategoricalEncoderTransformer(method="onehot")))

        # 3. Feature Scaling step
        if "scaling" in ops_by_type:
            scale_op = ops_by_type["scaling"]
            col_scales = scale_op.params.get("column_scalings", {}) if scale_op.params else {}
            steps.append(("scaler", FeatureScalerTransformer(method=scale_op.method, column_scalings=col_scales, column_encodings=col_encs)))

        # 4. Feature Engineering step
        if "feature_engineering" in ops_by_type:
            fe_op = ops_by_type["feature_engineering"]
            if fe_op.method == "log":
                steps.append(("fe_log", LogTransformTransformer()))
            elif fe_op.method == "interaction":
                steps.append(("fe_interaction", InteractionFeaturesTransformer()))
            elif fe_op.method == "polynomial":
                steps.append(("fe_poly", PolynomialFeaturesTransformer()))

        # 5. Model Estimator step
        model_estimator = ModelTrainerFactory.get_estimator(
            model_name=spec.model_name,
            task_type=task_type,
            random_state=random_state,
        )
        steps.append(("model", model_estimator))

        return Pipeline(steps=steps)
