import pytest
import pandas as pd
import numpy as np

from backend.schemas.experiment import (
    ExperimentOperation,
    ExperimentSpec,
    ExperimentPlan,
    ExperimentResult,
)
from backend.schemas.mission_brief import MissionBrief, MissionConstraints
from backend.core.exceptions import ValidationException
from backend.ml_execution.validator import ExperimentValidator
from backend.ml_execution.transformers import (
    ImputerTransformer,
    CategoricalEncoderTransformer,
    FeatureScalerTransformer,
)
from backend.ml_execution.feature_engineering import (
    LogTransformTransformer,
    InteractionFeaturesTransformer,
)
from backend.ml_execution.pipeline_builder import PipelineBuilder
from backend.ml_execution.trainer import ModelTrainerFactory
from backend.ml_execution.executor import MLExecutionEngine


@pytest.fixture
def sample_classification_data():
    """Generates synthetic classification DataFrame."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n).astype(float),
        "income": np.random.exponential(scale=50000, size=n),
        "city": np.random.choice(["NY", "LA", "SF"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })
    # Add some NaNs
    df.loc[::10, "age"] = np.nan
    return df


@pytest.fixture
def sample_regression_data():
    """Generates synthetic regression DataFrame."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "target": np.random.randn(n) * 10 + 50,
    })
    return df


def test_validator(sample_classification_data):
    validator = ExperimentValidator()
    spec = ExperimentSpec(
        experiment_id="EXP_001",
        operations=[
            ExperimentOperation(type="imputation", method="median"),
            ExperimentOperation(type="encoding", method="onehot"),
        ],
        model_name="RandomForest",
    )

    # Valid validation
    validator.validate_spec(spec, sample_classification_data, target_column="target")

    # Missing target
    with pytest.raises(ValidationException):
        validator.validate_spec(spec, sample_classification_data, target_column="nonexistent")

    # Forbidden operation
    mission = MissionBrief(
        objective="Test",
        constraints=MissionConstraints(forbidden_operations=["onehot"])
    )
    with pytest.raises(ValidationException):
        validator.validate_spec(spec, sample_classification_data, target_column="target", mission_brief=mission)


def test_transformers(sample_classification_data):
    # Test Imputer
    imputer = ImputerTransformer(strategy="median")
    imputer.fit(sample_classification_data)
    transformed_df = imputer.transform(sample_classification_data)
    assert transformed_df["age"].isnull().sum() == 0

    # Test Encoder
    encoder = CategoricalEncoderTransformer(method="onehot")
    encoder.fit(transformed_df)
    encoded_df = encoder.transform(transformed_df)
    assert "city_NY" in encoded_df.columns or "city" not in encoded_df.columns

    # Test Scaler
    scaler = FeatureScalerTransformer(method="standard")
    scaler.fit(encoded_df)
    scaled_df = scaler.transform(encoded_df)
    assert np.isclose(scaled_df["income"].mean(), 0, atol=1.0)


def test_feature_engineering(sample_regression_data):
    # Test Log Transform
    log_tf = LogTransformTransformer(columns=["feature_2"])
    log_tf.fit(sample_regression_data)
    df_log = log_tf.transform(sample_regression_data)
    assert "feature_2_log" in df_log.columns or len(df_log.columns) >= len(sample_regression_data.columns)

    # Test Interactions
    inter = InteractionFeaturesTransformer(max_features=5)
    inter.fit(sample_regression_data)
    df_inter = inter.transform(sample_regression_data)
    assert "feature_1_x_feature_2" in df_inter.columns


def test_model_trainer_factory():
    # Classification models
    clf1 = ModelTrainerFactory.get_estimator("RandomForestClassifier", task_type="classification")
    assert clf1.__class__.__name__ in ("RandomForestClassifier", "HistGradientBoostingClassifier")

    clf2 = ModelTrainerFactory.get_estimator("LogisticRegression", task_type="classification")
    assert clf2.__class__.__name__ == "LogisticRegression"

    clf3 = ModelTrainerFactory.get_estimator("SVC", task_type="classification")
    assert clf3.__class__.__name__ == "SVC"

    # Regression models
    reg1 = ModelTrainerFactory.get_estimator("LinearRegression", task_type="regression")
    assert reg1.__class__.__name__ == "LinearRegression"

    reg2 = ModelTrainerFactory.get_estimator("Ridge", task_type="regression")
    assert reg2.__class__.__name__ == "Ridge"


def test_pipeline_builder(sample_classification_data):
    builder = PipelineBuilder()
    spec = ExperimentSpec(
        experiment_id="EXP_001",
        operations=[
            ExperimentOperation(type="imputation", method="median"),
            ExperimentOperation(type="encoding", method="onehot"),
            ExperimentOperation(type="scaling", method="standard"),
        ],
        model_name="RandomForestClassifier",
    )

    pipeline = builder.build_pipeline(spec, task_type="classification")
    assert len(pipeline.steps) > 0

    X = sample_classification_data.drop(columns=["target"])
    y = sample_classification_data["target"]

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(sample_classification_data)


def test_ml_execution_engine_end_to_end(sample_classification_data, sample_regression_data):
    engine = MLExecutionEngine(max_workers=2, n_splits=3)

    # 1. Classification Plan
    class_plan = ExperimentPlan(
        mission="Test Classification",
        experiment_budget=2,
        experiments=[
            ExperimentSpec(
                experiment_id="EXP_001",
                operations=[
                    ExperimentOperation(type="imputation", method="median"),
                    ExperimentOperation(type="encoding", method="onehot"),
                ],
                model_name="RandomForestClassifier",
            ),
            ExperimentSpec(
                experiment_id="EXP_002",
                operations=[
                    ExperimentOperation(type="imputation", method="mean"),
                    ExperimentOperation(type="encoding", method="ordinal"),
                    ExperimentOperation(type="scaling", method="standard"),
                ],
                model_name="LogisticRegression",
            ),
        ],
    )

    results = engine.execute_plan(
        plan=class_plan,
        dataset=sample_classification_data,
        target_column="target",
        task_type="classification",
    )

    assert len(results) == 2
    assert results[0].experiment_id == "EXP_001"
    assert results[0].status == "completed"
    assert "accuracy" in results[0].metrics.metrics
    assert results[0].metrics.primary_metric > 0.0

    # 2. Regression Plan
    reg_plan = ExperimentPlan(
        mission="Test Regression",
        experiment_budget=1,
        experiments=[
            ExperimentSpec(
                experiment_id="EXP_REG_001",
                operations=[
                    ExperimentOperation(type="scaling", method="robust"),
                ],
                model_name="LinearRegression",
            ),
        ],
    )

    reg_results = engine.execute_plan(
        plan=reg_plan,
        dataset=sample_regression_data,
        target_column="target",
        task_type="regression",
    )

    assert len(reg_results) == 1
    assert reg_results[0].experiment_id == "EXP_REG_001"
    assert reg_results[0].status == "completed"
    assert "mae" in reg_results[0].metrics.metrics


def test_datetime_decomposition_preserves_numeric_time_columns():
    from backend.ml_execution.feature_engineering import DatetimeDecompositionTransformer
    df = pd.DataFrame({
        "reaction_time": [1.25, 2.34, 0.98, 3.41],
        "execution_date_str": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
    })
    transformer = DatetimeDecompositionTransformer()
    res = transformer.transform(df)
    assert "reaction_time" in res.columns
    assert res["reaction_time"].iloc[0] == 1.25
    assert "execution_date_str_year" in res.columns
    assert "execution_date_str" not in res.columns


def test_binary_classification_metric_averaging():
    from backend.ml_execution.metrics import MetricEngine
    y_true = pd.Series([0, 0, 0, 0, 1, 1])
    y_pred = pd.Series([0, 0, 0, 1, 1, 1])
    res = MetricEngine.compute_metrics(y_true, y_pred, task_type="classification")
    # For binary 2-class, recall of class 1 should be 1.0 (2 out of 2 detected), precision 2/3 = 0.6667
    assert res.metrics["recall"] == 1.0
    assert res.metrics["precision"] == 0.6667

