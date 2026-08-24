import pytest

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.pipeline import PipelineCandidateSet, PipelineCandidate
from backend.engine.pipeline_generator import PipelineGenerator


def test_small_classification_pipelines():
    """Verify candidate generation for small classification dataset profile."""
    col_age = ColumnProfileExtended(name="age", normalized_dtype="numeric")
    col_income = ColumnProfileExtended(name="income", normalized_dtype="numeric")
    prof = DatasetProfile(dataset_name="small_cls", rows=200, columns=2, detailed_column_profiles=[col_age, col_income], target_column="label", problem_type="binary_classification")

    generator = PipelineGenerator(max_pipelines=4)
    p_set = generator.generate_candidate_pipelines(prof)

    assert isinstance(p_set, PipelineCandidateSet)
    assert p_set.total_candidates > 0
    assert len(p_set.pipelines) <= 4
    for p in p_set.pipelines:
        assert p.pipeline_id != ""
        assert p.name != ""
        assert p.estimated_cost in ("LOW", "MEDIUM", "HIGH")


def test_large_classification_pipelines():
    """Verify candidate generation for large classification dataset profile."""
    col_f1 = ColumnProfileExtended(name="f1", normalized_dtype="numeric")
    col_c1 = ColumnProfileExtended(name="c1", normalized_dtype="categorical", distinct_count=50)
    prof = DatasetProfile(dataset_name="large_cls", rows=100000, columns=2, detailed_column_profiles=[col_f1, col_c1], target_column="label", problem_type="multiclass_classification")

    generator = PipelineGenerator(max_pipelines=3)
    p_set = generator.generate_candidate_pipelines(prof)

    assert len(p_set.pipelines) == 3
    # Verify ordinal/frequency encoding chosen over OHE for large dataset
    p1 = p_set.pipelines[0]
    assert any(s.action in ("ORDINAL_ENCODING", "FREQUENCY_ENCODING", "ONE_HOT_ENCODING") for s in p1.preprocessing_plan.steps)


def test_regression_pipelines():
    """Verify candidate generation for regression problem type."""
    col_sqft = ColumnProfileExtended(name="sqft", normalized_dtype="numeric")
    col_beds = ColumnProfileExtended(name="beds", normalized_dtype="numeric")
    prof = DatasetProfile(dataset_name="house_reg", rows=500, columns=2, detailed_column_profiles=[col_sqft, col_beds], target_column="price", problem_type="regression")

    generator = PipelineGenerator(max_pipelines=4)
    p_set = generator.generate_candidate_pipelines(prof)

    assert p_set.problem_type == "regression"
    for p in p_set.pipelines:
        model_family = p.model_spec.get("model_family", "")
        assert "RIDGE" in model_family or "ELASTICNET" in model_family or "GRADIENT_BOOSTING" in model_family or "RANDOM_FOREST" in model_family



def test_categorical_heavy_pipelines():
    """Verify candidate generation for categorical-heavy dataset profile."""
    cols = [ColumnProfileExtended(name=f"cat_{i}", normalized_dtype="categorical") for i in range(5)]
    cols.append(ColumnProfileExtended(name="num_1", normalized_dtype="numeric"))
    prof = DatasetProfile(dataset_name="cat_heavy", rows=1000, columns=6, detailed_column_profiles=cols, target_column="target", problem_type="classification")

    generator = PipelineGenerator(max_pipelines=4)
    p_set = generator.generate_candidate_pipelines(prof)

    assert p_set.total_candidates > 0


def test_numeric_only_pipelines():
    """Verify candidate generation for numeric-only dataset profile."""
    cols = [ColumnProfileExtended(name=f"n_{i}", normalized_dtype="numeric") for i in range(4)]
    prof = DatasetProfile(dataset_name="num_only", rows=800, columns=4, detailed_column_profiles=cols, target_column="target", problem_type="classification")

    generator = PipelineGenerator(max_pipelines=4)
    p_set = generator.generate_candidate_pipelines(prof)

    for p in p_set.pipelines:
        steps = p.preprocessing_plan.steps
        # Verify no unnecessary categorical encoding step in numeric only
        assert not any(s.stage == "ENCODING" for s in steps)


def test_mixed_dataset_pipelines():
    """Verify candidate generation for mixed (numeric + categorical + datetime + text) dataset profile."""
    col_num = ColumnProfileExtended(name="num", normalized_dtype="numeric")
    col_cat = ColumnProfileExtended(name="cat", normalized_dtype="categorical")
    col_dt = ColumnProfileExtended(name="dt", normalized_dtype="datetime")
    prof = DatasetProfile(dataset_name="mixed_ds", rows=500, columns=3, detailed_column_profiles=[col_num, col_cat, col_dt], target_column="target", problem_type="classification")

    generator = PipelineGenerator(max_pipelines=4)
    p_set = generator.generate_candidate_pipelines(prof)

    assert p_set.total_candidates == 4


def test_max_pipelines_capping():
    """Verify max_pipelines limit strictly caps returned candidates."""
    col_num = ColumnProfileExtended(name="num", normalized_dtype="numeric")
    prof = DatasetProfile(dataset_name="cap_ds", rows=500, columns=1, detailed_column_profiles=[col_num], target_column="target", problem_type="classification")

    generator = PipelineGenerator(max_pipelines=2)
    p_set = generator.generate_candidate_pipelines(prof)

    assert len(p_set.pipelines) == 2
