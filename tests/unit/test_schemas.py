import pytest
from pydantic import ValidationError

from backend.schemas import (
    JobStatus,
    DecisionType,
    SeverityLevel,
    ColumnType,
    ColumnProfile,
    QualityIssue,
    SemanticProfile,
    MissionConstraints,
    MissionBrief,
    ExperimentOperation,
    ExperimentSpec,
    ExperimentPlan,
    PipelineDefinition,
    MetricsResult,
    ExperimentResult,
    RankingItem,
    KnowledgeFinding,
    EvaluationReport,
    ResearchDirectorDecision,
    FinalRecommendation,
    WorkflowState,
    SuccessResponse,
    ErrorResponse,
)


def test_enums():
    assert JobStatus.QUEUED.value == "queued"
    assert DecisionType.STOP.value == "stop"
    assert SeverityLevel.CRITICAL.value == "critical"
    assert ColumnType.NUMERIC.value == "numeric"


def test_confidence_validation():
    # Valid confidence
    issue = QualityIssue(problem="high_skew", confidence=0.85)
    assert issue.confidence == 0.85

    # Invalid confidence > 1.0
    with pytest.raises(ValidationError):
        QualityIssue(problem="high_skew", confidence=1.5)

    # Invalid confidence < 0.0
    with pytest.raises(ValidationError):
        QualityIssue(problem="high_skew", confidence=-0.1)


def test_semantic_profile_serialization():
    col = ColumnProfile(name="age", type=ColumnType.NUMERIC, missing_pct=5.0)
    issue = QualityIssue(problem="missing_values", confidence=0.9, affected_columns=["age"])
    profile = SemanticProfile(
        dataset_summary={"rows": 1000, "columns": 5},
        column_profiles=[col],
        quality_issues=[issue]
    )

    json_data = profile.model_dump()
    assert json_data["dataset_summary"]["rows"] == 1000
    assert json_data["column_profiles"][0]["name"] == "age"
    assert json_data["quality_issues"][0]["problem"] == "missing_values"

    # Test roundtrip
    restored = SemanticProfile.model_validate(json_data)
    assert restored.dataset_summary["rows"] == 1000
    assert restored.column_profiles[0].name == "age"


def test_mission_brief_defaults():
    mb = MissionBrief(objective="Build fraud detection model")
    assert mb.objective == "Build fraud detection model"
    assert mb.constraints.max_row_loss == 0.05
    assert mb.constraints.training_time_limit_minutes == 30
    assert mb.dataset_characteristics.domain == "General"


def test_experiment_plan_and_result():
    op = ExperimentOperation(type="imputation", method="median")
    spec = ExperimentSpec(experiment_id="EXP_001", dataset_name="test_dataset", operations=[op])
    plan = ExperimentPlan(plan_id="PLAN_001", experiments=[spec])

    assert len(plan.experiments) == 1
    assert plan.experiments[0].operations[0].method == "median"

    pipeline = PipelineDefinition(pipeline_id="PIPE_001", steps=[{"type": "imputation", "method": "median"}])
    metrics = MetricsResult(primary_metric=0.92, metrics={"f1": 0.92, "roc_auc": 0.95})
    result = ExperimentResult(
        experiment_id="EXP_001",
        model="CatBoost",
        metrics=metrics,
        runtime=12.5,
    )

    assert result.experiment_id == "EXP_001"
    assert result.metrics.primary_metric == 0.92


def test_evaluation_report_and_decision():
    item = RankingItem(rank=1, experiment_id="EXP_001", score=0.92, model="CatBoost")
    finding = KnowledgeFinding(finding="Median imputation improved score", confidence=0.9)
    report = EvaluationReport(winner="EXP_001", ranking=[item], knowledge=[finding])

    assert report.winner == "EXP_001"
    assert report.ranking[0].rank == 1

    decision = ResearchDirectorDecision(
        decision=DecisionType.CONTINUE,
        confidence=0.88,
        knowledge=["Median imputation confirmed"]
    )
    assert decision.decision == DecisionType.CONTINUE.value


def test_workflow_state():
    state = WorkflowState(dataset_id="ds_12345", job_status=JobStatus.PLANNING)
    assert state.dataset_id == "ds_12345"
    assert state.job_status == "planning"
    assert state.experiment_results == []

    # Roundtrip test
    dumped = state.model_dump()
    loaded = WorkflowState.model_validate(dumped)
    assert loaded.dataset_id == "ds_12345"
    assert loaded.job_status == JobStatus.PLANNING


def test_response_envelopes():
    succ = SuccessResponse(data={"job_id": "job_99"})
    assert succ.data["job_id"] == "job_99"

    err = ErrorResponse(error_code="NOT_FOUND", message="Resource missing")
    assert err.error_code == "NOT_FOUND"
