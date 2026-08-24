import pandas as pd
from backend.profiling.target_analyzer import TargetAnalyzer
from backend.agents.strategy_planner import StrategyPlannerAgent
from backend.graph.state import create_initial_state
from backend.graph.nodes import profiling_node, planning_node


def test_task_type_explicit_classification_override():
    """Verifies explicit 'classification' selection overrides float target auto-detection."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        "target_col": [10.5, 20.3, 30.1, 40.8, 50.2] * 10,
    })

    # 1. TargetAnalyzer override test
    res_clf = TargetAnalyzer.analyze_target(df, target_column="target_col", user_task_type="classification")
    assert res_clf["task_type"] == "classification"

    # 2. StrategyPlannerAgent override test
    planner = StrategyPlannerAgent()
    plan = planner.get_fallback_data({
        "task_type": "classification",
        "experiment_budget": 3,
        "semantic_profile": {"dataset_summary": {"row_count": 50, "target": {"task_type": "classification"}}},
    })
    models = [exp["model_name"] for exp in plan["experiments"]]
    for m in models:
        assert "Regressor" not in m and m not in ("LinearRegression", "SVR")


def test_task_type_explicit_regression_override():
    """Verifies explicit 'regression' selection overrides categorical integer target auto-detection."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        "target_col": [0, 1, 0, 1, 0] * 10,
    })

    # 1. TargetAnalyzer override test
    res_reg = TargetAnalyzer.analyze_target(df, target_column="target_col", user_task_type="regression")
    assert res_reg["task_type"] == "regression"

    # 2. StrategyPlannerAgent override test
    planner = StrategyPlannerAgent()
    plan = planner.get_fallback_data({
        "task_type": "regression",
        "experiment_budget": 3,
        "semantic_profile": {"dataset_summary": {"row_count": 50, "target": {"task_type": "regression"}}},
    })
    models = [exp["model_name"] for exp in plan["experiments"]]
    for m in models:
        assert "Classifier" not in m and m not in ("LogisticRegression", "SVC")


def test_initial_state_preserves_task_type():
    """Verifies initial state initializes user_task_type correctly."""
    state = create_initial_state(dataset_id="ds_123", user_task_type="classification")
    assert state.get("user_task_type") == "classification"
