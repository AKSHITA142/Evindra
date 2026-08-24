import os
import tempfile
import pytest
import pandas as pd
import numpy as np

from backend.graph import (
    WorkflowStateDict, create_initial_state, build_research_graph, compile_graph, route_next
)
from backend.graph.nodes import (
    profiling_node, understanding_node, planning_node, execution_node, evaluation_node, decision_node, reporting_node
)
from backend.schemas.enums import JobStatus, DecisionType


@pytest.fixture
def synthetic_csv_file():
    """Fixture creating a temporary synthetic classification CSV file."""
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, size=50),
        "income": np.random.uniform(20000, 100000, size=50),
        "target": np.random.choice([0, 1], size=50),
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name

    yield path

    if os.path.exists(path):
        os.remove(path)


def test_create_initial_state():
    """Verify initial state creation."""
    state = create_initial_state(dataset_id="ds_100", user_goal="Predict target column")
    assert state["dataset_id"] == "ds_100"
    assert state["job_status"] == JobStatus.QUEUED.value
    assert state["iteration_count"] == 0
    assert state["max_iterations"] == 5


def test_route_next_logic():
    """Verify conditional router logic."""
    # STOP decision -> reporting
    state_stop = create_initial_state(dataset_id="ds_1")
    state_stop["decision"] = {"decision": DecisionType.STOP.value}
    assert route_next(state_stop) == "reporting"

    # CONTINUE decision -> planning
    state_cont = create_initial_state(dataset_id="ds_1")
    state_cont["decision"] = {"decision": DecisionType.CONTINUE.value}
    assert route_next(state_cont) == "planning"

    # Budget limit reached -> reporting
    state_budget = create_initial_state(dataset_id="ds_1", max_iterations=2)
    state_budget["iteration_count"] = 2
    assert route_next(state_budget) == "reporting"

    # Failed job -> __end__
    state_failed = create_initial_state(dataset_id="ds_1")
    state_failed["job_status"] = JobStatus.FAILED.value
    assert route_next(state_failed) == "__end__"


def test_nodes_sequential_execution(synthetic_csv_file):
    """Verify sequential node execution manually."""
    state = create_initial_state(dataset_id="ds_test", file_path=synthetic_csv_file)

    state = profiling_node(state)
    assert state["job_status"] == JobStatus.PROFILING.value
    assert state["semantic_profile"] is not None

    state = understanding_node(state)
    assert state["job_status"] == JobStatus.PLANNING.value
    assert state["mission_brief"] is not None

    state = planning_node(state)
    assert state["job_status"] == JobStatus.EXECUTING.value
    assert state["experiment_plan"] is not None

    state = execution_node(state)
    assert state["job_status"] == JobStatus.EVALUATING.value
    assert len(state["experiment_results"]) >= 2

    state = evaluation_node(state)
    assert state["evaluation_report"] is not None
    assert len(state["knowledge_base"]) >= 1

    state = decision_node(state)
    assert state["decision"] is not None

    state = reporting_node(state)
    assert state["job_status"] == JobStatus.COMPLETED.value
    assert state["final_report"] is not None


def test_full_langgraph_workflow_execution(synthetic_csv_file):
    """Verify compiled LangGraph state machine execution end-to-end."""
    app = compile_graph()

    initial_state = create_initial_state(
        dataset_id="ds_graph_test",
        file_path=synthetic_csv_file,
        user_goal="Optimize classification model",
        max_iterations=2,
    )

    config = {"configurable": {"thread_id": "thread_1"}}
    final_state = app.invoke(initial_state, config=config)

    assert final_state["job_status"] == JobStatus.COMPLETED.value
    assert final_state["semantic_profile"] is not None
    assert final_state["mission_brief"] is not None
    assert len(final_state["experiment_results"]) >= 2
    assert final_state["evaluation_report"] is not None
    assert final_state["final_report"] is not None
