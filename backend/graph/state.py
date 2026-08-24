from typing import TypedDict, Optional, List, Dict, Any
from backend.schemas.state import WorkflowState
from backend.schemas.enums import JobStatus


class WorkflowStateDict(TypedDict, total=False):
    """
    LangGraph state channel dictionary.
    Defines keys passed between nodes in the StateGraph.
    """
    job_id: str
    dataset_id: str
    file_path: Optional[str]
    job_status: str
    user_goal: Optional[str]
    user_task_type: Optional[str]
    semantic_profile: Optional[Dict[str, Any]]
    mission_brief: Optional[Dict[str, Any]]
    experiment_plan: Optional[Dict[str, Any]]
    experiment_results: List[Dict[str, Any]]
    evaluation_report: Optional[Dict[str, Any]]
    knowledge_base: List[Dict[str, Any]]
    decision: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    error_message: Optional[str]


def create_initial_state(
    dataset_id: str,
    job_id: Optional[str] = None,
    file_path: Optional[str] = None,
    user_goal: Optional[str] = None,
    user_task_type: Optional[str] = "general",
    max_iterations: int = 5,
) -> WorkflowStateDict:
    """Creates initial state dictionary for starting a research job graph."""
    return {
        "job_id": job_id or f"job_{dataset_id}",
        "dataset_id": dataset_id,
        "file_path": file_path,
        "job_status": JobStatus.QUEUED.value,
        "user_goal": user_goal,
        "user_task_type": user_task_type or "general",
        "semantic_profile": None,
        "mission_brief": None,
        "experiment_plan": None,
        "experiment_results": [],
        "evaluation_report": None,
        "knowledge_base": [],
        "decision": None,
        "final_report": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "error_message": None,
    }
