import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from backend.schemas.enums import JobStatus, DecisionType
from backend.schemas.semantic_profile import SemanticProfile
from backend.schemas.mission_brief import MissionBrief, MissionConstraints
from backend.schemas.experiment import ExperimentPlan, ExperimentResult
from backend.schemas.evaluation import EvaluationReport, ResearchDirectorDecision
from backend.schemas.report import FinalRecommendation

from backend.profiling import ProfilingEngine
from backend.profiling.loader import DataLoader
from backend.core.config import get_settings
from backend.ml_execution.executor import MLExecutionEngine
from backend.evaluation.evaluator import EvaluationEngine

from backend.agents import (
    DatasetUnderstandingAgent,
    ConstraintGoalAnalyzer,
    StrategyPlannerAgent,
    ResearchDirectorAgent,
    ReportGeneratorAgent,
)
from backend.graph.state import WorkflowStateDict

logger = logging.getLogger("datapilot.graph.nodes")


def _load_state_dataset_bytes(state: WorkflowStateDict) -> bytes:
    """Helper to fetch dataset bytes from local disk or Supabase Cloud Storage."""
    file_path = state.get("file_path")
    dataset_id = state.get("dataset_id") or ""

    # 1. Check local disk if available
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return f.read()

    # 2. Fetch directly from Supabase Cloud Storage
    try:
        from backend.services.storage.supabase_storage import SupabaseStorageService
        storage_svc = SupabaseStorageService()
        if storage_svc.is_configured:
            filename = os.path.basename(file_path) if file_path else "dataset.csv"
            remote_path = file_path if (file_path and "/" in file_path and not os.path.isabs(file_path)) else f"{dataset_id}/{filename}"
            return storage_svc.download_bytes(remote_path)
    except Exception as se:
        logger.warning(f"Failed to fetch dataset bytes from Supabase Storage: {se}")

    raise FileNotFoundError(f"Dataset '{dataset_id}' could not be loaded from cloud storage or local path: {file_path}")


def profiling_node(state: WorkflowStateDict) -> WorkflowStateDict:
    """Executes Phase 6 ProfilingEngine on dataset bytes in memory."""
    try:
        file_bytes = _load_state_dataset_bytes(state)
    except Exception as e:
        return {
            **state,
            "job_status": JobStatus.FAILED.value,
            "error_message": f"Dataset file not found: {str(e)}",
        }

    try:
        user_task_type = state.get("user_task_type") or "general"
        user_goal = state.get("user_goal") or ""
        filename = os.path.basename(state.get("file_path") or "dataset.csv")

        profile, hints = ProfilingEngine.profile_bytes(
            file_bytes,
            filename=filename,
            user_mission=user_goal,
            user_task_type=user_task_type,
        )
        profile_dict = profile.model_dump()
        profile_dict["user_task_type"] = user_task_type

        # Ensure user_task_type override is explicitly honored in target summary
        if user_task_type in ("classification", "regression"):
            if "dataset_summary" in profile_dict and "target" in profile_dict["dataset_summary"]:
                profile_dict["dataset_summary"]["target"]["task_type"] = user_task_type

        return {
            **state,
            "semantic_profile": profile_dict,
            "user_task_type": user_task_type,
            "job_status": JobStatus.PROFILING.value,
        }
    except Exception as e:
        logger.error(f"Profiling failed: {str(e)}", exc_info=True)
        return {
            **state,
            "job_status": JobStatus.FAILED.value,
            "error_message": f"Profiling failed: {str(e)}",
        }


def understanding_node(state: WorkflowStateDict) -> WorkflowStateDict:
    """Invokes Phase 7 DatasetUnderstandingAgent to build MissionBrief."""
    if state.get("job_status") == JobStatus.FAILED.value:
        return state

    try:
        profile_dict = dict(state.get("semantic_profile") or {})
        err = profile_dict.pop("error", None)
        if err:
            logger.warning(f"SemanticProfile fallback error encountered: {err}")
        profile = SemanticProfile(**profile_dict)
        user_goal = state.get("user_goal") or "Optimize machine learning model performance"

        agent = DatasetUnderstandingAgent()
        mission_brief = agent.run({
            "semantic_profile": profile,
            "user_goal": user_goal,
        })

        return {
            **state,
            "mission_brief": mission_brief.model_dump(),
            "job_status": JobStatus.PLANNING.value,
        }
    except Exception as e:
        logger.error(f"Dataset Understanding Agent failed: {str(e)}", exc_info=True)
        return {
            **state,
            "job_status": JobStatus.FAILED.value,
            "error_message": f"Dataset Understanding Agent failed: {str(e)}",
        }


def planning_node(state: WorkflowStateDict) -> WorkflowStateDict:
    """Invokes Phase 7 StrategyPlannerAgent to generate candidate ExperimentPlan."""
    if state.get("job_status") == JobStatus.FAILED.value:
        return state

    iteration = state.get("iteration_count", 0) + 1

    try:
        profile_dict = dict(state.get("semantic_profile") or {})
        err = profile_dict.pop("error", None)
        if err:
            logger.warning(f"SemanticProfile fallback error encountered: {err}")
        mission_dict = state.get("mission_brief") or {}

        profile = SemanticProfile(**profile_dict)
        mission = MissionBrief(**mission_dict)

        dataset_summary = profile_dict.get("dataset_summary", {})
        row_count = dataset_summary.get("row_count") or 1000
        target_info = dataset_summary.get("target", {})
        task_type = target_info.get("task_type") or "classification"

        # If user explicitly selected classification/regression, override auto-detection
        user_task_type = state.get("user_task_type") or profile_dict.get("user_task_type") or "general"
        if user_task_type in ("classification", "regression"):
            task_type = user_task_type

        # Dynamic experiment budget based on dataset size:
        # Small (< 5,000 rows): 6 models
        # Medium (5,000 - 50,000 rows): 5 models
        # Large (> 50,000 rows): 3 fast models
        if row_count < 5000:
            budget = 6
        elif row_count < 50000:
            budget = 5
        else:
            budget = 3

        planner = StrategyPlannerAgent()
        plan = planner.run({
            "semantic_profile": profile,
            "mission_brief": mission,
            "experiment_budget": budget,
            "task_type": task_type,
        })

        return {
            **state,
            "iteration_count": iteration,
            "experiment_plan": plan.model_dump(),
            "job_status": JobStatus.EXECUTING.value,
        }
    except Exception as e:
        logger.error(f"Strategy Planner failed: {str(e)}", exc_info=True)
        return {
            **state,
            "iteration_count": iteration,
            "job_status": JobStatus.FAILED.value,
            "error_message": f"Strategy Planner failed: {str(e)}",
        }


def execution_node(state: WorkflowStateDict) -> WorkflowStateDict:
    """Executes proposed experiments using Phase 4 ML Execution Engine."""
    if state.get("job_status") == JobStatus.FAILED.value:
        return state

    plan_dict = state.get("experiment_plan")
    file_path = state.get("file_path")
    profile_dict = state.get("semantic_profile") or {}
    dataset_summary = profile_dict.get("dataset_summary", {})
    target_info = dataset_summary.get("target", {})
    target_col = target_info.get("target_column") or "target"
    task_type = target_info.get("task_type") or "classification"

    # If user explicitly selected classification/regression, override auto-detection
    user_task_type = profile_dict.get("user_task_type", "general")
    if user_task_type in ("classification", "regression"):
        task_type = user_task_type

    if not plan_dict:
        return {
            **state,
            "job_status": JobStatus.FAILED.value,
            "error_message": "Execution failed: Missing experiment plan.",
        }

    try:
        settings = get_settings()
        file_bytes = _load_state_dataset_bytes(state)
        filename = os.path.basename(file_path or "dataset.csv")
        df, _, is_sampled = DataLoader.load_lazy_sample_from_bytes(
            file_bytes,
            filename=filename,
            max_sample_rows=settings.max_ml_sample_rows,
        )
        plan = ExperimentPlan(**plan_dict)

        job_id = state.get("job_id") or "job"
        for spec in plan.experiments:
            if not spec.experiment_id.startswith(f"{job_id}_"):
                spec.experiment_id = f"{job_id}_{spec.experiment_id}"

        cpu_count = os.cpu_count() or 4
        budget = plan.experiment_budget or len(plan.experiments) or 4
        max_workers = max(1, min(cpu_count, budget, 8))

        ml_engine = MLExecutionEngine(max_workers=max_workers)
        batch_results = ml_engine.execute_plan(
            plan=plan,
            dataset=df,
            target_column=target_col,
            task_type=task_type,
        )

        existing_results = list(state.get("experiment_results") or [])
        existing_results.extend([r.model_dump() for r in batch_results])
        return {
            **state,
            "experiment_results": existing_results,
            "job_status": JobStatus.EVALUATING.value,
        }

    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        return {
            **state,
            "job_status": JobStatus.FAILED.value,
            "error_message": f"Execution failed: {str(e)}",
        }


def evaluation_node(state: WorkflowStateDict) -> WorkflowStateDict:
    """Evaluates batch results using Phase 5 EvaluationEngine."""
    if state.get("job_status") == JobStatus.FAILED.value:
        return state

    results_dicts = state.get("experiment_results") or []
    if not results_dicts:
        return {
            **state,
            "job_status": JobStatus.FAILED.value,
            "error_message": "Evaluation failed: No experiment results found.",
        }

    try:
        results = [ExperimentResult(**r) for r in results_dicts]
        eval_engine = EvaluationEngine()
        eval_report, dec = eval_engine.evaluate_batch(
            results=results,
            job_id=state.get("job_id", "job_default"),
        )

        existing_kb = list(state.get("knowledge_base") or [])
        for f in eval_report.knowledge:
            existing_kb.append(f.model_dump())

        return {
            **state,
            "evaluation_report": eval_report.model_dump(),
            "knowledge_base": existing_kb,
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        return {
            **state,
            "job_status": JobStatus.FAILED.value,
            "error_message": f"Evaluation failed: {str(e)}",
        }


def decision_node(state: WorkflowStateDict) -> WorkflowStateDict:
    """Invokes Phase 7 ResearchDirectorAgent and budget manager to set decision."""
    if state.get("job_status") == JobStatus.FAILED.value:
        return state

    iteration = state.get("iteration_count", 1)
    max_iter = state.get("max_iterations", 5)

    eval_report_dict = state.get("evaluation_report")
    if not eval_report_dict:
        stop_decision = ResearchDirectorDecision(
            decision=DecisionType.STOP,
            confidence=1.0,
            knowledge=["No evaluation report available."],
        )
        return {
            **state,
            "decision": stop_decision.model_dump(),
        }

    try:
        eval_report = EvaluationReport(**eval_report_dict)
        director = ResearchDirectorAgent()
        decision = director.run({"evaluation_report": eval_report})

        # Override decision if budget limit reached
        if iteration >= max_iter:
            decision = ResearchDirectorDecision(
                decision=DecisionType.STOP,
                confidence=0.95,
                knowledge=decision.knowledge + [f"Budget limit reached ({max_iter} iterations)."],
            )

        return {
            **state,
            "decision": decision.model_dump(),
        }
    except Exception as e:
        logger.error(f"Research Director Agent failed: {str(e)}", exc_info=True)
        return {
            **state,
            "job_status": JobStatus.FAILED.value,
            "error_message": f"Research Director Agent failed: {str(e)}",
        }


def reporting_node(state: WorkflowStateDict) -> WorkflowStateDict:
    """Invokes Phase 7 ReportGeneratorAgent and generates HTML & Markdown reports."""
    if state.get("job_status") == JobStatus.FAILED.value:
        return state

    eval_report_dict = state.get("evaluation_report")
    if not eval_report_dict:
        return {
            **state,
            "job_status": JobStatus.FAILED.value,
            "error_message": "Reporting failed: Missing evaluation report.",
        }

    try:
        from backend.reports.html_generator import HTMLReportGenerator
        from backend.reports.markdown_generator import MarkdownReportGenerator
        from backend.schemas.semantic_profile import SemanticProfile

        eval_report = EvaluationReport(**eval_report_dict)
        reporter = ReportGeneratorAgent()
        final_rec = reporter.run({
            "evaluation_report": eval_report,
            "experiment_results": state.get("experiment_results", []),
            "semantic_profile": state.get("semantic_profile"),
        })

        job_id = state.get("job_id", "job_default")
        out_dir = os.path.join("storage", "reports", job_id)
        os.makedirs(out_dir, exist_ok=True)

        sem_prof = None
        raw_prof = state.get("semantic_profile")
        if isinstance(raw_prof, dict):
            try:
                sem_prof = SemanticProfile(**raw_prof)
            except Exception:
                sem_prof = None
        elif isinstance(raw_prof, SemanticProfile):
            sem_prof = raw_prof

        html_content = HTMLReportGenerator.generate_html(
            recommendation=final_rec,
            evaluation_report=eval_report,
            profile=sem_prof or raw_prof,
            mission_brief_str=state.get("mission_brief"),
            experiment_results=state.get("experiment_results", []),
        )

        md_content = MarkdownReportGenerator.generate_markdown(
            recommendation=final_rec,
            evaluation_report=eval_report,
            profile=sem_prof,
        )

        html_path = os.path.join(out_dir, "report.html")
        md_path = os.path.join(out_dir, "report.md")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        rec_dict = final_rec.model_dump()
        rec_dict["report_html_path"] = html_path
        rec_dict["report_md_path"] = md_path
        rec_dict["html_content"] = html_content
        rec_dict["md_content"] = md_content

        return {
            **state,
            "final_report": rec_dict,
            "job_status": JobStatus.COMPLETED.value,
        }
    except Exception as e:
        logger.error(f"Report Generator Agent failed: {str(e)}", exc_info=True)
        return {
            **state,
            "job_status": JobStatus.FAILED.value,
            "error_message": f"Report Generator Agent failed: {str(e)}",
        }
