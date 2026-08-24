import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.schemas.enums import JobStatus
from backend.database.connection import SessionLocal
from backend.repositories.job_repository import JobRepository
from backend.repositories.experiment_repository import ExperimentRepository
from backend.repositories.report_repository import ReportRepository
from backend.repositories.knowledge_repository import KnowledgeRepository
from backend.models.job import JobModel
from backend.models.experiment import ExperimentModel
from backend.models.report import ReportModel
from backend.models.knowledge import KnowledgeEntryModel
from backend.graph import compile_graph, create_initial_state
from backend.api.websocket_manager import ws_manager
from backend.core.config import get_settings
from backend.services.storage.supabase_storage import SupabaseStorageService

logger = logging.getLogger("datapilot.services.job_manager")


def _now_iso() -> str:
    """Returns the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    """
    Manager for dispatching long-running research jobs asynchronously,
    updating database records, and broadcasting WebSocket progress events.
    """

    @classmethod
    async def _broadcast(cls, job_id: str, payload: Dict[str, Any]):
        """
        Broadcasts a WebSocket event structured cleanly according to WSEvent schema:
        { "event": ..., "job_id": ..., "timestamp": ..., "data": { ... } }
        """
        event_name = payload.get("event", "log.message")
        timestamp = payload.get("timestamp") or _now_iso()

        if "data" in payload and isinstance(payload["data"], dict):
            data_content = payload["data"]
        else:
            data_content = {
                k: v for k, v in payload.items()
                if k not in ("event", "job_id", "timestamp")
            }

        formatted_payload = {
            "event": event_name,
            "job_id": job_id,
            "timestamp": timestamp,
            "data": data_content,
        }
        await ws_manager.broadcast_to_job(job_id, formatted_payload)

    @classmethod
    async def _broadcast_log(cls, job_id: str, message: str, stage: Optional[str] = None,
                             level: str = "info"):
        """Broadcasts a log.message event for the live execution console terminal."""
        await cls._broadcast(job_id, {
            "event": "log.message",
            "message": message,
            "level": level,
            "stage": stage,
            "progress_percent": None,
        })

    @classmethod
    async def run_job_async(
        cls,
        job_id: str,
        dataset_id: str,
        file_path: str,
        user_goal: Optional[str] = None,
        task_type: str = "general",
    ):
        """Asynchronous worker executing the compiled LangGraph state machine."""
        db: Session = SessionLocal()
        job_repo = JobRepository(db)
        exp_repo = ExperimentRepository(db)
        report_repo = ReportRepository(db)
        knowledge_repo = KnowledgeRepository(db)

        try:
            # 1. Update status to profiling & broadcast
            job_repo.update_status(job_id, JobStatus.PROFILING, progress_pct=10.0)
            await cls._broadcast(job_id, {
                "event": "job.status_changed",
                "status": JobStatus.PROFILING.value,
                "stage": "profiling",
                "progress_percent": 10.0,
            })
            await cls._broadcast_log(job_id, "Starting dataset profiling...", stage="profiling")

            # 2. Compile LangGraph workflow
            app = compile_graph()
            initial_state = create_initial_state(
                dataset_id=dataset_id,
                job_id=job_id,
                file_path=file_path,
                user_goal=user_goal,
                user_task_type=task_type,
                max_iterations=3,
            )

            config = {"configurable": {"thread_id": f"thread_{job_id}"}}

            await cls._broadcast_log(job_id, "Compiled research graph. Streaming execution stages...",
                                     stage="profiling", level="info")

            stage_configs = {
                "profiling": {"stage": "profiling", "progress": 20.0, "status": JobStatus.PROFILING, "msg": "Completed dataset profiling."},
                "understanding": {"stage": "understanding", "progress": 30.0, "status": JobStatus.PLANNING, "msg": "Semantic dataset understanding completed."},
                "planning": {"stage": "planning", "progress": 45.0, "status": JobStatus.PLANNING, "msg": "Formulated prioritized experiment plan."},
                "execution": {"stage": "executing", "progress": 75.0, "status": JobStatus.EXECUTING, "msg": "Executed ML pipeline experiments."},
                "evaluation": {"stage": "evaluating", "progress": 85.0, "status": JobStatus.EVALUATING, "msg": "Evaluated & ranked experiment results."},
                "directing": {"stage": "decision", "progress": 90.0, "status": JobStatus.EVALUATING, "msg": "Research Director decision rendered."},
                "reporting": {"stage": "reporting", "progress": 95.0, "status": JobStatus.COMPLETED, "msg": "Synthesized final recommendation report."},
            }

            accumulated_state: Dict[str, Any] = dict(initial_state)

            async for chunk in app.astream(initial_state, config):
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, dict):
                        accumulated_state.update(node_output)

                    # Check for node failure
                    node_status = node_output.get("job_status") if isinstance(node_output, dict) else None
                    err = node_output.get("error_message") if isinstance(node_output, dict) else None

                    if node_status == JobStatus.FAILED.value or err:
                        error_msg = err or f"Stage '{node_name}' failed."
                        job_repo.update_status(job_id, JobStatus.FAILED, error_message=error_msg)
                        await cls._broadcast(job_id, {
                            "event": "job.failed",
                            "status": "failed",
                            "message": error_msg,
                            "level": "error",
                            "error": error_msg,
                        })
                        await cls._broadcast_log(job_id, f"Job failed at stage '{node_name}': {error_msg}", level="error")
                        return

                    # Update status & broadcast stage transition
                    stg_cfg = stage_configs.get(node_name, {"stage": node_name, "progress": 50.0, "status": JobStatus.EXECUTING, "msg": f"Stage {node_name} finished."})
                    job_repo.update_status(job_id, stg_cfg["status"], progress_pct=stg_cfg["progress"])

                    await cls._broadcast(job_id, {
                        "event": "job.status_changed",
                        "status": stg_cfg["status"].value if hasattr(stg_cfg["status"], "value") else str(stg_cfg["status"]),
                        "stage": stg_cfg["stage"],
                        "progress_percent": stg_cfg["progress"],
                    })
                    await cls._broadcast_log(job_id, stg_cfg["msg"], stage=stg_cfg["stage"], level="info")

                    # Smoothly transition stage indicator for the next node about to execute
                    if node_name == "understanding":
                        await cls._broadcast(job_id, {"event": "job.status_changed", "status": JobStatus.PLANNING.value, "stage": "planning", "progress_percent": 35.0})
                        await cls._broadcast_log(job_id, "Starting research planning agent...", stage="planning", level="info")
                    elif node_name == "planning":
                        await cls._broadcast(job_id, {"event": "job.status_changed", "status": JobStatus.EXECUTING.value, "stage": "executing", "progress_percent": 50.0})
                        await cls._broadcast_log(job_id, "Executing ML pipeline experiments across model configurations...", stage="executing", level="info")
                    elif node_name == "execution":
                        await cls._broadcast(job_id, {"event": "job.status_changed", "status": JobStatus.EVALUATING.value, "stage": "evaluating", "progress_percent": 80.0})
                        await cls._broadcast_log(job_id, "Evaluating & ranking experiment results...", stage="evaluating", level="info")

                    await ws_manager.ping_heartbeat(job_id)

            final_state = accumulated_state

            # 3. Persist executed experiments to database
            import uuid
            exp_results = final_state.get("experiment_results") or []
            await cls._broadcast_log(job_id, f"Persisting {len(exp_results)} experiment results...",
                                     stage="evaluating")
            for idx, exp_dict in enumerate(exp_results):
                raw_status = exp_dict.get("status", "completed")
                err_msg = exp_dict.get("error_message")
                exp_code = exp_dict.get("experiment_id") or f"exp_{job_id}_{idx}_{uuid.uuid4().hex[:6]}"
                db_id = f"exp_db_{job_id}_{idx}_{uuid.uuid4().hex[:6]}"

                final_exp_status = "failed" if (raw_status == "failed" or err_msg) else "completed"

                try:
                    existing_exp = exp_repo.get_by_code(job_id, exp_code)
                    if not existing_exp:
                        metrics_payload = dict(exp_dict.get("metrics", {}) or {})
                        if err_msg:
                            metrics_payload["error"] = err_msg

                        hyperparams = (
                            exp_dict.get("hyperparameters")
                            or exp_dict.get("params")
                            or (exp_dict.get("pipeline", {}).get("params") if isinstance(exp_dict.get("pipeline"), dict) else {})
                            or {}
                        )

                        exp_repo.create(
                            ExperimentModel(
                                id=db_id,
                                job_id=job_id,
                                experiment_id_code=exp_code,
                                pipeline=exp_dict.get("pipeline", {}),
                                model_name=exp_dict.get("model", "unknown"),
                                hyperparameters=hyperparams,
                                metrics=metrics_payload,
                                runtime_seconds=exp_dict.get("runtime"),
                                status=final_exp_status,
                                artifact_paths=exp_dict.get("artifacts", {}),
                            )
                        )

                    event_type = "experiment.failed" if final_exp_status == "failed" else "experiment.completed"
                    event_level = "error" if final_exp_status == "failed" else "success"
                    event_msg = f"Experiment {exp_code} failed: {err_msg}" if final_exp_status == "failed" else f"Experiment {exp_code} completed ({exp_dict.get('model', 'unknown')})"

                    await cls._broadcast(job_id, {
                        "event": event_type,
                        "experiment_id": exp_code,
                        "message": event_msg,
                        "stage": "executing",
                        "level": event_level,
                        "status": final_exp_status,
                        "error": err_msg,
                    })
                except Exception as exp_err:
                    logger.error(f"Error persisting experiment {exp_code}: {exp_err}", exc_info=True)
                    try:
                        db.rollback()
                    except Exception:
                        pass

            # 4. Persist knowledge base findings to database
            kb_findings = final_state.get("knowledge_base") or []
            for k_idx, k_dict in enumerate(kb_findings):
                knowledge_repo.create(
                    KnowledgeEntryModel(
                        job_id=job_id,
                        finding=k_dict.get("finding", ""),
                        confidence=k_dict.get("confidence", 0.9),
                        source_experiment_ids=k_dict.get("source_experiment_ids", []),
                    )
                )


            if kb_findings:
                await cls._broadcast(job_id, {
                    "event": "knowledge.updated",
                    "message": f"Discovered {len(kb_findings)} knowledge findings",
                    "stage": "evaluating",
                    "level": "info",
                })

            # 5. Persist final report to database
            final_report_dict = final_state.get("final_report") or {}
            eval_report_dict = final_state.get("evaluation_report") or {}
            winning_id = eval_report_dict.get("winner") or final_report_dict.get("winning_experiment_id") or "exp_1"
            html_report_path = final_report_dict.get("report_html_path") or f"storage/reports/{job_id}/report.html"
            
            existing_report = report_repo.get_by_job(job_id)
            if not existing_report:
                report_repo.create(
                    ReportModel(
                        id=f"rep_{job_id}",
                        job_id=job_id,
                        winning_experiment_id=winning_id,
                        report_file_path=html_report_path,
                        summary=final_report_dict.get("summary") or "Final research report completed.",
                    )
                )
            else:
                existing_report.winning_experiment_id = winning_id
                existing_report.report_file_path = html_report_path
                existing_report.summary = final_report_dict.get("summary") or existing_report.summary
                db.commit()
            # Upload generated HTML report to Supabase Cloud Storage
            settings = get_settings()
            if settings.storage_backend.lower() == "supabase":
                try:
                    storage_svc = SupabaseStorageService()
                    if storage_svc.is_configured and os.path.exists(html_report_path):
                        storage_svc.ensure_bucket_exists()
                        storage_svc.upload_file(html_report_path, f"reports/{job_id}/report.html")
                        logger.info(f"Report uploaded to Supabase Storage: reports/{job_id}/report.html")
                except Exception as se:
                    logger.warning(f"Could not upload report to Supabase Storage: {se}")

            await cls._broadcast_log(job_id, "Final report generated successfully.",
                                     stage="reporting", level="success")

            # 6. Update job status to completed & broadcast
            job_repo.update_status(job_id, JobStatus.COMPLETED, progress_pct=100.0)
            await cls._broadcast(job_id, {
                "event": "job.completed",
                "status": "completed",
                "stage": "reporting",
                "progress_percent": 100.0,
                "winning_experiment_id": winning_id,
                "message": "Research job completed successfully!",
                "level": "success",
            })

            logger.info(f"Research job {job_id} completed successfully!")

        except Exception as e:
            logger.error(f"Error executing background research job {job_id}: {e}", exc_info=True)
            # Rollback any uncommitted transaction to prevent SQLite lock
            try:
                db.rollback()
            except Exception:
                pass
            job_repo.update_status(job_id, JobStatus.FAILED, error_message=str(e))
            await cls._broadcast(job_id, {
                "event": "job.failed",
                "status": "failed",
                "error": str(e),
                "message": f"Job failed: {str(e)}",
                "level": "error",
            })
        finally:
            db.close()

