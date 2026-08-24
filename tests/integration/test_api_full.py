import os
import tempfile
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from backend.main import app
from backend.database.connection import init_db, SessionLocal
from backend.schemas.enums import JobStatus
from backend.services.job_manager import JobManager

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def test_full_api_workflow_upload_to_report():
    """
    Full end-to-end integration test verifying:
    POST /api/v1/upload -> POST /api/v1/jobs/start -> Background Worker -> GET /api/v1/jobs/{id} -> GET /api/v1/experiments/{id} -> GET /api/v1/reports/{id}
    """
    # 1. Prepare synthetic CSV file
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "income": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        # 2. Upload dataset via POST /api/v1/upload
        with open(tmp_path, "rb") as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("test_full.csv", f, "text/csv")},
                data={"target_column": "target"},
            )

        assert response.status_code == 201
        res_data = response.json()
        assert "data" in res_data
        dataset_id = res_data["data"]["dataset_id"]
        assert dataset_id.startswith("ds_")

        # 3. Start research job via POST /api/v1/jobs/start
        start_resp = client.post(
            "/api/v1/jobs/start",
            json={"dataset_id": dataset_id, "user_goal": "Optimize churn classification"},
        )

        assert start_resp.status_code == 202
        start_data = start_resp.json()
        assert "data" in start_data
        job_id = start_data["data"]["job_id"]
        assert job_id.startswith("job_")

        # 4. Execute background job worker synchronously for test verification
        import asyncio
        asyncio.run(
            JobManager.run_job_async(
                job_id=job_id,
                dataset_id=dataset_id,
                file_path=res_data["data"]["file_path"],
                user_goal="Optimize churn classification",
            )
        )

        # 5. Verify GET /api/v1/jobs/{job_id} status is completed
        status_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert status_resp.status_code == 200
        job_info = status_resp.json()["data"]
        assert job_info["status"] == JobStatus.COMPLETED.value
        assert job_info["progress_pct"] == 100.0

        # 6. Verify GET /api/v1/experiments/{job_id} returns executed experiments
        exp_resp = client.get(f"/api/v1/experiments/{job_id}")
        assert exp_resp.status_code == 200
        experiments = exp_resp.json()["data"]
        assert len(experiments) >= 2
        assert experiments[0]["experiment_id"] is not None

        # 7. Verify GET /api/v1/reports/{job_id} returns final recommendation
        rep_resp = client.get(f"/api/v1/reports/{job_id}")
        assert rep_resp.status_code == 200
        report_info = rep_resp.json()["data"]
        assert report_info["job_id"] == job_id
        assert report_info["winning_experiment_id"] is not None
        assert report_info["summary"] is not None

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        # Clean up integration test entities from DB
        try:
            db = SessionLocal()
            from backend.models.dataset import DatasetModel
            from backend.models.job import JobModel
            from backend.models.experiment import ExperimentModel
            from backend.models.report import ReportModel
            from backend.models.knowledge import KnowledgeEntryModel

            if "job_id" in locals():
                db.query(ReportModel).filter(ReportModel.job_id == job_id).delete()
                db.query(KnowledgeEntryModel).filter(KnowledgeEntryModel.job_id == job_id).delete()
                db.query(ExperimentModel).filter(ExperimentModel.job_id == job_id).delete()
                db.query(JobModel).filter(JobModel.id == job_id).delete()
            if "dataset_id" in locals():
                db.query(DatasetModel).filter(DatasetModel.id == dataset_id).delete()
            db.commit()
            db.close()
        except Exception:
            pass
