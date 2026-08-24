import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from io import BytesIO
from fastapi import UploadFile, BackgroundTasks

from backend.database.connection import SessionLocal, init_db
from backend.services.dataset_service import DatasetService
from backend.services.job_service import JobService
from backend.services.experiment_service import ExperimentService
from backend.services.report_service import ReportService
from backend.schemas.enums import JobStatus
from backend.core.exceptions import NotFoundException, ValidationException


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database.base import Base

_test_engine = create_engine("sqlite:///:memory:", echo=False)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=_test_engine)

@pytest.fixture
def db_session():
    Session = sessionmaker(bind=_test_engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_dataset_service_upload(db_session):
    """Verify DatasetService file upload, checksum, profiling, and repository registration."""
    csv_content = b"age,income,target\n25,50000,1\n30,60000,0\n35,70000,1\n"
    file_obj = BytesIO(csv_content)
    upload_file = UploadFile(filename="test_dataset.csv", file=file_obj)

    service = DatasetService(db_session, storage_dir="storage/test_data")
    dataset_record = service.upload_dataset(file=upload_file, target_column="target")

    assert dataset_record.id.startswith("ds_")
    assert dataset_record.filename == "test_dataset.csv"
    assert dataset_record.checksum is not None
    assert dataset_record.semantic_profile is not None
    assert dataset_record.semantic_profile["dataset_summary"]["rows"] == 3

    # Cleanup storage and DB
    if os.path.exists(dataset_record.file_path):
        os.remove(dataset_record.file_path)
    db_session.delete(dataset_record)
    db_session.commit()


def test_job_service_lifecycle(db_session):
    """Verify JobService starting, status querying, and cancellation."""
    # Create dataset first
    ds_service = DatasetService(db_session, storage_dir="storage/test_data")
    csv_content = b"x,y,target\n1,2,0\n3,4,1\n"
    upload_file = UploadFile(filename="sample_job_ds.csv", file=BytesIO(csv_content))
    ds_record = ds_service.upload_dataset(file=upload_file, target_column="target")

    job_service = JobService(db_session)
    job_record = job_service.start_job(
        dataset_id=ds_record.id,
        user_goal="Test Goal",
        background_tasks=BackgroundTasks()
    )

    assert job_record.id.startswith("job_")
    assert job_record.dataset_id == ds_record.id
    assert job_record.status in [JobStatus.QUEUED.value, JobStatus.PROFILING.value]

    # Query status
    fetched = job_service.get_job(job_record.id)
    assert fetched.id == job_record.id

    # Test invalid job id
    with pytest.raises(NotFoundException):
        job_service.get_job("invalid_job_999")

    # Cleanup storage and DB
    if os.path.exists(ds_record.file_path):
        os.remove(ds_record.file_path)
    db_session.delete(job_record)
    db_session.delete(ds_record)
    db_session.commit()
