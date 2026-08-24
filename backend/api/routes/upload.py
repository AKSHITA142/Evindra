from typing import Optional, List
from fastapi import APIRouter, Depends, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.services.dataset_service import DatasetService
from backend.repositories.dataset_repository import DatasetRepository
from backend.schemas.response import SuccessResponse

router = APIRouter(prefix="/upload", tags=["Dataset Upload"])


def _dataset_to_frontend(ds) -> dict:
    """Converts a DatasetModel row into the dictionary shape the frontend Dataset type expects."""
    return {
        "dataset_id": ds.id,
        "filename": ds.filename,
        "file_size_bytes": ds.file_size_bytes or 0,
        "row_count": ds.row_count or 0,
        "column_count": ds.column_count or 0,
        "upload_timestamp": ds.created_at.isoformat() if ds.created_at else None,
        "status": "profiled" if ds.semantic_profile else "uploaded",
        "mission_brief": getattr(ds, "mission_brief", None),
        "profile": ds.semantic_profile,
        # Keep original fields
        "file_path": ds.file_path,
        "checksum": ds.checksum,
    }


@router.post("", response_model=SuccessResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    mission: Optional[str] = Form(None),
    task_type: Optional[str] = Form("general"),
    db: Session = Depends(get_db),
):
    """
    Uploads a raw CSV/Parquet dataset file, runs automated profiling, and registers dataset.
    task_type: One of 'classification', 'regression', or 'general' (auto-detect).
    """
    service = DatasetService(db)
    dataset_record = service.upload_dataset(
        file=file, target_column=target_column,
        mission_brief=mission, task_type=task_type or "general",
    )

    return SuccessResponse(
        data=_dataset_to_frontend(dataset_record),
        message="Dataset uploaded and profiled successfully.",
    )


@router.get("", response_model=SuccessResponse)
def list_all_datasets(db: Session = Depends(get_db)):
    """
    Lists all uploaded datasets (GET /api/v1/upload).
    Frontend also calls this via GET /api/v1/datasets (registered separately).
    """
    repo = DatasetRepository(db)
    datasets = repo.list(skip=0, limit=200)
    return SuccessResponse(
        data=[_dataset_to_frontend(ds) for ds in datasets],
        message=f"Retrieved {len(datasets)} datasets.",
    )


@router.get("/{dataset_id}", response_model=SuccessResponse)
def get_dataset_details(dataset_id: str, db: Session = Depends(get_db)):
    """
    Retrieves metadata and semantic profile for an uploaded dataset.
    """
    service = DatasetService(db)
    dataset_record = service.get_dataset(dataset_id)

    return SuccessResponse(
        data=_dataset_to_frontend(dataset_record),
        message="Dataset details retrieved successfully.",
    )

