import os
import hashlib
import uuid
import logging
from typing import Optional, Dict, Any, Tuple
from fastapi import UploadFile
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.core.exceptions import ValidationException, NotFoundException
from backend.repositories.dataset_repository import DatasetRepository
from backend.models.dataset import DatasetModel
from backend.profiling import ProfilingEngine
from backend.schemas.semantic_profile import SemanticProfile
from backend.services.storage.supabase_storage import SupabaseStorageService

logger = logging.getLogger("datapilot.services.dataset_service")

# Chunk size for reading uploaded files (64 KB) — prevents OOM on large uploads
_READ_CHUNK_SIZE = 64 * 1024


class DatasetService:
    """Service layer managing dataset uploads, cloud storage, profiling, and repository registration."""

    def __init__(self, db: Session, storage_dir: str = "storage/data"):
        self.db = db
        self.repository = DatasetRepository(db)
        self.storage_dir = storage_dir

    def upload_dataset(
        self,
        file: UploadFile,
        owner_id: str = "user_default",
        target_column: Optional[str] = None,
        mission_brief: Optional[str] = None,
        task_type: str = "general",
    ) -> DatasetModel:
        """
        Validates, streams file into memory buffer, computes checksum,
        uploads directly to Supabase Cloud Storage, and profiles in-memory.
        """
        import io

        settings = get_settings()
        max_bytes = settings.max_upload_size_mb * 1024 * 1024

        filename = file.filename or "uploaded_dataset.csv"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in [".csv", ".parquet", ".pq", ".txt"]:
            raise ValidationException(f"Unsupported file extension '{ext}'. Only CSV and Parquet are supported.")

        dataset_id = f"ds_{uuid.uuid4().hex[:8]}"

        # Stream file chunks into in-memory buffer while computing SHA-256 checksum
        hasher = hashlib.sha256()
        buffer = io.BytesIO()
        file_size = 0

        while True:
            chunk = file.file.read(_READ_CHUNK_SIZE)
            if not chunk:
                break
            file_size += len(chunk)

            if file_size > max_bytes:
                raise ValidationException(
                    f"File exceeds maximum upload size of {settings.max_upload_size_mb} MB."
                )

            hasher.update(chunk)
            buffer.write(chunk)

        checksum = hasher.hexdigest()
        file_bytes = buffer.getvalue()
        logger.info(f"Dataset '{filename}' received ({file_size} bytes, SHA256={checksum[:12]}...)")

        remote_path = f"{dataset_id}/{filename}"

        # Upload directly to Supabase Storage from memory if configured
        if settings.storage_backend.lower() == "supabase":
            try:
                storage_svc = SupabaseStorageService()
                if storage_svc.is_configured:
                    storage_svc.ensure_bucket_exists()
                    content_type = "application/octet-stream" if ext in [".parquet", ".pq"] else "text/csv"
                    storage_svc.upload_bytes(file_bytes, remote_path, content_type=content_type)
                    logger.info(f"Dataset successfully uploaded to Supabase Storage: {remote_path}")
            except Exception as se:
                logger.warning(f"Supabase Storage direct upload warning: {se}")

        # Check if checksum already exists (deduplication check)
        existing = self.repository.get_by_checksum(checksum)
        if existing:
            # Update existing mission_brief if provided
            if mission_brief and not existing.mission_brief:
                existing.mission_brief = mission_brief
                self.db.commit()

            # Ensure file is synced to Supabase if it was missing
            if settings.storage_backend.lower() == "supabase":
                try:
                    storage_svc = SupabaseStorageService()
                    if storage_svc.is_configured:
                        content_type = "application/octet-stream" if ext in [".parquet", ".pq"] else "text/csv"
                        storage_svc.upload_bytes(file_bytes, existing.file_path or remote_path, content_type=content_type)
                except Exception:
                    pass

            return existing

        # Execute ProfilingEngine directly on in-memory bytes
        rows, cols = 0, 0
        try:
            profile, _ = ProfilingEngine.profile_bytes(
                file_bytes,
                filename=filename,
                target_column=target_column,
                user_mission=mission_brief or "",
                user_task_type=task_type,
            )
            profile_dict = profile.model_dump()
            profile_dict["user_task_type"] = task_type
            profile_dict["remote_storage_path"] = remote_path
            summary = profile_dict.get("dataset_summary", {})
            rows = summary.get("rows", 0)
            cols = summary.get("columns", 0)
            target_info = summary.get("target", {}) if isinstance(summary.get("target"), dict) else {}
            profile_dict["detected_target_column"] = target_info.get("target_column")
            profile_dict["detected_task_type"] = target_info.get("task_type") or task_type
            profile_dict["row_count"] = rows
            profile_dict["column_count"] = cols
        except Exception as e:
            logger.error(f"In-memory profiling failed for dataset '{filename}': {e}", exc_info=True)
            profile_dict = {
                "dataset_summary": {"rows": 0, "columns": 0, "target": {"target_column": target_column, "task_type": task_type}},
                "user_task_type": task_type,
                "remote_storage_path": remote_path,
                "detected_target_column": target_column,
                "detected_task_type": task_type,
                "error": str(e),
            }

        # Cache local copy on disk if storage_dir is available for instant local/test access
        local_file_path = os.path.join(self.storage_dir, dataset_id, filename)
        try:
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            with open(local_file_path, "wb") as f:
                f.write(file_bytes)
        except Exception:
            pass

        # Create and persist Dataset record in database
        dataset_record = DatasetModel(
            id=dataset_id,
            owner_id=owner_id,
            filename=filename,
            file_path=local_file_path if os.path.exists(local_file_path) else remote_path,
            file_size_bytes=file_size,
            row_count=rows,
            column_count=cols,
            checksum=checksum,
            mission_brief=mission_brief,
            semantic_profile=profile_dict,
        )
        saved = self.repository.create(dataset_record)
        logger.info(f"Dataset '{filename}' registered in database (ID={dataset_id})")
        return saved

    def get_dataset(self, dataset_id: str) -> DatasetModel:
        """Retrieves dataset record by ID."""
        dataset = self.repository.get_by_id(dataset_id)
        if not dataset:
            raise NotFoundException(f"Dataset with ID '{dataset_id}' not found.")
        return dataset

    def get_dataset_bytes(self, dataset: DatasetModel) -> bytes:
        """
        Retrieves raw dataset bytes from Supabase Cloud Storage (or local disk if cached).
        """
        # 1. Try Supabase Storage
        remote_path = dataset.file_path or f"{dataset.id}/{dataset.filename}"
        try:
            storage_svc = SupabaseStorageService()
            if storage_svc.is_configured:
                return storage_svc.download_bytes(remote_path)
        except Exception as se:
            logger.warning(f"Cloud fetch for {remote_path} failed: {se}")

        # 2. Local fallback if file exists on disk
        if dataset.file_path and os.path.exists(dataset.file_path):
            with open(dataset.file_path, "rb") as f:
                return f.read()

        raise FileNotFoundError(f"Dataset '{dataset.id}' not found in Cloud Storage ({remote_path}) or local disk.")

    def ensure_local_file(self, dataset: DatasetModel) -> str:
        """
        Compatibility helper for local execution environments.
        """
        if dataset.file_path and os.path.exists(dataset.file_path):
            return dataset.file_path

        local_dest = os.path.join(self.storage_dir, dataset.id, dataset.filename)
        if os.path.exists(local_dest):
            return local_dest

        try:
            data = self.get_dataset_bytes(dataset)
            os.makedirs(os.path.dirname(local_dest), exist_ok=True)
            with open(local_dest, "wb") as f:
                f.write(data)
            return local_dest
        except Exception as e:
            logger.error(f"Failed to ensure local file for dataset {dataset.id}: {e}")
            raise FileNotFoundError(f"Dataset file not found in cloud storage: {e}")
