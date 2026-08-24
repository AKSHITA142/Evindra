import os
import pickle
from typing import Dict, Any, Optional
import pandas as pd

from backend.core.config import get_settings


class ArtifactExporter:
    """Exports dataset CSVs, trained model pickles, and report files to Supabase Storage and disk."""

    @classmethod
    def export_cleaned_dataset(cls, df: pd.DataFrame, job_id: str, storage_dir: Optional[str] = None) -> str:
        """Uploads processed cleaned dataset CSV to Supabase Storage and saves local fallback."""
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        remote_path = f"reports/{job_id}/cleaned_{job_id}.csv"

        try:
            from backend.services.storage.supabase_storage import SupabaseStorageService
            storage_svc = SupabaseStorageService()
            if storage_svc.is_configured:
                storage_svc.upload_bytes(csv_bytes, remote_path, content_type="text/csv")
        except Exception as se:
            pass

        base_dir = storage_dir or get_settings().storage_dir
        datasets_dir = os.path.join(base_dir, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        path = os.path.join(datasets_dir, f"cleaned_{job_id}.csv")
        with open(path, "wb") as f:
            f.write(csv_bytes)
        return path

    @classmethod
    def export_model_artifact(cls, fitted_pipeline: Any, job_id: str, storage_dir: Optional[str] = None) -> str:
        """Serializes trained scikit-learn pipeline to Supabase Storage and local fallback."""
        model_bytes = pickle.dumps(fitted_pipeline)
        remote_path = f"reports/{job_id}/model_{job_id}.pkl"

        try:
            from backend.services.storage.supabase_storage import SupabaseStorageService
            storage_svc = SupabaseStorageService()
            if storage_svc.is_configured:
                storage_svc.upload_bytes(model_bytes, remote_path, content_type="application/octet-stream")
        except Exception:
            pass

        base_dir = storage_dir or get_settings().storage_dir
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, f"model_{job_id}.pkl")
        with open(path, "wb") as f:
            f.write(model_bytes)
        return path

    @classmethod
    def export_report_files(
        cls,
        html_content: str,
        md_content: str,
        job_id: str,
        storage_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Saves HTML and Markdown report files to Supabase Storage and local fallback."""
        html_bytes = html_content.encode("utf-8")
        md_bytes = md_content.encode("utf-8")

        try:
            from backend.services.storage.supabase_storage import SupabaseStorageService
            storage_svc = SupabaseStorageService()
            if storage_svc.is_configured:
                storage_svc.upload_bytes(html_bytes, f"reports/{job_id}/report.html", content_type="text/html")
                storage_svc.upload_bytes(md_bytes, f"reports/{job_id}/report.md", content_type="text/markdown")
        except Exception:
            pass

        base_dir = storage_dir or get_settings().storage_dir
        reports_dir = os.path.join(base_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        html_path = os.path.join(reports_dir, f"report_{job_id}.html")
        with open(html_path, "wb") as f:
            f.write(html_bytes)

        md_path = os.path.join(reports_dir, f"report_{job_id}.md")
        with open(md_path, "wb") as f:
            f.write(md_bytes)

        return {
            "html": html_path,
            "markdown": md_path,
        }
