import os
import logging
from typing import Optional, Dict, Any, Union

try:
    from supabase import create_client, Client  # type: ignore # pyrefly: ignore
except ImportError:
    create_client = None
    Client = Any

from backend.core.config import get_settings

logger = logging.getLogger("datapilot.services.supabase_storage")


class SupabaseStorageService:
    """
    Manages cloud dataset uploads, downloads, and signed URLs using Supabase Storage.
    Provides graceful fallbacks and error handling.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.url = url if url is not None else settings.supabase_url
        self.key = key if key is not None else settings.supabase_key
        self.bucket_name = bucket_name or settings.supabase_bucket or "datasets"
        self._client: Optional[Client] = None

        if self.url:
            self.url = self.url.strip().rstrip("/")
            if self.url.endswith("/rest/v1"):
                self.url = self.url[:-8].rstrip("/")

        if self.url and self.key:
            try:
                self._client = create_client(self.url, self.key)
                logger.info(f"Supabase Storage initialized ({self.url}) for bucket: {self.bucket_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase client: {e}")

    @property
    def is_configured(self) -> bool:
        """Returns True if valid Supabase client is initialized."""
        return self._client is not None

    def ensure_bucket_exists(self) -> bool:
        """Checks if bucket exists or creates it if permissions allow."""
        if not self.is_configured:
            return False
        try:
            buckets = self._client.storage.list_buckets()
            existing = [b.name if hasattr(b, "name") else b.get("name") for b in buckets]
            if self.bucket_name not in existing:
                try:
                    self._client.storage.create_bucket(self.bucket_name, options={"public": True})
                    logger.info(f"Created Supabase storage bucket: {self.bucket_name}")
                except Exception as ce:
                    logger.warning(f"Could not auto-create bucket '{self.bucket_name}': {ce}")
            return True
        except Exception as e:
            logger.warning(f"Bucket check failed: {e}")
            return False

    def upload_bytes(
        self,
        file_bytes: bytes,
        remote_path: str,
        content_type: str = "text/csv",
    ) -> Dict[str, Any]:
        """
        Uploads in-memory bytes directly to Supabase Storage with auto-compression for large CSVs.
        """
        if not self.is_configured:
            raise RuntimeError("Supabase Storage is not configured. Set SUPABASE_URL and SUPABASE_KEY.")

        try:
            target_bytes = file_bytes
            target_content_type = content_type
            file_size = len(file_bytes)

            # If CSV and file size > 40MB, compress with dtype optimization + zstd parquet before uploading
            if file_size > 40 * 1024 * 1024 and not file_bytes.startswith(b"PAR1"):
                try:
                    import io
                    import pandas as pd
                    from backend.profiling.loader import DataLoader

                    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
                    df = DataLoader.optimize_dtypes(df)
                    buf = io.BytesIO()
                    df.to_parquet(buf, compression="zstd", index=False)
                    buf.seek(0)
                    compressed_bytes = buf.getvalue()
                    if len(compressed_bytes) < file_size:
                        target_bytes = compressed_bytes
                        target_content_type = "application/octet-stream"
                        logger.info(
                            f"Compressed in-memory CSV ({file_size / (1024*1024):.1f}MB) to ZSTD Parquet "
                            f"({len(compressed_bytes) / (1024*1024):.1f}MB) for cloud upload."
                        )
                except Exception as ce:
                    logger.warning(f"Could not pre-compress in-memory CSV to parquet: {ce}")

            file_options = {"content-type": target_content_type, "upsert": "true"}
            response = self._client.storage.from_(self.bucket_name).upload(
                path=remote_path,
                file=target_bytes,
                file_options=file_options,
            )
            logger.info(f"Uploaded {len(target_bytes)} bytes to Supabase: {self.bucket_name}/{remote_path}")
            return {"status": "success", "remote_path": remote_path, "response": response}
        except Exception as e:
            logger.error(f"Failed to upload bytes to Supabase {remote_path}: {e}")
            raise

    def upload_file(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        """
        Uploads a local file to the Supabase Storage bucket by streaming into upload_bytes.
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        content_type = "application/octet-stream" if local_path.endswith((".parquet", ".pq")) else "text/csv"
        with open(local_path, "rb") as f:
            data = f.read()
        return self.upload_bytes(data, remote_path, content_type=content_type)

    def download_bytes(self, remote_path: str) -> bytes:
        """
        Downloads a remote file from Supabase Storage directly into an in-memory bytes object.
        """
        if not self.is_configured:
            raise RuntimeError("Supabase Storage is not configured.")

        try:
            data = self._client.storage.from_(self.bucket_name).download(remote_path)
            logger.info(f"Downloaded Supabase file {self.bucket_name}/{remote_path} ({len(data)} bytes) to memory.")
            return data
        except Exception as e:
            logger.error(f"Failed to download {remote_path} from Supabase: {e}")
            raise

    def download_stream(self, remote_path: str):
        """
        Downloads a remote file from Supabase Storage and returns an io.BytesIO stream.
        """
        import io
        data = self.download_bytes(remote_path)
        return io.BytesIO(data)

    def download_file(self, remote_path: str, local_dest_path: str) -> str:
        """
        Downloads a remote file from Supabase Storage and saves it to local_dest_path.
        """
        data = self.download_bytes(remote_path)
        os.makedirs(os.path.dirname(local_dest_path), exist_ok=True)
        with open(local_dest_path, "wb") as f:
            f.write(data)
        logger.info(f"Saved Supabase file {self.bucket_name}/{remote_path} to {local_dest_path}")
        return local_dest_path

    def get_public_url(self, remote_path: str) -> str:
        """Returns public URL for a file in a public bucket."""
        if not self.is_configured:
            return ""
        try:
            return self._client.storage.from_(self.bucket_name).get_public_url(remote_path)
        except Exception as e:
            logger.warning(f"Failed to get public URL for {remote_path}: {e}")
            return ""

    def get_signed_url(self, remote_path: str, expires_in: int = 3600) -> Optional[str]:
        """Generates a temporary signed URL for private bucket access."""
        if not self.is_configured:
            return None
        try:
            res = self._client.storage.from_(self.bucket_name).create_signed_url(remote_path, expires_in)
            if isinstance(res, dict):
                return res.get("signedURL") or res.get("signed_url")
            return getattr(res, "signed_url", None) or str(res)
        except Exception as e:
            logger.warning(f"Failed to create signed URL for {remote_path}: {e}")
            return None
