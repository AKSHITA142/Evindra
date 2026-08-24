import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.core.config import get_settings
from backend.database.base import Base
import backend.models.user
import backend.models.dataset
import backend.models.job
import backend.models.experiment
import backend.models.report
import backend.models.knowledge

from backend.models.user import UserModel
from backend.services.storage.supabase_storage import SupabaseStorageService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("datapilot.reset")


def reset_cloud_data():
    settings = get_settings()
    target_url = settings.database_url

    if not target_url or target_url.startswith("sqlite"):
        logger.warning("Target DATABASE_URL is not PostgreSQL. Resetting local database...")
        engine = create_engine(target_url or "sqlite:///./datapilot.db", echo=False)
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Local database reset to 0.")
        return

    logger.info(f"Target Cloud Database: {target_url.split('@')[-1] if '@' in target_url else target_url}")

    cloud_engine = create_engine(
        target_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    try:
        with cloud_engine.connect() as conn:
            logger.info("Clearing all records from Supabase PostgreSQL tables...")
            # Truncate all tables in foreign key order with CASCADE
            conn.execute(text("TRUNCATE TABLE reports, knowledge_entries, experiments, jobs, datasets, users CASCADE;"))
            conn.commit()
            logger.info("All PostgreSQL tables truncated successfully.")

        # Re-seed default user for session isolation
        CloudSession = sessionmaker(bind=cloud_engine)
        db = CloudSession()
        default_user = UserModel(
            id="user_default",
            email="guest@datapilot.ai",
            full_name="Guest User",
            is_active=True
        )
        db.add(default_user)
        db.commit()
        db.close()
        logger.info("Default guest user seeded (user_default).")

    except Exception as e:
        logger.error(f"Error resetting database tables: {e}")
        return

    # Clear Supabase Storage bucket files if storage is configured
    try:
        storage_svc = SupabaseStorageService()
        if storage_svc.is_configured:
            bucket_name = storage_svc.bucket_name
            client = storage_svc._client
            if client:
                files = client.storage.from_(bucket_name).list()
                if files:
                    file_paths = []
                    for f in files:
                        name = f.get("name")
                        if name:
                            # If it's a folder, list inner files
                            inner_files = client.storage.from_(bucket_name).list(name)
                            if inner_files:
                                for inf in inner_files:
                                    iname = inf.get("name")
                                    if iname:
                                        file_paths.append(f"{name}/{iname}")
                            file_paths.append(name)

                    if file_paths:
                        client.storage.from_(bucket_name).remove(file_paths)
                        logger.info(f"Cleared {len(file_paths)} old test files from Supabase bucket '{bucket_name}'.")
    except Exception as se:
        logger.warning(f"Note on storage clean: {se}")

    logger.info("✨ Clean slate complete! All cloud data removed. The system is ready from 0.")


if __name__ == "__main__":
    reset_cloud_data()
