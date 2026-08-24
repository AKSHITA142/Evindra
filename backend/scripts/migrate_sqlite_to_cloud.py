import os
import sys
import logging
from sqlalchemy import create_engine, select
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

from backend.models.user import UserModel
from backend.models.dataset import DatasetModel
from backend.models.job import JobModel
from backend.models.experiment import ExperimentModel
from backend.models.report import ReportModel
from backend.services.storage.supabase_storage import SupabaseStorageService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("datapilot.migration")


def migrate_data():
    settings = get_settings()
    target_url = settings.database_url

    if not target_url or target_url.startswith("sqlite"):
        logger.warning(f"Target DATABASE_URL is SQLite ({target_url}). To migrate to cloud, configure a PostgreSQL URL in .env.")
        logger.info("Initializing schema on target database...")
        engine_local = create_engine(target_url or "sqlite:///./datapilot.db", echo=False)
        Base.metadata.create_all(bind=engine_local)
        logger.info("Schema initialization complete.")
        return

    logger.info(f"Target Cloud Database: {target_url.split('@')[-1] if '@' in target_url else target_url}")

    # 1. Initialize schema on Cloud PostgreSQL
    cloud_engine = create_engine(
        target_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    try:
        logger.info("Creating tables on Supabase PostgreSQL...")
        Base.metadata.create_all(bind=cloud_engine)
        logger.info("Tables created successfully on Supabase PostgreSQL.")
    except Exception as e:
        logger.error(f"Failed to connect or create schema on Supabase PostgreSQL: {e}")
        logger.error("Please verify your DATABASE_URL password in .env and check database connectivity.")
        return

    CloudSession = sessionmaker(bind=cloud_engine)
    cloud_db = CloudSession()

    # 2. Check if local SQLite exists to migrate records
    sqlite_path = "datapilot.db"
    if os.path.exists(sqlite_path):
        logger.info(f"Found local SQLite database: {sqlite_path}. Starting data transfer...")
        sqlite_engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)
        SqliteSession = sessionmaker(bind=sqlite_engine)
        sqlite_db = SqliteSession()

        try:
            from sqlalchemy.orm import make_transient

            # Ensure default user exists for foreign key integrity
            if not cloud_db.query(UserModel).filter_by(id="user_default").first():
                cloud_db.add(UserModel(id="user_default", email="default@datapilot.ai", full_name="Default User", is_active=True))
                cloud_db.commit()

            # Migrate Users
            users = sqlite_db.query(UserModel).all()
            for u in users:
                if not cloud_db.query(UserModel).filter_by(id=u.id).first():
                    sqlite_db.expunge(u)
                    make_transient(u)
                    cloud_db.add(u)
            cloud_db.commit()
            logger.info(f"Migrated {len(users)} users.")

            # Migrate Datasets
            datasets = sqlite_db.query(DatasetModel).all()
            for d in datasets:
                if not cloud_db.query(DatasetModel).filter_by(id=d.id).first():
                    sqlite_db.expunge(d)
                    make_transient(d)
                    cloud_db.add(d)
            cloud_db.commit()
            logger.info(f"Migrated {len(datasets)} datasets.")

            # Migrate Jobs
            valid_dataset_ids = set(r[0] for r in cloud_db.query(DatasetModel.id).all())
            jobs = sqlite_db.query(JobModel).all()
            migrated_jobs = 0
            for j in jobs:
                if j.dataset_id in valid_dataset_ids:
                    if not cloud_db.query(JobModel).filter_by(id=j.id).first():
                        sqlite_db.expunge(j)
                        make_transient(j)
                        cloud_db.add(j)
                        migrated_jobs += 1
            cloud_db.commit()
            logger.info(f"Migrated {migrated_jobs} research jobs.")

            # Helper for NaN sanitization
            def clean_nan_obj(obj):
                import math
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                elif isinstance(obj, dict):
                    return {k: clean_nan_obj(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nan_obj(v) for v in obj]
                return obj

            # Migrate Experiments
            valid_job_ids = set(r[0] for r in cloud_db.query(JobModel.id).all())
            experiments = sqlite_db.query(ExperimentModel).all()
            migrated_exps = 0
            for exp in experiments:
                if exp.job_id in valid_job_ids:
                    if not cloud_db.query(ExperimentModel).filter_by(id=exp.id).first():
                        sqlite_db.expunge(exp)
                        make_transient(exp)
                        exp.metrics = clean_nan_obj(exp.metrics)
                        exp.pipeline = clean_nan_obj(exp.pipeline)
                        exp.hyperparameters = clean_nan_obj(exp.hyperparameters)
                        cloud_db.add(exp)
                        migrated_exps += 1
            cloud_db.commit()
            logger.info(f"Migrated {migrated_exps} experiments.")

            # Migrate Reports
            reports = sqlite_db.query(ReportModel).all()
            migrated_reports = 0
            for r in reports:
                if r.job_id in valid_job_ids:
                    if not cloud_db.query(ReportModel).filter_by(id=r.id).first():
                        sqlite_db.expunge(r)
                        make_transient(r)
                        r.summary = clean_nan_obj(r.summary)
                        cloud_db.add(r)
                        migrated_reports += 1
            cloud_db.commit()
            logger.info(f"Migrated {migrated_reports} reports.")

        except Exception as me:
            logger.warning(f"Error during SQLite record migration: {me}")
            cloud_db.rollback()
        finally:
            sqlite_db.close()
    else:
        logger.info("No local datapilot.db found. Fresh cloud database initialized.")

    cloud_db.close()

    # 3. Sync local dataset files in storage/data to Supabase Storage
    try:
        storage_svc = SupabaseStorageService()
        if storage_svc.is_configured:
            storage_svc.ensure_bucket_exists()
            data_dir = "storage/data"
            if os.path.exists(data_dir):
                count = 0
                for root, _, files in os.walk(data_dir):
                    for file in files:
                        if file.endswith((".csv", ".parquet", ".pq", ".txt")):
                            local_path = os.path.join(root, file)
                            rel_path = os.path.relpath(local_path, data_dir)
                            try:
                                storage_svc.upload_file(local_path, rel_path)
                                count += 1
                            except Exception:
                                pass
                logger.info(f"Synced {count} local dataset files to Supabase Storage.")
    except Exception as se:
        logger.warning(f"Storage sync skipped: {se}")

    logger.info("🎉 Cloud migration complete! DataPilot-AI is now fully powered by Supabase Cloud PostgreSQL and Cloud Storage.")


if __name__ == "__main__":
    migrate_data()
