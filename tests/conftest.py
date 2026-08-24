import os
import pytest

# Ensure all tests run against an isolated SQLite test database rather than live Supabase
os.environ["DATABASE_URL"] = "sqlite:////tmp/datapilot_test_suite.db"
os.environ["STORAGE_BACKEND"] = "local"
os.environ["STORAGE_LOCAL_DIR"] = "/tmp/datapilot_test_storage"

from backend.database.connection import init_db

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Initializes isolated test database tables before tests run."""
    init_db()
    yield
    # Cleanup test db file after test suite completes
    if os.path.exists("/tmp/datapilot_test_suite.db"):
        try:
            os.remove("/tmp/datapilot_test_suite.db")
        except Exception:
            pass
