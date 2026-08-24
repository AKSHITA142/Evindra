from backend.core.config import Settings, get_settings


def test_settings_defaults():
    """Verify settings default values."""
    settings = Settings()
    assert settings.app_name == "DataPilot-AI"
    assert settings.environment in ["development", "production", "testing"]
    assert settings.max_upload_size_mb >= 100


def test_cors_origins_parsing():
    """Verify string CORS origins are converted to list."""
    settings = Settings(CORS_ORIGINS="http://localhost:3000, https://app.datapilot.ai")
    origins = settings.get_cors_origins_list()
    assert origins == ["http://localhost:3000", "https://app.datapilot.ai"]


def test_get_settings_lru_cache():
    """Verify get_settings returns cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
