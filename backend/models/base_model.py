import json
from typing import Any
from sqlalchemy.types import TypeDecorator, TEXT
from sqlalchemy.dialects.postgresql import JSONB


class JSONType(TypeDecorator):
    """
    Platform-independent JSON column type.
    Uses PostgreSQL JSONB in production and text-based JSON on SQLite/other DBs.
    """
    impl = TEXT
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(TEXT())

    def _clean_nan(self, obj: Any) -> Any:
        import math
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        elif isinstance(obj, dict):
            return {k: self._clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_nan(v) for v in obj]
        return obj

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        cleaned = self._clean_nan(value)
        if dialect.name == "postgresql":
            return cleaned
        return json.dumps(cleaned)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value


from sqlalchemy.types import String
from backend.schemas.enums import JobStatus


class JobStatusType(TypeDecorator):
    """
    Robust, case-insensitive SQLAlchemy column type for JobStatus enum.
    Maps string values (lowercase, uppercase, mixed) or Enum instances cleanly to JobStatus.
    """
    impl = String(50)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "value"):
            return str(value.value)
        return str(value).lower()

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        val_str = str(value).strip().lower()
        try:
            return JobStatus(val_str)
        except ValueError:
            for member in JobStatus:
                if member.name.lower() == val_str or member.value.lower() == val_str:
                    return member
            return value

