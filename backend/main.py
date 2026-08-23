import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import get_settings
from backend.database.connection import init_db
from backend.middleware.logging_middleware import logging_middleware_fn
from backend.middleware.auth_middleware import auth_middleware_fn
from backend.middleware.exception_middleware import register_exception_handlers
from backend.api.routes import (
    health_router,
    upload_router,
    jobs_router,
    experiments_router,
    reports_router,
    websocket_router,
    dashboard_router,
    datasets_router,
)

import os
from logging.handlers import RotatingFileHandler

settings = get_settings()

# Setup logging configuration (Console + Rotating File Log)
os.makedirs("backend/logs", exist_ok=True)
log_file_path = "backend/logs/datapilot.log"

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = RotatingFileHandler(
    log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
root_logger.handlers = [console_handler, file_handler]

logger = logging.getLogger("datapilot.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for FastAPI application startup and shutdown events."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version} ({settings.environment})")
    
    # Initialize database tables
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    yield

    logger.info(f"Shutting down {settings.app_name}")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="DataPilot-AI API Gateway - AI-Powered Data Quality & Preprocessing Copilot",
    lifespan=lifespan,
)

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. HTTP Custom Middlewares
@app.middleware("http")
async def custom_auth_middleware(request: Request, call_next):
    return await auth_middleware_fn(request, call_next)


@app.middleware("http")
async def custom_logging_middleware(request: Request, call_next):
    return await logging_middleware_fn(request, call_next)


# 3. Exception Handlers
register_exception_handlers(app)

# 4. Include API Routers (strictly /api/v1 prefixed)
api_prefix = "/api/v1"
app.include_router(health_router, prefix=api_prefix)
app.include_router(upload_router, prefix=api_prefix)
app.include_router(jobs_router, prefix=api_prefix)
app.include_router(experiments_router, prefix=api_prefix)
app.include_router(reports_router, prefix=api_prefix)
app.include_router(dashboard_router, prefix=api_prefix)
app.include_router(websocket_router, prefix=api_prefix)
app.include_router(datasets_router, prefix=api_prefix)


@app.get("/", tags=["Root"])
def root():
    """Root entry point directing to API docs."""
    return {
        "message": f"Welcome to {settings.app_name} API Gateway",
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_url": "/health",
    }

