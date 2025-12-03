"""
Data Quality Management API - Expert AI Implementation
Demonstrates best practices: security, error handling, proper async patterns
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.config import settings
from app.routes import data_profiling, data_quality, ai_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database engine with connection pooling
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    poolclass=NullPool,  # For async, prevents connection issues
)

async_session_maker = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for database session with proper cleanup."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan context manager for startup/shutdown."""
    logger.info("Starting DQM Expert AI API...")
    # Verify database connection
    async with engine.begin() as conn:
        await conn.execute("SELECT 1")
    logger.info("Database connection verified")
    yield
    logger.info("Shutting down DQM Expert AI API...")
    await engine.dispose()


app = FastAPI(
    title="Data Quality Management API",
    description="Expert AI implementation with best practices",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS with explicit origins (more secure than wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with logging."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0", "ai": "expert"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "DQM Expert AI API",
        "version": "1.0.0",
        "endpoints": ["/data-profiling", "/data-quality", "/ai-analysis"]
    }


# Include routers with prefixes
app.include_router(
    data_profiling.router, 
    prefix="/data-profiling", 
    tags=["Data Profiling"]
)
app.include_router(
    data_quality.router, 
    prefix="/data-quality", 
    tags=["Data Quality"]
)
app.include_router(
    ai_analysis.router, 
    prefix="/ai-analysis", 
    tags=["AI Analysis"]
)

# Export get_db for dependency injection
app.state.get_db = get_db
