"""Data Profiling API Routes - Expert Implementation"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.data_profiling import DataProfilingService, ProfileResult

router = APIRouter()


class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: str


class ProfilingResultResponse(BaseModel):
    table_name: str
    column_name: str
    profile_type: str
    result: dict
    created_at: Optional[str] = None


class ProfileRunResponse(BaseModel):
    table: str
    columns_profiled: int
    stored: int


# Dependency to get database session
async def get_db():
    from app.main import app
    async for session in app.state.get_db():
        yield session


@router.get("/tables", response_model=List[str])
async def list_tables(db: AsyncSession = Depends(get_db)):
    """List all available tables for profiling."""
    service = DataProfilingService(db)
    return await service.get_tables()


@router.get("/tables/{table}/columns", response_model=List[ColumnInfo])
async def get_table_columns(table: str, db: AsyncSession = Depends(get_db)):
    """Get column metadata for a table."""
    service = DataProfilingService(db)
    try:
        columns = await service.get_columns(table)
        return columns
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/profile/{table}/{column}")
async def profile_column(
    table: str, 
    column: str, 
    db: AsyncSession = Depends(get_db)
):
    """Profile a specific column."""
    service = DataProfilingService(db)
    try:
        result = await service.profile_column(table, column)
        return {
            "table": result.table_name,
            "column": result.column_name,
            "profile_type": result.profile_type,
            "result": result.result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profile/{table}/run", response_model=ProfileRunResponse)
async def run_table_profiling(
    table: str, 
    db: AsyncSession = Depends(get_db)
):
    """Run full profiling on a table and store results."""
    service = DataProfilingService(db)
    try:
        results = await service.profile_table(table)
        stored = await service.store_results(results)
        return {
            "table": table,
            "columns_profiled": len(results),
            "stored": stored
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/results")
async def get_profiling_results(
    table: Optional[str] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """Get stored profiling results."""
    service = DataProfilingService(db)
    return await service.get_stored_results(table=table, limit=limit)
