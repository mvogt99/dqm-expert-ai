"""Data Quality Rules API Routes - Expert Implementation"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.data_quality_rules import (
    DataQualityRulesService, RuleType, Severity
)

router = APIRouter()


class RuleCreate(BaseModel):
    name: str
    table: str
    column: str
    rule_type: str
    definition: dict
    severity: str = "warning"


class RuleResponse(BaseModel):
    id: int
    name: str
    table: str
    column: str
    rule_type: str
    definition: dict
    severity: str
    is_active: bool
    created_at: Optional[str] = None


class ExecutionResult(BaseModel):
    rule_id: int
    passed: bool
    total_count: int
    failed_count: int
    pass_rate: float
    failure_samples: list


async def get_db():
    from app.main import app
    async for session in app.state.get_db():
        yield session


@router.get("/rules", response_model=List[RuleResponse])
async def list_rules(
    active_only: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """List all data quality rules."""
    service = DataQualityRulesService(db)
    return await service.get_rules(active_only=active_only)


@router.post("/rules", response_model=dict)
async def create_rule(
    rule: RuleCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new data quality rule."""
    service = DataQualityRulesService(db)
    try:
        rule_id = await service.create_rule(
            name=rule.name,
            table=rule.table,
            column=rule.column,
            rule_type=RuleType(rule.rule_type),
            definition=rule.definition,
            severity=Severity(rule.severity)
        )
        return {"id": rule_id, "message": "Rule created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# IMPORTANT: /rules/suggest must be defined BEFORE /rules/{rule_id}
# to prevent "suggest" from being parsed as an integer rule_id
@router.get("/rules/suggest")
async def suggest_rules(db: AsyncSession = Depends(get_db)):
    """Suggest rules based on recent profiling results."""
    from app.services.data_profiling import DataProfilingService

    profiling_service = DataProfilingService(db)
    dq_service = DataQualityRulesService(db)

    # Get recent profiling results
    profiling_results = await profiling_service.get_stored_results(limit=50)

    # Suggest rules
    suggestions = await dq_service.suggest_rules(profiling_results)
    return {"suggestions": suggestions}


@router.get("/rules/{rule_id}")
async def get_rule(rule_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific rule by ID."""
    service = DataQualityRulesService(db)
    rule = await service.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return rule


@router.post("/rules/{rule_id}/execute")
async def execute_rule(rule_id: int, db: AsyncSession = Depends(get_db)):
    """Execute a data quality rule."""
    service = DataQualityRulesService(db)
    try:
        result = await service.execute_rule(rule_id)
        return {
            "rule_id": result.rule_id,
            "passed": result.passed,
            "total_count": result.total_count,
            "failed_count": result.failed_count,
            "pass_rate": result.pass_rate,
            "failure_samples": result.failure_samples
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/results")
async def get_execution_results(
    rule_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get rule execution results."""
    service = DataQualityRulesService(db)
    return await service.get_results(rule_id=rule_id)
