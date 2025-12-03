"""AI Analysis API Routes - Expert Implementation"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ai_analysis import AIAnalysisService

router = APIRouter()


class AnalysisResponse(BaseModel):
    id: int
    result_id: int
    analysis: dict
    confidence_score: float
    ai_model: str
    created_at: str = None


class BatchAnalyzeRequest(BaseModel):
    result_ids: List[int]


async def get_db():
    from app.main import app
    async for session in app.state.get_db():
        yield session


@router.post("/analyze/{result_id}")
async def analyze_failure(
    result_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Analyze a data quality failure using LOCAL AI."""
    async with AIAnalysisService(db) as service:
        try:
            analysis = await service.analyze_failure(result_id)
            return {
                "result_id": analysis.result_id,
                "root_cause": analysis.suggested_cause,
                "confidence_score": analysis.confidence_score,
                "evidence": analysis.supporting_evidence,
                "remediation": analysis.remediation_steps,
                "ai_model": analysis.ai_model
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-analyze")
async def batch_analyze(
    request: BatchAnalyzeRequest,
    db: AsyncSession = Depends(get_db)
):
    """Analyze multiple failures in batch."""
    async with AIAnalysisService(db) as service:
        analyses = await service.batch_analyze(request.result_ids)
        return {
            "analyzed": len(analyses),
            "results": [
                {
                    "result_id": a.result_id,
                    "root_cause": a.suggested_cause,
                    "confidence_score": a.confidence_score
                }
                for a in analyses
            ]
        }


@router.get("/analyses")
async def get_analyses(
    limit: int = 50,
    min_confidence: float = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get stored analyses."""
    async with AIAnalysisService(db) as service:
        return await service.get_analyses(limit=limit, min_confidence=min_confidence)


@router.get("/analyses/{analysis_id}")
async def get_analysis(
    analysis_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific analysis."""
    async with AIAnalysisService(db) as service:
        analysis = await service.get_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return analysis
