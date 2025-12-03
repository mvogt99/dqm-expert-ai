"""
AI Analysis Service - Expert Implementation
Uses LOCAL AI models for intelligent root cause analysis
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RootCauseAnalysis:
    """Structured root cause analysis result."""
    result_id: int
    suggested_cause: str
    confidence_score: float
    ai_model: str
    supporting_evidence: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class AIAnalysisService:
    """
    Service for AI-powered data quality analysis.
    Uses LOCAL AI models (RTX 5090/3050) for root cause analysis.
    """
    
    # Prompt templates for analysis
    ROOT_CAUSE_PROMPT = """Analyze this data quality failure and suggest root causes:

Rule: {rule_name}
Table: {table}
Column: {column}
Rule Type: {rule_type}
Failed Records: {failed_count} out of {total_count} ({fail_rate}%)
Sample Failures: {samples}

Provide:
1. Most likely root cause (1-2 sentences)
2. Confidence score (0-1)
3. Supporting evidence from the samples
4. Recommended remediation steps

Format as JSON:
{{"root_cause": "...", "confidence": 0.X, "evidence": ["..."], "remediation": ["..."]}}
"""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.http_client = httpx.AsyncClient(timeout=60.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
    
    async def _call_local_ai(
        self, 
        prompt: str, 
        model_url: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """Call local AI model for analysis."""
        url = model_url or settings.local_ai_planning_url
        
        try:
            response = await self.http_client.post(
                f"{url}/chat/completions",
                json={
                    "model": "auto",  # Use whatever model is loaded
                    "messages": [
                        {"role": "system", "content": "You are a data quality expert. Analyze issues and provide actionable insights."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Local AI call failed: {e}")
            raise
    
    async def analyze_failure(self, result_id: int) -> RootCauseAnalysis:
        """Analyze a specific DQ result failure using LOCAL AI."""
        # Get the result details
        query = text("""
            SELECT r.id, r.rule_id, r.passed, r.failed_count, r.total_count, 
                   r.failure_samples, dq.rule_name, dq.table_name, dq.column_name,
                   dq.rule_type
            FROM data_quality_results r
            JOIN data_quality_rules dq ON r.rule_id = dq.id
            WHERE r.id = :result_id
        """)
        result = await self.session.execute(query, {"result_id": result_id})
        row = result.fetchone()
        
        if not row:
            raise ValueError(f"Result {result_id} not found")
        
        if row[2]:  # passed = true
            raise ValueError("Cannot analyze a passing result")
        
        # Format prompt with result data
        fail_rate = round(row[3] / row[4] * 100, 2) if row[4] > 0 else 0
        samples = row[5] if isinstance(row[5], list) else json.loads(row[5] or "[]")
        
        prompt = self.ROOT_CAUSE_PROMPT.format(
            rule_name=row[6],
            table=row[7],
            column=row[8],
            rule_type=row[9],
            failed_count=row[3],
            total_count=row[4],
            fail_rate=fail_rate,
            samples=json.dumps(samples[:5], default=str, indent=2)
        )
        
        # Call LOCAL AI for analysis
        try:
            ai_response = await self._call_local_ai(prompt)
            
            # Parse AI response
            analysis = self._parse_ai_response(ai_response)
            
            root_cause = RootCauseAnalysis(
                result_id=result_id,
                suggested_cause=analysis.get("root_cause", "Unable to determine root cause"),
                confidence_score=float(analysis.get("confidence", 0.5)),
                ai_model="local-qwen-32b",
                supporting_evidence=analysis.get("evidence", []),
                remediation_steps=analysis.get("remediation", [])
            )
            
            # Store in database
            await self._store_analysis(root_cause)
            
            return root_cause
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            # Return fallback analysis
            return RootCauseAnalysis(
                result_id=result_id,
                suggested_cause=f"Analysis failed: {str(e)}. Manual review recommended.",
                confidence_score=0.0,
                ai_model="fallback",
                supporting_evidence=[],
                remediation_steps=["Review the failure samples manually", "Check data source systems"]
            )
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response, handling various formats."""
        # Try to extract JSON from response
        try:
            # Look for JSON block
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Fallback: construct from text
        return {
            "root_cause": response[:500],
            "confidence": 0.5,
            "evidence": [],
            "remediation": ["Manual review recommended"]
        }
    
    async def _store_analysis(self, analysis: RootCauseAnalysis) -> int:
        """Store analysis in database."""
        query = text("""
            INSERT INTO root_cause_analysis 
            (result_id, suggested_cause, confidence_score, ai_model, created_at)
            VALUES (:result_id, :cause, :confidence, :model, :created_at)
            RETURNING id
        """)
        result = await self.session.execute(query, {
            "result_id": analysis.result_id,
            "cause": json.dumps({
                "cause": analysis.suggested_cause,
                "evidence": analysis.supporting_evidence,
                "remediation": analysis.remediation_steps
            }),
            "confidence": analysis.confidence_score,
            "model": analysis.ai_model,
            "created_at": analysis.created_at
        })
        await self.session.commit()
        return result.fetchone()[0]
    
    async def batch_analyze(self, result_ids: List[int]) -> List[RootCauseAnalysis]:
        """Analyze multiple failures in batch."""
        analyses = []
        for result_id in result_ids:
            try:
                analysis = await self.analyze_failure(result_id)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze result {result_id}: {e}")
        return analyses
    
    async def get_analyses(
        self, 
        limit: int = 50,
        min_confidence: float = 0
    ) -> List[Dict[str, Any]]:
        """Get stored analyses with optional confidence filter."""
        query = text("""
            SELECT rca.id, rca.result_id, rca.suggested_cause, rca.confidence_score,
                   rca.ai_model, rca.created_at, dqr.rule_id, dq.rule_name
            FROM root_cause_analysis rca
            JOIN data_quality_results dqr ON rca.result_id = dqr.id
            JOIN data_quality_rules dq ON dqr.rule_id = dq.id
            WHERE rca.confidence_score >= :min_confidence
            ORDER BY rca.created_at DESC
            LIMIT :limit
        """)
        result = await self.session.execute(query, {
            "min_confidence": min_confidence,
            "limit": limit
        })
        
        return [
            {
                "id": row[0],
                "result_id": row[1],
                "analysis": row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}"),
                "confidence_score": float(row[3]) if row[3] else 0,
                "ai_model": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
                "rule_id": row[6],
                "rule_name": row[7]
            }
            for row in result.fetchall()
        ]
    
    async def get_analysis(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific analysis by ID."""
        query = text("""
            SELECT rca.id, rca.result_id, rca.suggested_cause, rca.confidence_score,
                   rca.ai_model, rca.created_at
            FROM root_cause_analysis rca
            WHERE rca.id = :id
        """)
        result = await self.session.execute(query, {"id": analysis_id})
        row = result.fetchone()
        
        if not row:
            return None
        
        return {
            "id": row[0],
            "result_id": row[1],
            "analysis": row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}"),
            "confidence_score": float(row[3]) if row[3] else 0,
            "ai_model": row[4],
            "created_at": row[5].isoformat() if row[5] else None
        }
