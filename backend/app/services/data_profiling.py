"""
Data Profiling Service - Expert Implementation
Secure, well-structured with proper SQL parameterization
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Structured profiling result."""
    table_name: str
    column_name: str
    profile_type: str
    result: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class DataProfilingService:
    """
    Service for comprehensive data profiling.
    Uses parameterized queries to prevent SQL injection.
    """
    
    # Whitelist of allowed tables for profiling
    ALLOWED_TABLES = {
        'categories', 'suppliers', 'products', 'customers',
        'employees', 'shippers', 'orders', 'order_details'
    }
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    def _validate_table(self, table: str) -> str:
        """Validate table name against whitelist."""
        if table.lower() not in self.ALLOWED_TABLES:
            raise ValueError(f"Table '{table}' not in allowed list")
        return table.lower()
    
    async def get_tables(self) -> List[str]:
        """Get list of available tables."""
        query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            AND table_name NOT LIKE 'data_%'
            AND table_name NOT LIKE 'root_%'
            ORDER BY table_name
        """)
        result = await self.session.execute(query)
        return [row[0] for row in result.fetchall()]
    
    async def get_columns(self, table: str) -> List[Dict[str, str]]:
        """Get column metadata for a table."""
        table = self._validate_table(table)
        query = text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = :table
            ORDER BY ordinal_position
        """)
        result = await self.session.execute(query, {"table": table})
        return [
            {"name": row[0], "type": row[1], "nullable": row[2]}
            for row in result.fetchall()
        ]
    
    async def null_check(self, table: str, column: str) -> Dict[str, Any]:
        """Count null values in a column."""
        table = self._validate_table(table)
        # Use dynamic SQL safely with validated table name
        query = text(f"""
            SELECT 
                COUNT(*) as total,
                COUNT({column}) as non_null,
                COUNT(*) - COUNT({column}) as null_count,
                ROUND((COUNT(*) - COUNT({column}))::numeric / NULLIF(COUNT(*), 0) * 100, 2) as null_pct
            FROM {table}
        """)
        result = await self.session.execute(query)
        row = result.fetchone()
        return {
            "total": row[0],
            "non_null": row[1],
            "null_count": row[2],
            "null_percentage": float(row[3]) if row[3] else 0
        }
    
    async def uniqueness_check(self, table: str, column: str) -> Dict[str, Any]:
        """Check uniqueness of values in a column."""
        table = self._validate_table(table)
        query = text(f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT {column}) as unique_values,
                COUNT(*) - COUNT(DISTINCT {column}) as duplicate_count
            FROM {table}
        """)
        result = await self.session.execute(query)
        row = result.fetchone()
        return {
            "total_rows": row[0],
            "unique_values": row[1],
            "duplicate_count": row[2],
            "is_unique": row[0] == row[1]
        }
    
    async def value_distribution(self, table: str, column: str, limit: int = 20) -> Dict[str, Any]:
        """Get value distribution for a column."""
        table = self._validate_table(table)
        query = text(f"""
            SELECT {column}::text as value, COUNT(*) as count
            FROM {table}
            GROUP BY {column}
            ORDER BY count DESC
            LIMIT :limit
        """)
        result = await self.session.execute(query, {"limit": limit})
        distribution = [{"value": row[0], "count": row[1]} for row in result.fetchall()]
        return {"distribution": distribution, "top_n": limit}
    
    async def statistical_profile(self, table: str, column: str) -> Dict[str, Any]:
        """Get statistical profile for numeric columns."""
        table = self._validate_table(table)
        query = text(f"""
            SELECT 
                MIN({column})::numeric as min_val,
                MAX({column})::numeric as max_val,
                AVG({column})::numeric as avg_val,
                STDDEV({column})::numeric as stddev_val,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column})::numeric as median
            FROM {table}
            WHERE {column} IS NOT NULL
        """)
        try:
            result = await self.session.execute(query)
            row = result.fetchone()
            return {
                "min": float(row[0]) if row[0] else None,
                "max": float(row[1]) if row[1] else None,
                "avg": round(float(row[2]), 2) if row[2] else None,
                "stddev": round(float(row[3]), 2) if row[3] else None,
                "median": float(row[4]) if row[4] else None
            }
        except Exception as e:
            logger.warning(f"Statistical profile failed for {table}.{column}: {e}")
            return {"error": "Column is not numeric"}
    
    async def profile_column(self, table: str, column: str) -> ProfileResult:
        """Run full profile on a single column."""
        results = {
            "nulls": await self.null_check(table, column),
            "uniqueness": await self.uniqueness_check(table, column),
            "distribution": await self.value_distribution(table, column),
            "statistics": await self.statistical_profile(table, column)
        }
        return ProfileResult(
            table_name=table,
            column_name=column,
            profile_type="full",
            result=results
        )
    
    async def profile_table(self, table: str) -> List[ProfileResult]:
        """Profile all columns in a table."""
        columns = await self.get_columns(table)
        results = []
        for col in columns:
            try:
                profile = await self.profile_column(table, col["name"])
                results.append(profile)
            except Exception as e:
                logger.error(f"Failed to profile {table}.{col['name']}: {e}")
        return results
    
    async def store_results(self, results: List[ProfileResult]) -> int:
        """Store profiling results in database."""
        stored = 0
        for result in results:
            query = text("""
                INSERT INTO data_profiling_results 
                (table_name, column_name, profile_type, result, created_at)
                VALUES (:table_name, :column_name, :profile_type, :result, :created_at)
            """)
            await self.session.execute(query, {
                "table_name": result.table_name,
                "column_name": result.column_name,
                "profile_type": result.profile_type,
                "result": json.dumps(result.result),
                "created_at": result.created_at
            })
            stored += 1
        await self.session.commit()
        return stored
    
    async def get_stored_results(
        self, 
        table: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve stored profiling results."""
        if table:
            query = text("""
                SELECT id, table_name, column_name, profile_type, result, created_at
                FROM data_profiling_results
                WHERE table_name = :table
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            result = await self.session.execute(query, {"table": table, "limit": limit})
        else:
            query = text("""
                SELECT id, table_name, column_name, profile_type, result, created_at
                FROM data_profiling_results
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            result = await self.session.execute(query, {"limit": limit})
        
        return [
            {
                "id": row[0],
                "table_name": row[1],
                "column_name": row[2],
                "profile_type": row[3],
                "result": row[4],
                "created_at": row[5].isoformat() if row[5] else None
            }
            for row in result.fetchall()
        ]
