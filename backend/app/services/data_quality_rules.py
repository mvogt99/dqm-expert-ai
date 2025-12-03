"""
Data Quality Rules Service - Expert Implementation
Rule suggestion, creation, execution, and failure analysis
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)


class RuleType(str, Enum):
    """Supported data quality rule types."""
    NULL_CHECK = "null_check"
    UNIQUE_CHECK = "unique_check"
    RANGE_CHECK = "range_check"
    PATTERN_CHECK = "pattern_check"
    REFERENTIAL_CHECK = "referential_check"
    CUSTOM_SQL = "custom_sql"


class Severity(str, Enum):
    """Rule severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class RuleDefinition:
    """Structured rule definition."""
    rule_type: RuleType
    threshold: Optional[float] = None
    pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    reference_table: Optional[str] = None
    reference_column: Optional[str] = None
    custom_sql: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class RuleResult:
    """Result of rule execution."""
    rule_id: int
    passed: bool
    total_count: int
    failed_count: int
    failure_samples: List[Dict[str, Any]] = field(default_factory=list)
    executed_at: datetime = None
    
    def __post_init__(self):
        if self.executed_at is None:
            self.executed_at = datetime.utcnow()
    
    @property
    def pass_rate(self) -> float:
        if self.total_count == 0:
            return 100.0
        return round((self.total_count - self.failed_count) / self.total_count * 100, 2)


class DataQualityRulesService:
    """Service for managing and executing data quality rules."""
    
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
    
    async def suggest_rules(self, profiling_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze profiling results and suggest appropriate DQ rules.
        Uses heuristics to identify potential data quality issues.
        """
        suggestions = []
        
        for profile in profiling_results:
            table = profile.get("table_name")
            column = profile.get("column_name")
            result = profile.get("result", {})
            
            # Suggest null check if nulls found
            nulls = result.get("nulls", {})
            if nulls.get("null_percentage", 0) > 0:
                suggestions.append({
                    "table": table,
                    "column": column,
                    "rule_type": RuleType.NULL_CHECK.value,
                    "reason": f"Column has {nulls['null_percentage']}% null values",
                    "suggested_threshold": 0 if nulls['null_percentage'] < 5 else 5,
                    "severity": Severity.CRITICAL.value if nulls['null_percentage'] > 10 else Severity.WARNING.value
                })
            
            # Suggest uniqueness check if duplicates found
            uniqueness = result.get("uniqueness", {})
            if uniqueness.get("duplicate_count", 0) > 0:
                suggestions.append({
                    "table": table,
                    "column": column,
                    "rule_type": RuleType.UNIQUE_CHECK.value,
                    "reason": f"Column has {uniqueness['duplicate_count']} duplicate values",
                    "severity": Severity.WARNING.value
                })
            
            # Suggest range check for numeric columns with outliers
            stats = result.get("statistics", {})
            if stats.get("min") is not None and stats.get("max") is not None:
                min_val, max_val = stats["min"], stats["max"]
                # Check for potential outliers (negative values where positives expected)
                if min_val < 0 and "price" in column.lower():
                    suggestions.append({
                        "table": table,
                        "column": column,
                        "rule_type": RuleType.RANGE_CHECK.value,
                        "reason": f"Negative values found (min={min_val}) for price column",
                        "suggested_min": 0,
                        "severity": Severity.CRITICAL.value
                    })
        
        return suggestions
    
    async def create_rule(
        self,
        name: str,
        table: str,
        column: str,
        rule_type: RuleType,
        definition: Dict[str, Any],
        severity: Severity = Severity.WARNING
    ) -> int:
        """Create a new data quality rule."""
        table = self._validate_table(table)
        
        query = text("""
            INSERT INTO data_quality_rules 
            (rule_name, table_name, column_name, rule_type, rule_definition, severity)
            VALUES (:name, :table, :column, :rule_type, :definition, :severity)
            RETURNING id
        """)
        result = await self.session.execute(query, {
            "name": name,
            "table": table,
            "column": column,
            "rule_type": rule_type.value if isinstance(rule_type, RuleType) else rule_type,
            "definition": json.dumps(definition),
            "severity": severity.value if isinstance(severity, Severity) else severity
        })
        await self.session.commit()
        return result.fetchone()[0]
    
    async def get_rules(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all rules, optionally filtered by active status."""
        if active_only:
            query = text("""
                SELECT id, rule_name, table_name, column_name, rule_type, 
                       rule_definition, severity, is_active, created_at
                FROM data_quality_rules
                WHERE is_active = true
                ORDER BY created_at DESC
            """)
        else:
            query = text("""
                SELECT id, rule_name, table_name, column_name, rule_type,
                       rule_definition, severity, is_active, created_at
                FROM data_quality_rules
                ORDER BY created_at DESC
            """)
        result = await self.session.execute(query)
        return [
            {
                "id": row[0],
                "name": row[1],
                "table": row[2],
                "column": row[3],
                "rule_type": row[4],
                "definition": row[5],
                "severity": row[6],
                "is_active": row[7],
                "created_at": row[8].isoformat() if row[8] else None
            }
            for row in result.fetchall()
        ]
    
    async def get_rule(self, rule_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific rule by ID."""
        query = text("""
            SELECT id, rule_name, table_name, column_name, rule_type,
                   rule_definition, severity, is_active, created_at
            FROM data_quality_rules
            WHERE id = :id
        """)
        result = await self.session.execute(query, {"id": rule_id})
        row = result.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "table": row[2],
            "column": row[3],
            "rule_type": row[4],
            "definition": row[5],
            "severity": row[6],
            "is_active": row[7],
            "created_at": row[8].isoformat() if row[8] else None
        }
    
    async def execute_rule(self, rule_id: int) -> RuleResult:
        """Execute a data quality rule and return results."""
        rule = await self.get_rule(rule_id)
        if not rule:
            raise ValueError(f"Rule {rule_id} not found")
        
        table = rule["table"]
        column = rule["column"]
        rule_type = rule["rule_type"]
        definition = rule["definition"] if isinstance(rule["definition"], dict) else json.loads(rule["definition"])
        
        # Execute based on rule type
        if rule_type == RuleType.NULL_CHECK.value:
            result = await self._execute_null_check(table, column, definition)
        elif rule_type == RuleType.UNIQUE_CHECK.value:
            result = await self._execute_unique_check(table, column)
        elif rule_type == RuleType.RANGE_CHECK.value:
            result = await self._execute_range_check(table, column, definition)
        elif rule_type == RuleType.PATTERN_CHECK.value:
            result = await self._execute_pattern_check(table, column, definition)
        else:
            raise ValueError(f"Unsupported rule type: {rule_type}")
        
        result.rule_id = rule_id
        
        # Store result
        await self._store_result(result)
        
        return result
    
    async def _execute_null_check(
        self, table: str, column: str, definition: Dict[str, Any]
    ) -> RuleResult:
        """Execute null check rule."""
        threshold = definition.get("threshold", 0)
        
        query = text(f"""
            SELECT 
                COUNT(*) as total,
                COUNT(*) - COUNT({column}) as null_count
            FROM {table}
        """)
        result = await self.session.execute(query)
        row = result.fetchone()
        
        total = row[0]
        failed = row[1]
        null_pct = (failed / total * 100) if total > 0 else 0
        passed = null_pct <= threshold
        
        # Get sample failures
        samples = []
        if failed > 0:
            sample_query = text(f"""
                SELECT * FROM {table} WHERE {column} IS NULL LIMIT 5
            """)
            sample_result = await self.session.execute(sample_query)
            samples = [dict(row._mapping) for row in sample_result.fetchall()]
        
        return RuleResult(
            rule_id=0,
            passed=passed,
            total_count=total,
            failed_count=failed,
            failure_samples=samples
        )
    
    async def _execute_unique_check(self, table: str, column: str) -> RuleResult:
        """Execute uniqueness check rule."""
        query = text(f"""
            SELECT {column}, COUNT(*) as cnt
            FROM {table}
            GROUP BY {column}
            HAVING COUNT(*) > 1
        """)
        result = await self.session.execute(query)
        duplicates = result.fetchall()
        
        total_query = text(f"SELECT COUNT(*) FROM {table}")
        total_result = await self.session.execute(total_query)
        total = total_result.fetchone()[0]
        
        failed = sum(row[1] - 1 for row in duplicates)  # Count extra occurrences
        samples = [{"value": str(row[0]), "count": row[1]} for row in duplicates[:5]]
        
        return RuleResult(
            rule_id=0,
            passed=len(duplicates) == 0,
            total_count=total,
            failed_count=failed,
            failure_samples=samples
        )
    
    async def _execute_range_check(
        self, table: str, column: str, definition: Dict[str, Any]
    ) -> RuleResult:
        """Execute range check rule."""
        min_val = definition.get("min_value")
        max_val = definition.get("max_value")
        
        conditions = []
        if min_val is not None:
            conditions.append(f"{column} < {min_val}")
        if max_val is not None:
            conditions.append(f"{column} > {max_val}")
        
        if not conditions:
            raise ValueError("Range check requires min_value or max_value")
        
        where_clause = " OR ".join(conditions)
        
        query = text(f"""
            SELECT COUNT(*) as failed FROM {table} WHERE {where_clause}
        """)
        result = await self.session.execute(query)
        failed = result.fetchone()[0]
        
        total_query = text(f"SELECT COUNT(*) FROM {table}")
        total_result = await self.session.execute(total_query)
        total = total_result.fetchone()[0]
        
        # Get samples
        sample_query = text(f"""
            SELECT * FROM {table} WHERE {where_clause} LIMIT 5
        """)
        sample_result = await self.session.execute(sample_query)
        samples = [dict(row._mapping) for row in sample_result.fetchall()]
        
        return RuleResult(
            rule_id=0,
            passed=failed == 0,
            total_count=total,
            failed_count=failed,
            failure_samples=samples
        )
    
    async def _execute_pattern_check(
        self, table: str, column: str, definition: Dict[str, Any]
    ) -> RuleResult:
        """Execute regex pattern check rule."""
        pattern = definition.get("pattern")
        if not pattern:
            raise ValueError("Pattern check requires 'pattern' in definition")
        
        query = text(f"""
            SELECT COUNT(*) as failed 
            FROM {table} 
            WHERE {column} IS NOT NULL 
            AND {column}::text !~ :pattern
        """)
        result = await self.session.execute(query, {"pattern": pattern})
        failed = result.fetchone()[0]
        
        total_query = text(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NOT NULL")
        total_result = await self.session.execute(total_query)
        total = total_result.fetchone()[0]
        
        return RuleResult(
            rule_id=0,
            passed=failed == 0,
            total_count=total,
            failed_count=failed,
            failure_samples=[]
        )
    
    async def _store_result(self, result: RuleResult) -> int:
        """Store rule execution result."""
        query = text("""
            INSERT INTO data_quality_results 
            (rule_id, passed, failed_count, total_count, failure_samples, executed_at)
            VALUES (:rule_id, :passed, :failed_count, :total_count, :samples, :executed_at)
            RETURNING id
        """)
        db_result = await self.session.execute(query, {
            "rule_id": result.rule_id,
            "passed": result.passed,
            "failed_count": result.failed_count,
            "total_count": result.total_count,
            "samples": json.dumps(result.failure_samples, default=str),
            "executed_at": result.executed_at
        })
        await self.session.commit()
        return db_result.fetchone()[0]
    
    async def get_results(self, rule_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get execution results, optionally filtered by rule."""
        if rule_id:
            query = text("""
                SELECT r.id, r.rule_id, dq.rule_name, r.passed, r.failed_count, 
                       r.total_count, r.failure_samples, r.executed_at
                FROM data_quality_results r
                JOIN data_quality_rules dq ON r.rule_id = dq.id
                WHERE r.rule_id = :rule_id
                ORDER BY r.executed_at DESC
            """)
            result = await self.session.execute(query, {"rule_id": rule_id})
        else:
            query = text("""
                SELECT r.id, r.rule_id, dq.rule_name, r.passed, r.failed_count,
                       r.total_count, r.failure_samples, r.executed_at
                FROM data_quality_results r
                JOIN data_quality_rules dq ON r.rule_id = dq.id
                ORDER BY r.executed_at DESC
                LIMIT 100
            """)
            result = await self.session.execute(query)
        
        return [
            {
                "id": row[0],
                "rule_id": row[1],
                "rule_name": row[2],
                "passed": row[3],
                "failed_count": row[4],
                "total_count": row[5],
                "pass_rate": round((row[5] - row[4]) / row[5] * 100, 2) if row[5] > 0 else 100,
                "failure_samples": row[6],
                "executed_at": row[7].isoformat() if row[7] else None
            }
            for row in result.fetchall()
        ]
