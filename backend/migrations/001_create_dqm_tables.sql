
CREATE TABLE data_profiling_results (
    id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    profile_type TEXT NOT NULL,
    result JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_data_profiling_results_table_column ON data_profiling_results (table_name, column_name);

CREATE TABLE data_quality_rules (
    id SERIAL PRIMARY KEY,
    rule_name TEXT NOT NULL,
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    rule_type TEXT NOT NULL,
    rule_definition JSONB NOT NULL,
    severity TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_data_quality_rules_table_column ON data_quality_rules (table_name, column_name);

CREATE TABLE data_quality_results (
    id SERIAL PRIMARY KEY,
    rule_id INT REFERENCES data_quality_rules(id) ON DELETE CASCADE,
    passed BOOLEAN NOT NULL,
    failed_count INT NOT NULL,
    total_count INT NOT NULL,
    failure_samples JSONB,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_data_quality_results_rule_id ON data_quality_results (rule_id);

CREATE TABLE root_cause_analysis (
    id SERIAL PRIMARY KEY,
    result_id INT REFERENCES data_quality_results(id) ON DELETE CASCADE,
    suggested_cause JSONB NOT NULL,
    confidence_score NUMERIC(5, 2) NOT NULL,
    ai_model TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_root_cause_analysis_result_id ON root_cause_analysis (result_id);

