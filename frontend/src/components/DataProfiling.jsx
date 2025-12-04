import React, { useState, useEffect } from 'react';

function DataProfiling({ apiBase }) {
  const [tables, setTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState('');
  const [columns, setColumns] = useState([]);
  const [profilingResults, setProfilingResults] = useState(null);
  const [storedResults, setStoredResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${apiBase}/data-profiling/tables`)
      .then(res => res.json())
      .then(data => setTables(data))
      .catch(err => setError('Failed to load tables'));
  }, [apiBase]);

  useEffect(() => {
    if (selectedTable) {
      fetch(`${apiBase}/data-profiling/tables/${selectedTable}/columns`)
        .then(res => res.json())
        .then(data => setColumns(data))
        .catch(err => setError('Failed to load columns'));
    }
  }, [selectedTable, apiBase]);

  const runProfiling = async () => {
    if (!selectedTable) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/data-profiling/profile/${selectedTable}/run`, {
        method: 'POST'
      });
      const data = await res.json();
      setProfilingResults(data);
      // Refresh stored results
      loadStoredResults();
    } catch (err) {
      setError('Profiling failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadStoredResults = async () => {
    try {
      const res = await fetch(`${apiBase}/data-profiling/results?table=${selectedTable}`);
      const data = await res.json();
      setStoredResults(data);
    } catch (err) {
      console.error('Failed to load results:', err);
    }
  };

  useEffect(() => {
    if (selectedTable) loadStoredResults();
  }, [selectedTable]);

  return (
    <div className="data-profiling">
      <h2>Data Profiling</h2>
      
      <div className="controls">
        <select 
          value={selectedTable} 
          onChange={e => setSelectedTable(e.target.value)}
          className="select"
        >
          <option value="">Select a table...</option>
          {tables.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
        
        <button 
          onClick={runProfiling} 
          disabled={!selectedTable || loading}
          className="btn primary"
        >
          {loading ? 'Running...' : 'Run Profiling'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {columns.length > 0 && (
        <div className="columns-info">
          <h3>Columns ({columns.length})</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>Column</th>
                <th>Type</th>
                <th>Nullable</th>
              </tr>
            </thead>
            <tbody>
              {columns.map(col => (
                <tr key={col.name}>
                  <td>{col.name}</td>
                  <td>{col.type}</td>
                  <td>{col.nullable}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {profilingResults && (
        <div className="profiling-summary">
          <h3>Profiling Complete</h3>
          <p>Columns profiled: {profilingResults.columns_profiled}</p>
          <p>Results stored: {profilingResults.stored}</p>
        </div>
      )}

      {storedResults.length > 0 && (
        <div className="stored-results">
          <h3>Profiling Results</h3>
          {storedResults.map(result => (
            <div key={result.id} className="result-card">
              <h4>{result.column_name}</h4>
              <div className="metrics">
                {result.result.nulls && (
                  <div className="metric">
                    <span className="label">Nulls:</span>
                    <span className={`value ${result.result.nulls.null_percentage > 0 ? 'warning' : 'success'}`}>
                      {result.result.nulls.null_percentage}%
                    </span>
                  </div>
                )}
                {result.result.uniqueness && (
                  <div className="metric">
                    <span className="label">Unique:</span>
                    <span className={`value ${result.result.uniqueness.is_unique ? 'success' : 'info'}`}>
                      {result.result.uniqueness.unique_values} / {result.result.uniqueness.total_rows}
                    </span>
                  </div>
                )}
                {result.result.statistics?.min !== undefined && (
                  <div className="metric">
                    <span className="label">Range:</span>
                    <span className="value">
                      {result.result.statistics.min} - {result.result.statistics.max}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default DataProfiling;
