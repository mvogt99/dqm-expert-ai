import React, { useState, useEffect } from 'react';

function DataQualityRules({ apiBase }) {
  const [rules, setRules] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newRule, setNewRule] = useState({
    name: '', table: '', column: '', rule_type: 'null_check', 
    definition: {}, severity: 'warning'
  });

  useEffect(() => {
    loadRules();
    loadResults();
  }, []);

  const loadRules = async () => {
    const res = await fetch(`${apiBase}/data-quality/rules`);
    const data = await res.json();
    setRules(data);
  };

  const loadResults = async () => {
    const res = await fetch(`${apiBase}/data-quality/results`);
    const data = await res.json();
    setResults(data);
  };

  const getSuggestions = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/data-quality/rules/suggest`);
      const data = await res.json();
      setSuggestions(data.suggestions || []);
    } catch (err) {
      console.error('Failed to get suggestions:', err);
    } finally {
      setLoading(false);
    }
  };

  const executeRule = async (ruleId) => {
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/data-quality/rules/${ruleId}/execute`, {
        method: 'POST'
      });
      const data = await res.json();
      alert(`Rule executed: ${data.passed ? 'PASSED' : 'FAILED'} (${data.pass_rate}% pass rate)`);
      loadResults();
    } catch (err) {
      console.error('Execution failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const createRule = async () => {
    try {
      const res = await fetch(`${apiBase}/data-quality/rules`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newRule)
      });
      if (res.ok) {
        setShowCreateForm(false);
        loadRules();
      }
    } catch (err) {
      console.error('Failed to create rule:', err);
    }
  };

  return (
    <div className="data-quality-rules">
      <h2>Data Quality Rules</h2>
      
      <div className="actions">
        <button onClick={() => setShowCreateForm(true)} className="btn primary">
          + Create Rule
        </button>
        <button onClick={getSuggestions} disabled={loading} className="btn secondary">
          {loading ? 'Loading...' : '💡 Get Suggestions'}
        </button>
      </div>

      {showCreateForm && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Create New Rule</h3>
            <div className="form-group">
              <label>Name</label>
              <input 
                value={newRule.name} 
                onChange={e => setNewRule({...newRule, name: e.target.value})}
              />
            </div>
            <div className="form-group">
              <label>Table</label>
              <input 
                value={newRule.table} 
                onChange={e => setNewRule({...newRule, table: e.target.value})}
              />
            </div>
            <div className="form-group">
              <label>Column</label>
              <input 
                value={newRule.column} 
                onChange={e => setNewRule({...newRule, column: e.target.value})}
              />
            </div>
            <div className="form-group">
              <label>Rule Type</label>
              <select 
                value={newRule.rule_type}
                onChange={e => setNewRule({...newRule, rule_type: e.target.value})}
              >
                <option value="null_check">Null Check</option>
                <option value="unique_check">Unique Check</option>
                <option value="range_check">Range Check</option>
                <option value="pattern_check">Pattern Check</option>
              </select>
            </div>
            <div className="form-actions">
              <button onClick={createRule} className="btn primary">Create</button>
              <button onClick={() => setShowCreateForm(false)} className="btn">Cancel</button>
            </div>
          </div>
        </div>
      )}

      {suggestions.length > 0 && (
        <div className="suggestions">
          <h3>Suggested Rules</h3>
          {suggestions.map((s, i) => (
            <div key={i} className="suggestion-card">
              <span className={`severity ${s.severity}`}>{s.severity}</span>
              <strong>{s.table}.{s.column}</strong>
              <span>{s.rule_type}</span>
              <p>{s.reason}</p>
            </div>
          ))}
        </div>
      )}

      <div className="rules-list">
        <h3>Active Rules ({rules.length})</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Table</th>
              <th>Column</th>
              <th>Type</th>
              <th>Severity</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {rules.map(rule => (
              <tr key={rule.id}>
                <td>{rule.name}</td>
                <td>{rule.table}</td>
                <td>{rule.column}</td>
                <td>{rule.rule_type}</td>
                <td><span className={`severity ${rule.severity}`}>{rule.severity}</span></td>
                <td>
                  <button 
                    onClick={() => executeRule(rule.id)}
                    className="btn small"
                    disabled={loading}
                  >
                    Run
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {results.length > 0 && (
        <div className="results">
          <h3>Recent Results</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>Rule</th>
                <th>Status</th>
                <th>Pass Rate</th>
                <th>Failed</th>
                <th>Executed</th>
              </tr>
            </thead>
            <tbody>
              {results.slice(0, 10).map(r => (
                <tr key={r.id} className={r.passed ? 'passed' : 'failed'}>
                  <td>{r.rule_name}</td>
                  <td>{r.passed ? '✓ PASS' : '✗ FAIL'}</td>
                  <td>{r.pass_rate}%</td>
                  <td>{r.failed_count}</td>
                  <td>{new Date(r.executed_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default DataQualityRules;
