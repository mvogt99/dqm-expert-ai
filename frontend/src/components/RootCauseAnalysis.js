import React, { useState, useEffect } from 'react';

function RootCauseAnalysis({ apiBase }) {
  const [failedResults, setFailedResults] = useState([]);
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(null);

  useEffect(() => {
    loadFailedResults();
    loadAnalyses();
  }, []);

  const loadFailedResults = async () => {
    try {
      const res = await fetch(`${apiBase}/data-quality/results`);
      const data = await res.json();
      setFailedResults(data.filter(r => !r.passed));
    } catch (err) {
      console.error('Failed to load results:', err);
    }
  };

  const loadAnalyses = async () => {
    try {
      const res = await fetch(`${apiBase}/ai-analysis/analyses`);
      const data = await res.json();
      setAnalyses(data);
    } catch (err) {
      console.error('Failed to load analyses:', err);
    }
  };

  const analyzeFailure = async (resultId) => {
    setAnalyzing(resultId);
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/ai-analysis/analyze/${resultId}`, {
        method: 'POST'
      });
      const data = await res.json();
      alert(`Analysis complete!\n\nRoot Cause: ${data.root_cause}\nConfidence: ${(data.confidence_score * 100).toFixed(0)}%`);
      loadAnalyses();
    } catch (err) {
      console.error('Analysis failed:', err);
      alert('Analysis failed: ' + err.message);
    } finally {
      setLoading(false);
      setAnalyzing(null);
    }
  };

  const batchAnalyze = async () => {
    const resultIds = failedResults.slice(0, 5).map(r => r.id);
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/ai-analysis/batch-analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result_ids: resultIds })
      });
      const data = await res.json();
      alert(`Batch analysis complete! Analyzed ${data.analyzed} failures.`);
      loadAnalyses();
    } catch (err) {
      console.error('Batch analysis failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="root-cause-analysis">
      <h2>Root Cause Analysis</h2>
      <p className="subtitle">AI-powered analysis using LOCAL models (RTX 5090)</p>

      <div className="failed-results">
        <div className="section-header">
          <h3>Failed DQ Results ({failedResults.length})</h3>
          <button 
            onClick={batchAnalyze} 
            disabled={loading || failedResults.length === 0}
            className="btn secondary"
          >
            {loading ? 'Analyzing...' : '🤖 Batch Analyze (Top 5)'}
          </button>
        </div>

        {failedResults.length === 0 ? (
          <p className="empty-state">No failed results to analyze. Run some data quality rules first!</p>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Rule</th>
                <th>Failed Count</th>
                <th>Pass Rate</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {failedResults.map(r => (
                <tr key={r.id}>
                  <td>{r.rule_name}</td>
                  <td className="failed">{r.failed_count}</td>
                  <td>{r.pass_rate}%</td>
                  <td>
                    <button 
                      onClick={() => analyzeFailure(r.id)}
                      disabled={loading}
                      className="btn small primary"
                    >
                      {analyzing === r.id ? '🔄 Analyzing...' : '🔍 Analyze'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="analyses">
        <h3>AI Analyses ({analyses.length})</h3>
        {analyses.length === 0 ? (
          <p className="empty-state">No analyses yet. Click "Analyze" on a failed result above.</p>
        ) : (
          <div className="analysis-cards">
            {analyses.map(a => (
              <div key={a.id} className="analysis-card">
                <div className="card-header">
                  <strong>{a.rule_name}</strong>
                  <span className={`confidence ${a.confidence_score > 0.7 ? 'high' : a.confidence_score > 0.4 ? 'medium' : 'low'}`}>
                    {(a.confidence_score * 100).toFixed(0)}% confident
                  </span>
                </div>
                <div className="card-body">
                  <div className="root-cause">
                    <h4>Root Cause</h4>
                    <p>{a.analysis?.cause || a.analysis}</p>
                  </div>
                  {a.analysis?.evidence?.length > 0 && (
                    <div className="evidence">
                      <h4>Evidence</h4>
                      <ul>
                        {a.analysis.evidence.map((e, i) => <li key={i}>{e}</li>)}
                      </ul>
                    </div>
                  )}
                  {a.analysis?.remediation?.length > 0 && (
                    <div className="remediation">
                      <h4>Remediation Steps</h4>
                      <ol>
                        {a.analysis.remediation.map((r, i) => <li key={i}>{r}</li>)}
                      </ol>
                    </div>
                  )}
                </div>
                <div className="card-footer">
                  <span className="model">Model: {a.ai_model}</span>
                  <span className="timestamp">{new Date(a.created_at).toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default RootCauseAnalysis;
