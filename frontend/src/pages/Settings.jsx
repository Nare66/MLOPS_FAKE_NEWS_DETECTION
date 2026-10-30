import React, { useState } from 'react';
import { Settings as SettingsIcon, Save, RefreshCw, Server, Key, Database, CheckCircle, AlertCircle } from 'lucide-react';
import { api, getAdminKey } from '../api';

export default function Settings({ modelInfo }) {
  const [apiUrl, setApiUrl] = useState(localStorage.getItem('API_BASE_URL') || 'http://localhost:8000');
  const [adminKey, setAdminKey] = useState(getAdminKey());
  const [testResult, setTestResult] = useState(null);
  const [testing, setTesting] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const handleSave = (e) => {
    e.preventDefault();
    localStorage.setItem('API_BASE_URL', apiUrl.trim());
    localStorage.setItem('ADMIN_API_KEY', adminKey.trim());
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  const testConnection = async () => {
    setTesting(true);
    setTestResult(null);
    const start = performance.now();
    try {
      const data = await api.healthCheck();
      const latency = Math.round(performance.now() - start);
      setTestResult({
        success: true,
        latency,
        data
      });
    } catch (err) {
      setTestResult({
        success: false,
        message: err.message || 'Cannot reach FastAPI server.'
      });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="page-container">
      <div className="dashboard-card" style={{ maxWidth: '800px', margin: '0 auto 24px' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <SettingsIcon size={20} style={{ color: '#3b82f6' }} />
              Platform & Environment Configuration
            </div>
            <div className="card-subtitle">
              Configure backend service URLs, administrative keys, and connection diagnostics
            </div>
          </div>
        </div>

        <form onSubmit={handleSave}>
          <div className="form-group">
            <label className="form-label">
              <Server size={14} style={{ display: 'inline', marginRight: '6px' }} />
              FastAPI Backend Base URL
            </label>
            <input
              type="text"
              className="input-control"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
              required
            />
            <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginTop: '6px' }}>
              Base endpoint for REST API routes (/predict, /metrics, /retrain, etc.)
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">
              <Key size={14} style={{ display: 'inline', marginRight: '6px' }} />
              Administrative API Key (Continuous Learning Auth)
            </label>
            <input
              type="password"
              className="input-control"
              value={adminKey}
              onChange={(e) => setAdminKey(e.target.value)}
              placeholder="admin-mlops-secret-2026"
              required
            />
            <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginTop: '6px' }}>
              Required to trigger automated model retraining pipelines
            </div>
          </div>

          {saveSuccess && (
            <div style={{ padding: '12px', backgroundColor: 'rgba(16,185,129,0.12)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '8px', color: '#34d399', marginBottom: '16px', fontSize: '0.85rem' }}>
              Configuration successfully saved to local storage!
            </div>
          )}

          <div style={{ display: 'flex', gap: '12px' }}>
            <button type="submit" className="btn btn-primary">
              <Save size={15} /> Save Settings
            </button>
            <button type="button" className="btn btn-secondary" onClick={testConnection} disabled={testing}>
              <RefreshCw size={15} className={testing ? 'animate-spin' : ''} />
              {testing ? 'Pinging Backend...' : 'Test Connection'}
            </button>
          </div>
        </form>

        {testResult && (
          <div style={{
            marginTop: '20px',
            padding: '16px',
            borderRadius: '8px',
            backgroundColor: testResult.success ? 'rgba(16,185,129,0.08)' : 'rgba(244,63,94,0.08)',
            border: `1px solid ${testResult.success ? 'rgba(16,185,129,0.3)' : 'rgba(244,63,94,0.3)'}`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, color: testResult.success ? '#10b981' : '#f43f5e', marginBottom: '6px' }}>
              {testResult.success ? <CheckCircle size={18} /> : <AlertCircle size={18} />}
              {testResult.success ? `Connected Successfully (${testResult.latency} ms roundtrip)` : 'Connection Failed'}
            </div>
            {testResult.success ? (
              <div style={{ fontSize: '0.82rem', color: '#cbd5e1' }}>
                Service: <strong>{testResult.data.service}</strong> | Model: <strong>{testResult.data.active_model_version}</strong> ({testResult.data.model_status})
              </div>
            ) : (
              <div style={{ fontSize: '0.82rem', color: '#fb7185' }}>
                {testResult.message}
              </div>
            )}
          </div>
        )}
      </div>

      {/* System Architecture Specifications */}
      <div className="dashboard-card" style={{ maxWidth: '800px', margin: '0 auto' }}>
        <div className="card-header">
          <div>
            <div className="card-title">Architecture Specifications</div>
            <div className="card-subtitle">MLOps stack integration overview</div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '14px', fontSize: '0.85rem' }}>
          <div style={{ background: 'var(--bg-secondary)', padding: '14px', borderRadius: '8px' }}>
            <div style={{ color: '#94a3b8', marginBottom: '4px' }}>Deep Learning Framework</div>
            <div style={{ fontWeight: 700, color: '#f8fafc' }}>PyTorch & Hugging Face</div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '14px', borderRadius: '8px' }}>
            <div style={{ color: '#94a3b8', marginBottom: '4px' }}>Experiment Tracking</div>
            <div style={{ fontWeight: 700, color: '#f8fafc' }}>MLflow 3.x (SQLite URI)</div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '14px', borderRadius: '8px' }}>
            <div style={{ color: '#94a3b8', marginBottom: '4px' }}>Data Version Control</div>
            <div style={{ fontWeight: 700, color: '#f8fafc' }}>DVC 3.x Pipeline</div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '14px', borderRadius: '8px' }}>
            <div style={{ color: '#94a3b8', marginBottom: '4px' }}>Drift & Quality Monitoring</div>
            <div style={{ fontWeight: 700, color: '#f8fafc' }}>Evidently AI 0.7.x</div>
          </div>
        </div>
      </div>
    </div>
  );
}
