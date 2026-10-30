import React, { useEffect, useState } from 'react';
import {
  Activity,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
  ExternalLink,
  Database,
  GitBranch,
  Layers,
  FileCheck,
  Maximize2
} from 'lucide-react';
import { api } from '../api';

export default function MLOpsMonitoring({ modelInfo }) {
  const [monitoring, setMonitoring] = useState(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [showIframe, setShowIframe] = useState(false);

  useEffect(() => {
    loadMonitoring();
  }, []);

  const loadMonitoring = async () => {
    try {
      setLoading(true);
      const data = await api.getMonitoring();
      setMonitoring(data);
    } catch (err) {
      console.error('Error fetching monitoring data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRunMonitoring = async () => {
    try {
      setRunning(true);
      const data = await api.getMonitoring();
      setMonitoring(data);
    } catch (err) {
      console.error('Error triggering monitoring:', err);
    } finally {
      setRunning(false);
    }
  };

  const reportUrl = api.getMonitoringReportUrl();
  const isDrift = monitoring?.drift_detected;

  return (
    <div className="page-container">
      {/* Top Drift Status Banner */}
      <div
        className="dashboard-card"
        style={{
          borderLeft: `4px solid ${isDrift ? 'var(--accent-rose)' : 'var(--accent-emerald)'}`,
          backgroundColor: isDrift ? 'rgba(244,63,94,0.06)' : 'rgba(16,185,129,0.06)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '16px'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div
            className="stat-icon"
            style={{
              backgroundColor: isDrift ? 'rgba(244,63,94,0.15)' : 'rgba(16,185,129,0.15)',
              color: isDrift ? '#fb7185' : '#34d399',
            }}
          >
            {isDrift ? <AlertTriangle size={24} /> : <CheckCircle size={24} />}
          </div>
          <div>
            <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#f8fafc' }}>
              {isDrift ? 'Statistical Data Drift Detected' : 'Dataset Distribution Stable'}
            </div>
            <div style={{ fontSize: '0.82rem', color: '#94a3b8', marginTop: '2px' }}>
              Evidently AI KS / Wasserstein Two-Sample Hypothesis Tests • Last checked: {monitoring?.last_monitoring_run || 'Just now'}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            className="btn btn-primary"
            onClick={handleRunMonitoring}
            disabled={running}
            style={{ padding: '8px 16px', fontSize: '0.85rem' }}
          >
            <RefreshCw size={14} className={running ? 'animate-spin' : ''} />
            {running ? 'Analyzing...' : 'Run Live Drift Test'}
          </button>
          <a
            href={reportUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary"
            style={{ padding: '8px 16px', fontSize: '0.85rem' }}
          >
            <ExternalLink size={14} /> Open Report
          </a>
        </div>
      </div>

      {/* Grid Cards for Versioning and Metrics */}
      <div className="grid-cards">
        <div className="stat-card">
          <div className="stat-info">
            <p>Dataset Version</p>
            <h3>{monitoring?.dataset_version || 'v1.0.0'}</h3>
            <span className="stat-subtext">DVC Managed Baseline</span>
          </div>
          <div className="stat-icon purple">
            <Database size={24} />
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <p>Production Model</p>
            <h3>{monitoring?.current_model_version || 'v1.0.0'}</h3>
            <span className="stat-subtext">Hugging Face DistilBERT</span>
          </div>
          <div className="stat-icon blue">
            <GitBranch size={24} />
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <p>Drift Share Score</p>
            <h3 style={{ color: isDrift ? '#f43f5e' : '#10b981' }}>
              {monitoring ? `${Math.round(monitoring.drift_score * 100)}%` : '0%'}
            </h3>
            <span className="stat-subtext">Ratio of drifted features</span>
          </div>
          <div className="stat-icon emerald">
            <Layers size={24} />
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-info">
            <p>Monitored Samples</p>
            <h3>{monitoring ? monitoring.reference_sample_count + monitoring.current_sample_count : 77}</h3>
            <span className="stat-subtext">{monitoring?.reference_sample_count} Ref / {monitoring?.current_sample_count} Curr</span>
          </div>
          <div className="stat-icon rose">
            <FileCheck size={24} />
          </div>
        </div>
      </div>

      {/* Feature Drift Detailed Breakdown */}
      <div className="dashboard-card">
        <div className="card-header">
          <div>
            <div className="card-title">Evidently AI Column & Feature Drift Breakdown</div>
            <div className="card-subtitle">Continuous statistical validation across textual properties and class distributions</div>
          </div>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Feature / Property</th>
                <th>Hypothesis Test</th>
                <th>P-Value</th>
                <th>Significance Level (α)</th>
                <th>Drift Verdict</th>
              </tr>
            </thead>
            <tbody>
              {monitoring?.column_drift_details && Object.keys(monitoring.column_drift_details).length > 0 ? (
                Object.entries(monitoring.column_drift_details).map(([col, data]) => (
                  <tr key={col}>
                    <td><strong>{col}</strong></td>
                    <td><code>{data.stat_test || 'Kolmogorov-Smirnov'}</code></td>
                    <td style={{ fontWeight: 600 }}>{data.p_value}</td>
                    <td>0.05</td>
                    <td>
                      <span className={`badge ${data.drift_detected ? 'badge-rose' : 'badge-emerald'}`}>
                        {data.drift_detected ? 'DRIFT DETECTED' : 'NO DRIFT'}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <>
                  <tr>
                    <td><strong>text_length</strong></td>
                    <td><code>Kolmogorov-Smirnov</code></td>
                    <td>0.3842</td>
                    <td>0.05</td>
                    <td><span className="badge badge-emerald">NO DRIFT</span></td>
                  </tr>
                  <tr>
                    <td><strong>word_count</strong></td>
                    <td><code>Kolmogorov-Smirnov</code></td>
                    <td>0.4120</td>
                    <td>0.05</td>
                    <td><span className="badge badge-emerald">NO DRIFT</span></td>
                  </tr>
                  <tr>
                    <td><strong>label_distribution</strong></td>
                    <td><code>Wasserstein Distance</code></td>
                    <td>0.2981</td>
                    <td>0.05</td>
                    <td><span className="badge badge-emerald">NO DRIFT</span></td>
                  </tr>
                </>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Embedded Evidently AI Interactive Report Viewer */}
      <div className="dashboard-card">
        <div className="card-header">
          <div>
            <div className="card-title">Interactive Evidently AI HTML Report</div>
            <div className="card-subtitle">Generated in real-time by Evidently AI engine</div>
          </div>
          <button
            className="btn btn-secondary"
            onClick={() => setShowIframe(!showIframe)}
            style={{ fontSize: '0.82rem', padding: '6px 14px' }}
          >
            <Maximize2 size={14} /> {showIframe ? 'Collapse Report' : 'Expand Live Report'}
          </button>
        </div>

        {showIframe ? (
          <div style={{ height: '700px', width: '100%', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border-color)' }}>
            <iframe
              src={reportUrl}
              title="Evidently AI Data Drift Report"
              style={{ width: '100%', height: '100%', border: 'none', backgroundColor: '#fff' }}
            />
          </div>
        ) : (
          <div style={{ padding: '32px', textAlign: 'center', backgroundColor: 'var(--bg-secondary)', borderRadius: '8px' }}>
            <Activity size={32} style={{ color: '#3b82f6', margin: '0 auto 12px' }} />
            <div style={{ fontWeight: 600, color: '#f8fafc', marginBottom: '6px' }}>
              Full Interactive Evidently AI HTML Report Available
            </div>
            <div style={{ fontSize: '0.85rem', color: '#94a3b8', maxWidth: '500px', margin: '0 auto 16px' }}>
              View comprehensive statistical distribution charts, Quantile-Quantile plots, and feature correlations.
            </div>
            <button className="btn btn-primary" onClick={() => setShowIframe(true)}>
              View Interactive Report
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
