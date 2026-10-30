import React, { useState, useEffect } from 'react';
import {
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  Play,
  RotateCw,
  Sliders,
  Terminal,
  ShieldCheck,
  Lock,
  GitCompare,
  ArrowUpRight
} from 'lucide-react';
import { api, getAdminKey } from '../api';

export default function ContinuousLearning({ modelInfo, setModelInfo }) {
  const [adminKey, setAdminKey] = useState(getAdminKey());
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  // Retraining controls
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(8);
  const [learningRate, setLearningRate] = useState(0.00002);
  const [versionTag, setVersionTag] = useState('');
  const [forcePromote, setForcePromote] = useState(false);

  // Status & Polling
  const [retrainStatus, setRetrainStatus] = useState(null);
  const [isRetraining, setIsRetraining] = useState(false);

  useEffect(() => {
    checkStatus();
    const interval = setInterval(() => {
      checkStatus();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const checkStatus = async () => {
    try {
      const data = await api.getRetrainStatus();
      setRetrainStatus(data);
      if (data.status === 'RUNNING' || data.status === 'QUEUED') {
        setIsRetraining(true);
      } else {
        setIsRetraining(false);
      }
    } catch (err) {
      console.error('Status check error:', err);
    }
  };

  const handleAdminKeyChange = (val) => {
    setAdminKey(val);
    localStorage.setItem('ADMIN_API_KEY', val);
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setUploadStatus(null);
      setUploadError(null);
    }
  };

  const handleUploadAndValidate = async () => {
    if (!file) return;
    setUploadLoading(true);
    setUploadError(null);
    try {
      const result = await api.uploadDataset(file);
      setUploadStatus(result);
    } catch (err) {
      setUploadError(err.response?.data?.detail || 'Failed to upload dataset.');
    } finally {
      setUploadLoading(false);
    }
  };

  const handleStartRetraining = async () => {
    try {
      setIsRetraining(true);
      await api.triggerRetraining({
        epochs: Number(epochs),
        batch_size: Number(batchSize),
        learning_rate: Number(learningRate),
        model_version_tag: versionTag || undefined,
        force_promote: forcePromote
      });
      checkStatus();
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to initiate retraining.');
      setIsRetraining(false);
    }
  };

  return (
    <div className="page-container">
      {/* Admin Auth Notice */}
      <div className="dashboard-card" style={{ padding: '16px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="stat-icon blue" style={{ width: '40px', height: '40px' }}>
            <Lock size={18} />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>Administrator Authorization</div>
            <div style={{ fontSize: '0.78rem', color: '#94a3b8' }}>Protected MLOps and Continuous Learning Retraining Operations</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <input
            type="password"
            className="input-control"
            value={adminKey}
            onChange={(e) => handleAdminKeyChange(e.target.value)}
            placeholder="Enter Admin API Key..."
            style={{ width: '260px', padding: '8px 12px', fontSize: '0.85rem' }}
          />
          <span className="badge badge-emerald">Key Configured</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '24px' }}>
        {/* Step 1: Upload & Validate New Labelled Data */}
        <div className="dashboard-card">
          <div className="card-header">
            <div>
              <div className="card-title">1. Upload & Validate Labelled CSV</div>
              <div className="card-subtitle">Feed verified REAL and FAKE articles into the continuous learning pool</div>
            </div>
          </div>

          <div
            className="dropzone"
            onClick={() => document.getElementById('csvFileInput').click()}
          >
            <UploadCloud size={40} style={{ color: '#3b82f6', margin: '0 auto 12px' }} />
            <div style={{ fontWeight: 600, color: '#f8fafc', marginBottom: '4px' }}>
              {file ? file.name : 'Click to select labelled news CSV file'}
            </div>
            <div style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
              Expected format: columns `title`, `text`, `label` (REAL / FAKE)
            </div>
            <input
              id="csvFileInput"
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
          </div>

          {file && (
            <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.85rem', color: '#cbd5e1' }}>
                Selected: <strong>{file.name}</strong> ({(file.size / 1024).toFixed(1)} KB)
              </span>
              <button
                className="btn btn-primary"
                onClick={handleUploadAndValidate}
                disabled={uploadLoading}
                style={{ padding: '8px 16px', fontSize: '0.85rem' }}
              >
                {uploadLoading ? 'Validating CSV...' : 'Validate & Stage Data'}
              </button>
            </div>
          )}

          {uploadError && (
            <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(244,63,94,0.12)', border: '1px solid rgba(244,63,94,0.3)', borderRadius: '8px', color: '#fb7185', fontSize: '0.85rem' }}>
              {uploadError}
            </div>
          )}

          {uploadStatus && (
            <div style={{ marginTop: '20px', padding: '16px', backgroundColor: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#10b981', fontWeight: 700, marginBottom: '8px' }}>
                <CheckCircle2 size={18} /> {uploadStatus.validation_message}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginTop: '12px' }}>
                <div style={{ background: 'var(--bg-secondary)', padding: '10px', borderRadius: '6px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Total Records</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 800 }}>{uploadStatus.total_records}</div>
                </div>
                <div style={{ background: 'var(--bg-secondary)', padding: '10px', borderRadius: '6px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>REAL Articles</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#10b981' }}>{uploadStatus.real_count}</div>
                </div>
                <div style={{ background: 'var(--bg-secondary)', padding: '10px', borderRadius: '6px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>FAKE Articles</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#f43f5e' }}>{uploadStatus.fake_count}</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Step 2: Configure & Trigger Retraining */}
        <div className="dashboard-card">
          <div className="card-header">
            <div>
              <div className="card-title">2. Retraining Configuration & Gating</div>
              <div className="card-subtitle">Tune hyperparameters and enforce promotion thresholds</div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Training Epochs</label>
              <input
                type="number"
                min="1"
                max="10"
                className="input-control"
                value={epochs}
                onChange={(e) => setEpochs(e.target.value)}
                disabled={isRetraining}
              />
            </div>

            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Batch Size</label>
              <input
                type="number"
                min="2"
                max="32"
                className="input-control"
                value={batchSize}
                onChange={(e) => setBatchSize(e.target.value)}
                disabled={isRetraining}
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Learning Rate</label>
              <input
                type="number"
                step="0.00001"
                className="input-control"
                value={learningRate}
                onChange={(e) => setLearningRate(e.target.value)}
                disabled={isRetraining}
              />
            </div>

            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Version Tag</label>
              <input
                type="text"
                placeholder="e.g. v1.1.0"
                className="input-control"
                value={versionTag}
                onChange={(e) => setVersionTag(e.target.value)}
                disabled={isRetraining}
              />
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px', padding: '10px 14px', backgroundColor: 'var(--bg-secondary)', borderRadius: '8px' }}>
            <input
              type="checkbox"
              id="forcePromoteCheck"
              checked={forcePromote}
              onChange={(e) => setForcePromote(e.target.checked)}
              disabled={isRetraining}
            />
            <label htmlFor="forcePromoteCheck" style={{ fontSize: '0.82rem', color: '#cbd5e1', cursor: 'pointer' }}>
              Override Evaluation Gating (Force promote candidate even if performance regresses)
            </label>
          </div>

          <button
            className="btn btn-primary"
            onClick={handleStartRetraining}
            disabled={isRetraining}
            style={{ width: '100%', padding: '14px' }}
          >
            {isRetraining ? (
              <>
                <RotateCw size={18} className="animate-spin" /> Retraining in Progress...
              </>
            ) : (
              <>
                <Play size={18} /> Trigger Continuous Learning Pipeline
              </>
            )}
          </button>
        </div>
      </div>

      {/* Retraining Status & Live Console */}
      {retrainStatus && (
        <div className="dashboard-card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <Terminal size={20} style={{ color: '#3b82f6' }} />
                Retraining Workflow Status & Execution Logs
              </div>
              <div className="card-subtitle">
                Job ID: <code>{retrainStatus.job_id}</code>
              </div>
            </div>
            <span className={`badge ${
              retrainStatus.status === 'RUNNING' ? 'badge-blue' :
              retrainStatus.status === 'COMPLETED' ? (retrainStatus.promoted ? 'badge-emerald' : 'badge-amber') :
              retrainStatus.status === 'FAILED' ? 'badge-rose' : 'badge-purple'
            }`}>
              {retrainStatus.status}
            </span>
          </div>

          {/* Progress Bar */}
          <div style={{ marginBottom: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
              <span>Step: <strong>{retrainStatus.current_step}</strong></span>
              <span>{retrainStatus.progress}%</span>
            </div>
            <div className="progress-track" style={{ height: '8px' }}>
              <div
                className="progress-fill"
                style={{
                  width: `${retrainStatus.progress}%`,
                  background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
                }}
              />
            </div>
          </div>

          {/* Performance Comparison (if completed) */}
          {retrainStatus.prior_accuracy !== null && (
            <div style={{
              padding: '16px',
              backgroundColor: retrainStatus.promoted ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
              border: `1px solid ${retrainStatus.promoted ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
              borderRadius: '8px',
              marginBottom: '16px'
            }}>
              <div style={{ fontWeight: 700, color: retrainStatus.promoted ? '#10b981' : '#f59e0b', marginBottom: '8px' }}>
                {retrainStatus.message}
              </div>
              <div style={{ display: 'flex', gap: '24px', fontSize: '0.85rem' }}>
                <div>Prior Accuracy: <strong>{Math.round(retrainStatus.prior_accuracy * 100)}%</strong></div>
                <div>Candidate Accuracy: <strong>{Math.round(retrainStatus.new_accuracy * 100)}%</strong></div>
                <div>Prior F1: <strong>{Math.round(retrainStatus.prior_f1 * 100)}%</strong></div>
                <div>Candidate F1: <strong>{Math.round(retrainStatus.new_f1 * 100)}%</strong></div>
              </div>
            </div>
          )}

          {/* Console Output */}
          <div style={{
            backgroundColor: '#090d16',
            borderRadius: '8px',
            padding: '16px',
            fontFamily: 'monospace',
            fontSize: '0.82rem',
            color: '#38bdf8',
            maxHeight: '220px',
            overflowY: 'auto',
            border: '1px solid #1e293b'
          }}>
            {retrainStatus.logs && retrainStatus.logs.length > 0 ? (
              retrainStatus.logs.map((log, idx) => (
                <div key={idx} style={{ marginBottom: '4px' }}>{log}</div>
              ))
            ) : (
              <div style={{ color: '#64748b' }}>Awaiting pipeline execution logs...</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
