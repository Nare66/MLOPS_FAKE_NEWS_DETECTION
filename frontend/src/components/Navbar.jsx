import React from 'react';
import { Shield, Sparkles, Database, GitBranch } from 'lucide-react';

export default function Navbar({ activeTab, modelInfo, onNavigate }) {
  const titles = {
    dashboard: { title: 'Executive Overview', subtitle: 'Real-time metrics, detection volume, and model health' },
    detect: { title: 'News Detection Studio', subtitle: 'Analyze articles using fine-tuned BERT sequence classification' },
    history: { title: 'Prediction Audit History', subtitle: 'Persistent SQLite event log of analyzed news stories' },
    performance: { title: 'Model Performance & Evaluation', subtitle: 'Confusion matrix, precision-recall metrics, and loss history' },
    continuous: { title: 'Continuous Learning Pipeline', subtitle: 'Automated DVC dataset versioning and BERT retraining gate' },
    monitoring: { title: 'MLOps Monitoring & Drift', subtitle: 'Evidently AI statistical distribution and data drift reports' },
    settings: { title: 'Platform Settings', subtitle: 'Configure endpoints, authentication credentials, and hyperparameters' },
  };

  const current = titles[activeTab] || titles.dashboard;

  return (
    <header className="top-navbar">
      <div className="navbar-title">
        <h1>{current.title}</h1>
        <p>{current.subtitle}</p>
      </div>

      <div className="navbar-actions">
        <span className="badge badge-blue">
          <GitBranch size={13} />
          {modelInfo?.active_version || 'v1.0.0'}
        </span>
        <span className="badge badge-purple">
          <Database size={13} />
          DVC Tracked
        </span>
        <span className="badge badge-emerald">
          <Sparkles size={13} />
          MLflow Active
        </span>
      </div>
    </header>
  );
}
