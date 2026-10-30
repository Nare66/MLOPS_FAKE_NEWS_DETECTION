import React from 'react';
import {
  LayoutDashboard,
  Search,
  History,
  LineChart,
  RefreshCw,
  Activity,
  Settings as SettingsIcon,
  ShieldCheck,
  Cpu
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, modelInfo }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'detect', label: 'Detect News', icon: Search },
    { id: 'history', label: 'Prediction History', icon: History },
    { id: 'performance', label: 'Model Performance', icon: LineChart },
    { id: 'continuous', label: 'Continuous Learning', icon: RefreshCw },
    { id: 'monitoring', label: 'MLOps Monitoring', icon: Activity },
    { id: 'settings', label: 'Settings', icon: SettingsIcon },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-icon">
          <ShieldCheck size={24} />
        </div>
        <div>
          <div className="brand-title">FakeNews AI</div>
          <div className="brand-subtitle">BERT & MLOps Engine</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon size={18} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="system-status-pill">
          <div className="status-dot"></div>
          <span>FastAPI & BERT Operational</span>
        </div>
        
        <div style={{ padding: '8px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', fontSize: '0.75rem', color: '#64748b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px', color: '#94a3b8', fontWeight: 600 }}>
            <Cpu size={14} /> Active Model
          </div>
          <div>{modelInfo?.active_version || 'v1.0.0'} (DistilBERT)</div>
        </div>
      </div>
    </aside>
  );
}
