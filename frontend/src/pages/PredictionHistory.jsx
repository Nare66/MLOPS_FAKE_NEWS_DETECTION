import React, { useEffect, useState } from 'react';
import { History, Search, Download, RefreshCw, Calendar, Eye, X } from 'lucide-react';
import { api } from '../api';

export default function PredictionHistory() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterClass, setFilterClass] = useState('ALL');
  const [selectedItem, setSelectedItem] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const data = await api.getHistory(100, 0);
      setItems(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredItems = items.filter((item) => {
    const matchesSearch = item.text.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesClass = filterClass === 'ALL' || item.prediction === filterClass;
    return matchesSearch && matchesClass;
  });

  const exportCSV = () => {
    if (!items.length) return;
    const headers = ['ID', 'Prediction', 'Confidence', 'Latency_ms', 'Model_Version', 'Timestamp', 'Content'];
    const rows = items.map((i) => [
      i.id,
      i.prediction,
      i.confidence,
      i.latency_ms,
      i.model_version,
      `"${i.created_at}"`,
      `"${i.text.replace(/"/g, '""')}"`,
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `fake_news_history_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="page-container">
      <div className="dashboard-card">
        <div className="card-header">
          <div>
            <div className="card-title">
              <History size={20} style={{ color: '#8b5cf6' }} />
              Audit Log & Prediction History
            </div>
            <div className="card-subtitle">
              Persistent SQLite storage of classifications, confidence ratings, and model versions
            </div>
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button className="btn btn-secondary" onClick={exportCSV} disabled={!items.length}>
              <Download size={15} /> Export CSV
            </button>
            <button className="btn btn-secondary" onClick={fetchHistory} disabled={loading}>
              <RefreshCw size={15} className={loading ? 'animate-spin' : ''} /> Refresh
            </button>
          </div>
        </div>

        {/* Filter Controls */}
        <div style={{ display: 'flex', gap: '16px', marginBottom: '20px', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: '240px', position: 'relative' }}>
            <input
              type="text"
              className="input-control"
              placeholder="Search history by keywords..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ paddingLeft: '38px' }}
            />
            <Search size={16} style={{ position: 'absolute', left: '12px', top: '14px', color: '#64748b' }} />
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            {['ALL', 'REAL', 'FAKE'].map((cls) => (
              <button
                key={cls}
                type="button"
                className={`btn btn-secondary ${filterClass === cls ? 'active' : ''}`}
                style={{
                  fontSize: '0.82rem',
                  padding: '8px 16px',
                  backgroundColor: filterClass === cls ? 'var(--bg-card-hover)' : 'transparent',
                  borderColor: filterClass === cls ? 'var(--accent-blue)' : 'var(--border-color)',
                  color: filterClass === cls ? 'var(--accent-blue)' : 'var(--text-secondary)'
                }}
                onClick={() => setFilterClass(cls)}
              >
                {cls}
              </button>
            ))}
          </div>
        </div>

        {/* Table */}
        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Content Excerpt</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Latency</th>
                <th>Model Version</th>
                <th>Audited Date & Time</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan="8" style={{ textAlign: 'center', padding: '36px', color: '#94a3b8' }}>
                    Loading prediction history from SQLite...
                  </td>
                </tr>
              ) : filteredItems.length === 0 ? (
                <tr>
                  <td colSpan="8" style={{ textAlign: 'center', padding: '36px', color: '#64748b' }}>
                    No prediction records matching your criteria.
                  </td>
                </tr>
              ) : (
                filteredItems.map((item) => (
                  <tr key={item.id}>
                    <td><code>#{item.id}</code></td>
                    <td style={{ maxWidth: '360px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {item.text}
                    </td>
                    <td>
                      <span className={`badge ${item.prediction === 'REAL' ? 'badge-emerald' : 'badge-rose'}`}>
                        {item.prediction}
                      </span>
                    </td>
                    <td style={{ fontWeight: 600 }}>{Math.round(item.confidence * 100)}%</td>
                    <td>{item.latency_ms} ms</td>
                    <td><code>{item.model_version}</code></td>
                    <td style={{ color: '#94a3b8', fontSize: '0.82rem' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <Calendar size={13} />
                        {new Date(item.created_at).toLocaleString()}
                      </span>
                    </td>
                    <td>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: '6px 10px', fontSize: '0.78rem' }}
                        onClick={() => setSelectedItem(item)}
                      >
                        <Eye size={13} /> View
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal Dialog for View Details */}
      {selectedItem && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.75)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 50,
          padding: '20px'
        }}>
          <div style={{
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-color)',
            borderRadius: '14px',
            maxWidth: '650px',
            width: '100%',
            padding: '24px',
            position: 'relative',
            maxHeight: '90vh',
            overflowY: 'auto'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span className={`badge ${selectedItem.prediction === 'REAL' ? 'badge-emerald' : 'badge-rose'}`}>
                  {selectedItem.prediction} ({Math.round(selectedItem.confidence * 100)}%)
                </span>
                <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Record #{selectedItem.id}</span>
              </div>
              <button
                onClick={() => setSelectedItem(null)}
                style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
              >
                <X size={20} />
              </button>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '6px' }}>Full Article Content:</div>
              <div style={{ backgroundColor: 'var(--bg-secondary)', padding: '16px', borderRadius: '8px', lineHeight: 1.6, fontSize: '0.92rem', color: '#f8fafc', whiteSpace: 'pre-wrap' }}>
                {selectedItem.text}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', fontSize: '0.8rem', color: '#94a3b8' }}>
              <div>Model: <strong style={{ color: '#fff' }}>{selectedItem.model_version}</strong></div>
              <div>Inference: <strong style={{ color: '#fff' }}>{selectedItem.latency_ms} ms</strong></div>
              <div>Timestamp: <strong style={{ color: '#fff' }}>{new Date(selectedItem.created_at).toLocaleDateString()}</strong></div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
