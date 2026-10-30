import React, { useState } from 'react';
import {
  Search,
  CheckCircle2,
  AlertOctagon,
  Sparkles,
  Info,
  Clock,
  RotateCcw,
  BookOpen
} from 'lucide-react';
import { api } from '../api';

const SAMPLES = [
  {
    title: 'Real Science Sample (JWST)',
    text: "NASA's James Webb Space Telescope has captured deep-field observations revealing galaxies that formed just 300 million years after the Big Bang. Astronomers utilized the Near-Infrared Camera to analyze cosmological redshift values confirming unprecedented distances in early cosmic evolution.",
    type: 'real'
  },
  {
    title: 'Disinformation Sample (Alien Pyramid)',
    text: "Declassified military satellite scans have conclusively confirmed a massive triangular crystalline pyramid buried beneath the Queen Maud Land ice shelf. Whistleblowers allege the facility has been generating high-frequency quantum energy since prehistoric times and is linked to extraterrestrial contact covertly hidden by world governments.",
    type: 'fake'
  },
  {
    title: 'Real Economic Sample (Fed Rates)',
    text: "The Federal Reserve announced on Wednesday that it will maintain the benchmark interest rate within the current target range as policymakers evaluate economic indicators. Federal Reserve Chairman stated that while inflation has decelerated substantially from its peak, sustained economic growth and tight labor markets warrant continued vigilance before implementing monetary easing.",
    type: 'real'
  },
  {
    title: 'Fake Health Myth (Lemon Juice)',
    text: "Renowned homeopathic healers have exposed an suppressed natural cure that eradicates all malignant tumors within two days. The combination of freshly squeezed lemon juice and pure sodium bicarbonate alkalizes systemic bloodstream pH levels, starving cancer cells instantly without conventional chemotherapy.",
    type: 'fake'
  }
];

export default function DetectNews({ modelInfo }) {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!text.trim() || text.trim().length < 10) {
      setError('Please provide at least 10 characters of news content.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await api.predictNews(text.trim());
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze text. Please ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const loadSample = (sample) => {
    setText(sample.text);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setError(null);
  };

  const isReal = result?.prediction === 'REAL';
  const confidencePercent = result ? Math.round(result.confidence * 100) : 0;
  const fakeProb = result ? Math.round((result.probabilities?.FAKE || 0) * 100) : 0;
  const realProb = result ? Math.round((result.probabilities?.REAL || 0) * 100) : 0;

  return (
    <div className="page-container">
      <div className="dashboard-card">
        <div className="card-header">
          <div>
            <div className="card-title">
              <Search size={20} style={{ color: '#3b82f6' }} />
              Live BERT News Detection Engine
            </div>
            <div className="card-subtitle">
              Input article text or headline to classify credibility via fine-tuned Transformers
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            {SAMPLES.map((s, idx) => (
              <button
                key={idx}
                type="button"
                className="btn btn-secondary"
                style={{ fontSize: '0.75rem', padding: '6px 10px' }}
                onClick={() => loadSample(s)}
              >
                <BookOpen size={12} /> {s.title}
              </button>
            ))}
          </div>
        </div>

        <form onSubmit={handleAnalyze}>
          <div className="form-group">
            <label className="form-label">
              Article Content or Headline
              <span style={{ float: 'right', color: '#64748b', fontWeight: 400, fontSize: '0.8rem' }}>
                {text.length} characters | {text.trim() ? text.trim().split(/\s+/).length : 0} words
              </span>
            </label>
            <textarea
              className="textarea-control"
              placeholder="Paste the news story, transcript, or editorial text here for deep BERT analysis..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={6}
            />
          </div>

          {error && (
            <div style={{ padding: '12px 16px', backgroundColor: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.3)', borderRadius: '8px', color: '#fb7185', marginBottom: '16px', fontSize: '0.88rem' }}>
              {error}
            </div>
          )}

          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || text.trim().length < 10}
              style={{ minWidth: '180px' }}
            >
              {loading ? (
                <>
                  <Sparkles size={16} className="animate-spin" /> Evaluating with BERT...
                </>
              ) : (
                <>
                  <Search size={16} /> Analyze News Article
                </>
              )}
            </button>

            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleClear}
              disabled={loading || (!text && !result)}
            >
              <RotateCcw size={14} /> Clear
            </button>
          </div>
        </form>

        {/* Prediction Results Banner */}
        {result && (
          <div className={`prediction-result-banner ${isReal ? 'banner-real' : 'banner-fake'}`}>
            <div className="banner-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                <span className={`banner-badge ${isReal ? 'badge-real-big' : 'badge-fake-big'}`}>
                  {isReal ? <CheckCircle2 size={24} /> : <AlertOctagon size={24} />}
                  {isReal ? 'REAL NEWS' : 'FAKE NEWS'}
                </span>
                <div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff' }}>
                    {isReal ? 'Credible & Authentic Article' : 'Potential Disinformation Detected'}
                  </div>
                  <div style={{ fontSize: '0.82rem', color: '#cbd5e1', display: 'flex', gap: '12px', marginTop: '2px' }}>
                    <span>Confidence: <strong>{confidencePercent}%</strong></span>
                    <span>•</span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Clock size={13} /> {result.latency_ms} ms latency
                    </span>
                    <span>•</span>
                    <span>Model: <code>{result.model_version}</code></span>
                  </div>
                </div>
              </div>
            </div>

            {/* Probability Breakdown */}
            <div className="confidence-bar-container">
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '6px', color: '#cbd5e1' }}>
                <span>Fake Probability: {fakeProb}%</span>
                <span>Real Probability: {realProb}%</span>
              </div>
              <div className="progress-track" style={{ height: '12px', display: 'flex', overflow: 'hidden' }}>
                <div style={{ width: `${fakeProb}%`, backgroundColor: '#f43f5e', transition: 'width 0.5s ease' }} title={`Fake: ${fakeProb}%`} />
                <div style={{ width: `${realProb}%`, backgroundColor: '#10b981', transition: 'width 0.5s ease' }} title={`Real: ${realProb}%`} />
              </div>
            </div>

            {/* Mandatory Disclaimer */}
            <div className="disclaimer-box">
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                <Info size={16} style={{ color: '#f59e0b', flexShrink: 0, marginTop: '2px' }} />
                <span>
                  <strong>Assessment Disclaimer:</strong> {result.disclaimer}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
