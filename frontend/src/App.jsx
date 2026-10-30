import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import DetectNews from './pages/DetectNews';
import PredictionHistory from './pages/PredictionHistory';
import ModelPerformance from './pages/ModelPerformance';
import ContinuousLearning from './pages/ContinuousLearning';
import MLOpsMonitoring from './pages/MLOpsMonitoring';
import Settings from './pages/Settings';
import { api } from './api';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const data = await api.getModelInfo();
      setModelInfo(data);
    } catch (err) {
      console.warn('Backend not currently reachable. Ensure FastAPI server is running on port 8000.');
    }
  };

  const renderActivePage = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard setActiveTab={setActiveTab} modelInfo={modelInfo} setModelInfo={setModelInfo} />;
      case 'detect':
        return <DetectNews modelInfo={modelInfo} />;
      case 'history':
        return <PredictionHistory />;
      case 'performance':
        return <ModelPerformance modelInfo={modelInfo} />;
      case 'continuous':
        return <ContinuousLearning modelInfo={modelInfo} setModelInfo={setModelInfo} />;
      case 'monitoring':
        return <MLOpsMonitoring modelInfo={modelInfo} />;
      case 'settings':
        return <Settings modelInfo={modelInfo} />;
      default:
        return <Dashboard setActiveTab={setActiveTab} modelInfo={modelInfo} setModelInfo={setModelInfo} />;
    }
  };

  return (
    <div className="app-container">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        modelInfo={modelInfo}
      />
      <main className="main-content">
        <Navbar
          activeTab={activeTab}
          modelInfo={modelInfo}
          onNavigate={setActiveTab}
        />
        {renderActivePage()}
      </main>
    </div>
  );
}
