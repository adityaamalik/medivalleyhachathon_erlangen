import './App.css';
import { useState, useEffect } from 'react';

function App() {
  const [healthMetrics, setHealthMetrics] = useState({
    heartRate: 72,
    bloodPressure: { systolic: 120, diastolic: 80 },
    oxygenLevel: 98,
    respiratoryRate: 16,
    temperature: 98.6,
    stressLevel: 'Low'
  });

  // Simulate real-time updates (you'll replace this with actual rPPG data later)
  useEffect(() => {
    const interval = setInterval(() => {
      setHealthMetrics(prev => ({
        ...prev,
        heartRate: 70 + Math.floor(Math.random() * 10),
        oxygenLevel: 96 + Math.floor(Math.random() * 3)
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="App">
      <header className="dashboard-header">
        <h1>Caire - Driver Health Monitor</h1>
        <p className="subtitle">Real-time health monitoring system</p>
      </header>

      <div className="metrics-container">
        <div className="metric-card heart-rate">
          <div className="metric-icon">❤️</div>
          <div className="metric-content">
            <h3>Heart Rate</h3>
            <div className="metric-value">{healthMetrics.heartRate}</div>
            <div className="metric-unit">BPM</div>
          </div>
          <div className={`status-indicator ${healthMetrics.heartRate > 60 && healthMetrics.heartRate < 100 ? 'normal' : 'warning'}`}></div>
        </div>

        <div className="metric-card blood-pressure">
          <div className="metric-icon">🩸</div>
          <div className="metric-content">
            <h3>Blood Pressure</h3>
            <div className="metric-value">
              {healthMetrics.bloodPressure.systolic}/{healthMetrics.bloodPressure.diastolic}
            </div>
            <div className="metric-unit">mmHg</div>
          </div>
          <div className="status-indicator normal"></div>
        </div>

        <div className="metric-card oxygen">
          <div className="metric-icon">🫁</div>
          <div className="metric-content">
            <h3>Oxygen Level</h3>
            <div className="metric-value">{healthMetrics.oxygenLevel}</div>
            <div className="metric-unit">%</div>
          </div>
          <div className={`status-indicator ${healthMetrics.oxygenLevel >= 95 ? 'normal' : 'warning'}`}></div>
        </div>

        <div className="metric-card respiratory">
          <div className="metric-icon">💨</div>
          <div className="metric-content">
            <h3>Respiratory Rate</h3>
            <div className="metric-value">{healthMetrics.respiratoryRate}</div>
            <div className="metric-unit">breaths/min</div>
          </div>
          <div className="status-indicator normal"></div>
        </div>

        <div className="metric-card temperature">
          <div className="metric-icon">🌡️</div>
          <div className="metric-content">
            <h3>Temperature</h3>
            <div className="metric-value">{healthMetrics.temperature}</div>
            <div className="metric-unit">°F</div>
          </div>
          <div className="status-indicator normal"></div>
        </div>

        <div className="metric-card stress">
          <div className="metric-icon">🧠</div>
          <div className="metric-content">
            <h3>Stress Level</h3>
            <div className="metric-value-text">{healthMetrics.stressLevel}</div>
          </div>
          <div className="status-indicator normal"></div>
        </div>
      </div>

      <div className="camera-feed-placeholder">
        <h3>Camera Feed</h3>
        <div className="feed-box">
          <p>Video feed will be integrated here</p>
        </div>
      </div>
    </div>
  );
}

export default App;
