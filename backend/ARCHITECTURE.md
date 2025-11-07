# Architecture Documentation

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                            │
│                      caire-dashboard/src/App.js                     │
│                                                                     │
│  • Captures video via webcam                                       │
│  • Sends frames via WebSocket                                      │
│  • Receives rPPG + arrhythmia data                                 │
│  • Displays heart rate, waveform, and arrhythmia status            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket
                                    │ ws://localhost:8000/ws/
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND (Python)                         │
│                        backend/main.py                              │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │              WebSocket Proxy Handler                          │ │
│  │  • Accepts client connections                                 │ │
│  │  • Forwards video frames to external API                      │ │
│  │  • Receives rPPG signals from external API                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                    │                                │
│                                    ▼                                │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │              Signal Buffer                                    │ │
│  │  • Accumulates rPPG samples                                   │ │
│  │  • Sliding window (max 2000 samples)                          │ │
│  │  • Triggers prediction at 1000 samples                        │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                    │                                │
│                                    ▼                                │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │              PPG Preprocessor                                 │ │
│  │  • Bandpass filter (0.5-8.0 Hz)                               │ │
│  │  • Z-score normalization                                      │ │
│  │  • Resampling to 1000 samples                                 │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                    │                                │
│                                    ▼                                │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │         Arrhythmia Detector (CNN-LSTM Model)                  │ │
│  │  • Model: cnn_lstm_arrhythmia_detector.h5                     │ │
│  │  • Input: (1, 1000, 1) - preprocessed PPG signal              │ │
│  │  • Output: probability [0-1]                                  │ │
│  │  • Threshold: 0.5 (>0.5 = arrhythmic)                         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                    │                                │
│                                    ▼                                │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │              Response Enhancer                                │ │
│  │  • Adds arrhythmia prediction to response                     │ │
│  │  • Includes confidence scores                                 │ │
│  │  • Sends enhanced data back to frontend                       │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket
                                    │ ws://3.67.186.245:8003/ws/
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EXTERNAL rPPG API                                │
│                   (3.67.186.245:8003)                               │
│                                                                     │
│  • Receives video frames                                           │
│  • Extracts rPPG signal from facial video                          │
│  • Calculates heart rate                                           │
│  • Returns rPPG waveform + HR                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Video Frame Transmission

```
Frontend → FastAPI → External API
```

**Payload:**
```json
{
  "datapt_id": "uuid",
  "state": "stream",
  "frame_data": "base64_jpeg",
  "timestamp": "1234567890.123",
  "advanced": true
}
```

### 2. rPPG Signal Reception

```
External API → FastAPI → Signal Buffer
```

**Response:**
```json
{
  "inference": {
    "hr": 75
  },
  "advanced": {
    "rppg": [0.1, 0.2, 0.15, ...]
  }
}
```

### 3. Arrhythmia Detection

```
Signal Buffer → Preprocessor → Model → Prediction
```

**Processing:**
1. Buffer accumulates rPPG samples
2. When 1000 samples collected:
   - Extract latest 1000 samples
   - Apply bandpass filter (0.5-8.0 Hz)
   - Normalize (z-score)
   - Reshape to (1, 1000, 1)
   - Run model inference
   - Generate prediction

### 4. Enhanced Response

```
FastAPI → Frontend
```

**Enhanced Payload:**
```json
{
  "inference": {
    "hr": 75
  },
  "advanced": {
    "rppg": [0.1, 0.2, ...]
  },
  "arrhythmia": {
    "is_arrhythmic": false,
    "probability": 0.23,
    "confidence": 0.77,
    "status": "Healthy",
    "samples_analyzed": 1000
  }
}
```

## Component Details

### WebSocket Proxy Handler

**Responsibilities:**
- Accept WebSocket connections from frontend
- Establish connection to external rPPG API
- Bidirectional message forwarding
- Connection lifecycle management

**Key Features:**
- Async/await for concurrent operations
- Error handling and reconnection logic
- Per-connection signal buffers

### Signal Buffer

**Configuration:**
- Max samples: 1000 (required for prediction)
- Buffer capacity: 2000 (allows overflow)
- Strategy: Sliding window (most recent samples)

**Operations:**
- `add_samples()`: Append new rPPG data
- `is_ready()`: Check if 1000 samples available
- `get_segment()`: Extract latest 1000 samples
- `clear()`: Reset buffer

### PPG Preprocessor

**Pipeline:**
1. **Bandpass Filter**
   - Type: Butterworth (4th order)
   - Frequency range: 0.5-8.0 Hz
   - Purpose: Remove noise outside physiological range

2. **Z-score Normalization**
   - Formula: `(x - mean) / std`
   - Purpose: Standardize signal amplitude

3. **Resampling** (if needed)
   - Method: Cubic interpolation
   - Target: 1000 samples
   - Purpose: Handle variable sampling rates

### Arrhythmia Detector

**Model Architecture:**
- Type: CNN-LSTM
- Input shape: (batch, 1000, 1)
- Output: Single probability [0-1]
- Activation: Sigmoid

**Prediction Logic:**
```python
if probability > 0.5:
    status = "Arrhythmic"
    confidence = probability
else:
    status = "Healthy"
    confidence = 1 - probability
```

## Timing & Performance

### Latency Breakdown

```
Video Frame → FastAPI:           ~5ms
FastAPI → External API:          ~10ms
External API Processing:         ~50-100ms
External API → FastAPI:          ~10ms
Signal Buffering:                ~1ms
Preprocessing (when ready):      ~5-10ms
Model Inference:                 ~10-50ms
Response → Frontend:             ~5ms
────────────────────────────────────────
Total (without prediction):      ~80-130ms
Total (with prediction):         ~95-190ms
```

### Buffer Accumulation

- **Sampling rate**: ~30 Hz (from video FPS)
- **Samples per second**: ~30
- **Time to 1000 samples**: ~33 seconds
- **Update frequency**: Every new frame (after first prediction)

## State Management

### Connection States

```
Frontend Connection:
  idle → connecting → connected → streaming → closed

External API Connection:
  idle → connecting → connected → streaming → closed

Signal Buffer:
  empty → collecting → ready → predicting → ready
```

### Error Handling

**Connection Errors:**
- Retry external API connection
- Send error message to frontend
- Clean up resources

**Prediction Errors:**
- Log error details
- Return error status to frontend
- Continue processing new data

## Security Considerations

### Current Implementation
- ✅ API key passed through to external service
- ✅ CORS enabled (all origins)
- ❌ No authentication on FastAPI endpoints
- ❌ No rate limiting
- ❌ No input validation

### Production Recommendations
- Add authentication middleware
- Implement rate limiting
- Validate input payloads
- Restrict CORS origins
- Add request logging
- Implement API key management

## Scalability

### Current Limitations
- Single-threaded model inference
- In-memory signal buffers
- No persistence layer
- No load balancing

### Scaling Options
1. **Horizontal Scaling**
   - Deploy multiple FastAPI instances
   - Use load balancer
   - Share model across workers

2. **Vertical Scaling**
   - GPU acceleration for model
   - Increase worker count
   - Optimize buffer management

3. **Caching**
   - Cache model predictions
   - Implement result TTL
   - Share predictions across connections

## Monitoring & Observability

### Logging
- Connection events
- Buffer status
- Prediction results
- Error conditions

### Metrics (Future)
- Request rate
- Prediction latency
- Buffer fill rate
- Model accuracy
- Error rate

### Health Checks
- `/health`: Detailed system status
- `/`: Basic health check
- Model load status
- External API connectivity

## Configuration

### Environment Variables (Recommended)

```bash
# Server
PORT=8000
HOST=0.0.0.0
WORKERS=4

# External API
EXTERNAL_API_URL=ws://3.67.186.245:8003/ws/
EXTERNAL_API_KEY=your_key_here

# Model
MODEL_PATH=cnn_lstm_arrhythmia_detector.h5
PREDICTION_THRESHOLD=0.5

# Signal Processing
REQUIRED_SAMPLES=1000
SAMPLING_RATE=100
LOWCUT_FREQ=0.5
HIGHCUT_FREQ=8.0
```

### Current Configuration (Hardcoded)

See `main.py` constants:
```python
EXTERNAL_API_BASE = "ws://3.67.186.245:8003/ws/"
MODEL_PATH = "cnn_lstm_arrhythmia_detector.h5"
REQUIRED_SAMPLES = 1000
TARGET_SAMPLING_RATE = 100
```

## Testing Strategy

### Unit Tests
- `test_preprocessor()`: Signal preprocessing
- `test_signal_buffer()`: Buffer operations
- `test_detector()`: Model inference
- `test_resampling()`: Signal resampling

### Integration Tests (Future)
- End-to-end WebSocket flow
- External API mocking
- Concurrent connections
- Error scenarios

### Load Tests (Future)
- Multiple simultaneous clients
- High-frequency frame rates
- Long-running connections
- Memory leak detection
