# Arrhythmia Detection API

FastAPI backend that proxies video frames to external rPPG API and performs real-time arrhythmia detection.

## Architecture

```
Frontend → FastAPI Backend → External rPPG API
              ↓
       Arrhythmia Model
              ↓
       Enhanced Response
```

## Features

- **WebSocket Proxy**: Forwards video frames to external rPPG API
- **Signal Buffering**: Accumulates rPPG samples for analysis
- **Real-time Detection**: Runs arrhythmia detection when sufficient data is collected
- **Enhanced Responses**: Adds arrhythmia predictions to API responses

## Installation

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Verify Model File

Ensure the trained model exists:
```bash
ls -lh cnn_lstm_arrhythmia_detector.h5
```

## Running the Server

### Development Mode

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check

```bash
GET http://localhost:8000/
```

Response:
```json
{
  "status": "running",
  "service": "Arrhythmia Detection API",
  "model_loaded": true,
  "required_samples": 1000
}
```

### Detailed Health Check

```bash
GET http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model": {
    "loaded": true,
    "path": "cnn_lstm_arrhythmia_detector.h5",
    "input_shape": "(None, 1000, 1)"
  },
  "config": {
    "required_samples": 1000,
    "sampling_rate": 100,
    "external_api": "ws://3.67.186.245:8003/ws/"
  }
}
```

### WebSocket Connection

```
WS ws://localhost:8000/ws/?api_key=YOUR_API_KEY
```

**Parameters:**
- `api_key`: API key for external rPPG service

## WebSocket Protocol

### Client → Server (Video Frames)

Same format as external rPPG API:

```json
{
  "datapt_id": "uuid",
  "state": "stream",
  "frame_data": "base64_encoded_jpeg",
  "timestamp": "1234567890.123",
  "advanced": true
}
```

### Server → Client (Enhanced Response)

Original rPPG response + arrhythmia detection:

```json
{
  "inference": {
    "hr": 75
  },
  "advanced": {
    "rppg": [0.1, 0.2, 0.15, ...]
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

### Arrhythmia Response Fields

**While collecting data (< 1000 samples):**
```json
{
  "arrhythmia": {
    "status": "Collecting data...",
    "samples_collected": 450,
    "samples_needed": 1000
  }
}
```

**After first prediction:**
```json
{
  "arrhythmia": {
    "is_arrhythmic": true,
    "probability": 0.87,
    "confidence": 0.87,
    "status": "Arrhythmic",
    "samples_analyzed": 1000
  }
}
```

**Fields:**
- `is_arrhythmic`: Boolean indicating arrhythmia detection
- `probability`: Raw model output (0-1, where >0.5 = arrhythmic)
- `confidence`: Confidence score (distance from 0.5 threshold)
- `status`: Human-readable status ("Healthy", "Arrhythmic", "Collecting data...")
- `samples_analyzed`: Number of samples used for prediction

## Signal Processing

### Buffer Management
- Accumulates rPPG samples in a sliding window buffer
- Maintains up to 2000 samples (allows overflow)
- Uses most recent 1000 samples for prediction

### Preprocessing Pipeline
1. **Bandpass Filter**: 0.5-8.0 Hz (removes noise outside physiological range)
2. **Z-score Normalization**: Standardizes signal amplitude
3. **Resampling**: Interpolates to 1000 samples if needed

### Prediction Strategy
- **Sliding Window**: Continuously updates with latest 1000 samples
- **Real-time Updates**: New prediction every time buffer receives data
- **Last Prediction Cache**: Shows last result while collecting new data

## Configuration

Edit these constants in `main.py`:

```python
EXTERNAL_API_BASE = "ws://3.67.186.245:8003/ws/"  # External rPPG API
MODEL_PATH = "cnn_lstm_arrhythmia_detector.h5"    # Model file
REQUIRED_SAMPLES = 1000                            # Samples needed (10s @ 100Hz)
TARGET_SAMPLING_RATE = 100                         # Hz
```

## Frontend Integration (Future)

To connect frontend to this backend, change WebSocket URL in `App.js`:

```javascript
// OLD
const WS_BASE = "ws://3.67.186.245:8003/ws/";

// NEW
const WS_BASE = "ws://localhost:8000/ws/";
```

Then add arrhythmia status display:

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  // Existing heart rate update
  if (data.inference && data.inference.hr) {
    setHealthMetrics(prev => ({
      ...prev,
      heartRate: Math.round(data.inference.hr)
    }));
  }
  
  // NEW: Arrhythmia status update
  if (data.arrhythmia) {
    setArrhythmiaStatus(data.arrhythmia);
  }
};
```

## Testing

### Test Health Endpoint

```bash
curl http://localhost:8000/health
```

### Test WebSocket (with wscat)

```bash
npm install -g wscat
wscat -c "ws://localhost:8000/ws/?api_key=YOUR_API_KEY"
```

## Logging

The API logs important events:
- ✅ Model loading
- ✅ Client connections
- ✅ External API connections
- 📊 Buffer status and predictions
- ❌ Errors and exceptions

View logs in console output.

## Troubleshooting

### Model Loading Error

```
❌ Failed to load model: ...
```

**Solution**: Verify model file exists and is valid:
```bash
ls -lh cnn_lstm_arrhythmia_detector.h5
python -c "import tensorflow as tf; tf.keras.models.load_model('cnn_lstm_arrhythmia_detector.h5')"
```

### External API Connection Failed

```
❌ WebSocket error: ...
```

**Solution**: Check external API is accessible:
```bash
wscat -c "ws://3.67.186.245:8003/ws/?api_key=YOUR_API_KEY"
```

### No Arrhythmia Predictions

**Solution**: Ensure rPPG data is being received:
- Check logs for "Buffer: X samples"
- Verify `advanced: true` in client payload
- Wait for 1000 samples to accumulate

## Performance

- **Model Inference**: ~10-50ms per prediction
- **Preprocessing**: ~5-10ms per segment
- **WebSocket Latency**: Minimal overhead (<5ms)
- **Memory**: ~500MB (model + buffers)

## Security Notes

- API key is passed through to external service
- No authentication on FastAPI endpoints (add if needed)
- CORS enabled for all origins (restrict in production)

## Future Enhancements

- [ ] Add authentication/authorization
- [ ] Implement rate limiting
- [ ] Add prediction history/logging
- [ ] Support multiple concurrent clients
- [ ] Add metrics/monitoring endpoints
- [ ] Implement configurable thresholds
- [ ] Add model versioning support
