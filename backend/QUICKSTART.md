# Quick Start Guide - Arrhythmia Detection API

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Test the Setup

```bash
python test_api.py
```

Expected output:
```
✅ ALL TESTS PASSED!
```

### Step 3: Start the Server

```bash
python main.py
```

Or with auto-reload for development:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: **http://localhost:8000**

---

## ✅ Verify It's Working

### Check Health Status

Open browser or use curl:
```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "model": {
    "loaded": true,
    "path": "cnn_lstm_arrhythmia_detector.h5"
  }
}
```

---

## 🔌 Connect Frontend (When Ready)

In `caire-dashboard/src/App.js`, change line 86:

```javascript
// OLD
const WS_BASE = "ws://3.67.186.245:8003/ws/";

// NEW  
const WS_BASE = "ws://localhost:8000/ws/";
```

That's it! The API will automatically:
1. ✅ Proxy video frames to external rPPG API
2. ✅ Receive rPPG signals
3. ✅ Buffer and preprocess signals
4. ✅ Run arrhythmia detection
5. ✅ Send enhanced responses with predictions

---

## 📊 What You'll Get

The API adds an `arrhythmia` field to every response:

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

---

## 🐛 Troubleshooting

### Model Not Found
```bash
ls -lh cnn_lstm_arrhythmia_detector.h5
```
Make sure the model file exists in the backend directory.

### Port Already in Use
```bash
# Use a different port
uvicorn main:app --port 8001
```

### Dependencies Issues
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📖 Full Documentation

See `API_README.md` for complete documentation.

---

## 🎯 Next Steps

1. ✅ Start the backend server
2. ⏳ Test with frontend (when ready to modify)
3. ⏳ Add arrhythmia status display to UI
4. ⏳ Deploy to production

**Note**: Frontend modifications are on hold per your request. The backend is ready and waiting!
