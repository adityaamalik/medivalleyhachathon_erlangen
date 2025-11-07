# Backend Setup Instructions

## Setup Summary

The backend has been successfully configured to run with **Python 3.11** in an isolated virtual environment.

## What Was Done

1. **Removed old Python 3.10 virtual environment** - Cleaned up the previous `.venv` directory
2. **Created new Python 3.11 virtual environment** - Using Homebrew's Python 3.11 installation
3. **Installed all dependencies** - All packages from `requirements.txt` installed successfully
4. **Created `.env` file** - Environment configuration file for API keys
5. **Tested backend** - Server starts successfully and responds to health checks

## Current Setup

- **Python Version**: Python 3.11.14 (from Homebrew)
- **Virtual Environment**: `.venv/` (using Python 3.11)
- **Server URL**: http://localhost:8000
- **Health Check**: http://localhost:8000/health

## How to Run the Backend

### Option 1: Using the convenience script (Recommended)

```bash
cd backend
./start_backend.sh
```

### Option 2: Manual activation

```bash
cd backend
source .venv/bin/activate  # Activate the virtual environment
python main.py             # Start the server
```

### Option 3: Direct execution

```bash
cd backend
.venv/bin/python main.py
```

### Option 4: With auto-reload for development

```bash
cd backend
.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Verify It's Working

After starting the server, open a new terminal and run:

```bash
curl http://localhost:8000/health
```

Expected response:
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

## Environment Variables

The `.env` file contains:

```bash
# External rPPG API Configuration
RPPG_API_KEY=your_api_key_here

# Model Configuration
MODEL_PATH=cnn_lstm_arrhythmia_detector.h5
```

**Note**: Update `RPPG_API_KEY` with your actual API key if you have one. The server will start without it, but WebSocket connections to the external rPPG API will require a valid key.

## Python Versions on System

- **Python 3.11.14**: `/opt/homebrew/bin/python3.11` (Homebrew installation) ✅ Used
- **Python 3.10.7**: `/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10`

## Installed Dependencies

All packages from `requirements.txt` have been installed, including:
- FastAPI & Uvicorn (Web framework & ASGI server)
- TensorFlow & Keras (Deep learning)
- Scikit-learn, XGBoost, LightGBM, CatBoost (ML models)
- NumPy, Pandas, SciPy (Data processing)
- Matplotlib, Seaborn, Plotly (Visualization)
- And many more...

## Troubleshooting

### Virtual environment not found
```bash
cd backend
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Port already in use
```bash
# Use a different port
.venv/bin/uvicorn main:app --port 8001
```

### Permission denied for start_backend.sh
```bash
chmod +x start_backend.sh
```

## Next Steps

1. ✅ Backend is ready to run independently
2. Update `RPPG_API_KEY` in `.env` file with your actual API key
3. Connect the frontend to the backend (see `QUICKSTART.md`)
4. Test the full application workflow

## Files Created/Modified

- `.venv/` - New virtual environment with Python 3.11
- `.env` - Environment configuration file
- `start_backend.sh` - Convenience script to start the backend
- `SETUP_INSTRUCTIONS.md` - This file
