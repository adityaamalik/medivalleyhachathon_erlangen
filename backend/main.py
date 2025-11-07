"""
FastAPI Backend for Real-time Arrhythmia Detection
Proxies video frames to external rPPG API and performs arrhythmia detection
"""

import asyncio
import json
import logging
from typing import Optional, Dict, List
from collections import deque
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import websockets
import tensorflow as tf
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Arrhythmia Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# External rPPG API configuration
EXTERNAL_API_BASE = "ws://3.67.186.245:8003/ws/"

# Model configuration
MODEL_PATH = "cnn_lstm_arrhythmia_detector.h5"
REQUIRED_SAMPLES = 1000  # 10 seconds at 100 Hz
TARGET_SAMPLING_RATE = 100  # Hz


class PPGPreprocessor:
    """Minimal preprocessing for PPG signals"""
    
    def __init__(self, sampling_rate=100, lowcut=0.5, highcut=8.0):
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        
    def bandpass_filter(self, ppg_signal):
        """Apply bandpass filter to remove noise outside physiological range"""
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Design Butterworth bandpass filter
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered = scipy_signal.filtfilt(b, a, ppg_signal)
        
        return filtered
    
    def normalize(self, ppg_signal):
        """Z-score normalization"""
        mean = np.mean(ppg_signal)
        std = np.std(ppg_signal)
        
        if std == 0:
            return ppg_signal - mean
        
        return (ppg_signal - mean) / std
    
    def preprocess(self, ppg_signal):
        """Complete preprocessing pipeline"""
        # Apply bandpass filter
        filtered = self.bandpass_filter(ppg_signal)
        
        # Normalize
        normalized = self.normalize(filtered)
        
        return normalized


class ArrhythmiaDetector:
    """Handles arrhythmia detection using trained CNN-LSTM model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.preprocessor = PPGPreprocessor()
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ Model loaded successfully from {self.model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def resample_signal(self, signal_data: List[float], target_length: int = REQUIRED_SAMPLES) -> np.ndarray:
        """Resample signal to target length using interpolation"""
        if len(signal_data) == target_length:
            return np.array(signal_data)
        
        # Create interpolation function
        x_old = np.linspace(0, 1, len(signal_data))
        x_new = np.linspace(0, 1, target_length)
        
        f = interp1d(x_old, signal_data, kind='cubic', fill_value='extrapolate')
        resampled = f(x_new)
        
        return resampled
    
    def predict(self, rppg_signal: List[float]) -> Dict:
        """
        Predict arrhythmia from rPPG signal
        
        Args:
            rppg_signal: List of rPPG values
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Convert to numpy array
            signal_array = np.array(rppg_signal)
            
            # Resample to required length if needed
            if len(signal_array) != REQUIRED_SAMPLES:
                logger.info(f"Resampling signal from {len(signal_array)} to {REQUIRED_SAMPLES} samples")
                signal_array = self.resample_signal(signal_array, REQUIRED_SAMPLES)
            
            # Preprocess the signal
            preprocessed = self.preprocessor.preprocess(signal_array)
            
            # Reshape for model input: (batch_size, sequence_length, features)
            model_input = preprocessed.reshape(1, REQUIRED_SAMPLES, 1)
            
            # Run inference
            prediction = self.model.predict(model_input, verbose=0)
            probability = float(prediction[0][0])
            
            # Binary classification: threshold at 0.5
            is_arrhythmic = probability > 0.5
            confidence = probability if is_arrhythmic else (1 - probability)
            
            result = {
                "is_arrhythmic": bool(is_arrhythmic),
                "probability": probability,
                "confidence": float(confidence),
                "status": "Arrhythmic" if is_arrhythmic else "Healthy",
                "samples_analyzed": len(signal_array)
            }
            
            logger.info(f"🔍 Prediction: {result['status']} (confidence: {confidence:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return {
                "error": str(e),
                "is_arrhythmic": None,
                "status": "Error"
            }


class SignalBuffer:
    """Buffers rPPG signals for arrhythmia detection"""
    
    def __init__(self, max_samples: int = REQUIRED_SAMPLES):
        self.max_samples = max_samples
        self.buffer = deque(maxlen=max_samples * 2)  # Allow some overflow
        self.last_prediction = None
        
    def add_samples(self, samples: List[float]):
        """Add new samples to buffer"""
        self.buffer.extend(samples)
        
    def is_ready(self) -> bool:
        """Check if buffer has enough samples for prediction"""
        return len(self.buffer) >= self.max_samples
    
    def get_segment(self) -> List[float]:
        """Get the latest segment for prediction"""
        if not self.is_ready():
            return None
        
        # Return the most recent max_samples
        return list(self.buffer)[-self.max_samples:]
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.last_prediction = None


# Global detector instance
detector = ArrhythmiaDetector(MODEL_PATH)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Arrhythmia Detection API starting up...")
    logger.info(f"📊 Model loaded and ready")
    logger.info(f"🔗 External rPPG API: {EXTERNAL_API_BASE}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Arrhythmia Detection API",
        "model_loaded": detector.model is not None,
        "required_samples": REQUIRED_SAMPLES
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model": {
            "loaded": detector.model is not None,
            "path": MODEL_PATH,
            "input_shape": str(detector.model.input_shape) if detector.model else None
        },
        "config": {
            "required_samples": REQUIRED_SAMPLES,
            "sampling_rate": TARGET_SAMPLING_RATE,
            "external_api": EXTERNAL_API_BASE
        }
    }


@app.websocket("/ws/")
async def websocket_endpoint(
    websocket: WebSocket,
    api_key: str = Query(..., description="API key for external rPPG service")
):
    """
    WebSocket endpoint that proxies to external rPPG API and adds arrhythmia detection
    """
    await websocket.accept()
    logger.info(f"✅ Client connected")
    
    # Initialize signal buffer for this connection
    signal_buffer = SignalBuffer()
    
    # Connect to external rPPG API
    external_ws_url = f"{EXTERNAL_API_BASE}?api_key={api_key}"
    external_ws = None
    
    try:
        # Establish connection to external API
        external_ws = await websockets.connect(external_ws_url)
        logger.info(f"✅ Connected to external rPPG API")
        
        # Create tasks for bidirectional communication
        async def forward_to_external():
            """Forward messages from client to external API"""
            try:
                while True:
                    # Receive from client
                    data = await websocket.receive_text()
                    
                    # Forward to external API
                    await external_ws.send(data)
                    
            except WebSocketDisconnect:
                logger.info("Client disconnected")
            except Exception as e:
                logger.error(f"Error forwarding to external: {e}")
        
        async def forward_from_external():
            """Forward messages from external API to client, with arrhythmia detection"""
            try:
                while True:
                    # Receive from external API
                    response = await external_ws.recv()
                    
                    try:
                        # Parse response
                        data = json.loads(response)
                        
                        # Check if response contains rPPG data
                        if "advanced" in data and "rppg" in data["advanced"]:
                            rppg_data = data["advanced"]["rppg"]
                            
                            if isinstance(rppg_data, list) and len(rppg_data) > 0:
                                # Add to buffer
                                signal_buffer.add_samples(rppg_data)
                                
                                # Check if we have enough data for prediction
                                if signal_buffer.is_ready():
                                    segment = signal_buffer.get_segment()
                                    
                                    # Run arrhythmia detection
                                    prediction = detector.predict(segment)
                                    
                                    # Add prediction to response
                                    data["arrhythmia"] = prediction
                                    signal_buffer.last_prediction = prediction
                                    
                                    logger.info(f"📊 Buffer: {len(signal_buffer.buffer)} samples, Status: {prediction['status']}")
                                else:
                                    # Include last prediction if available
                                    if signal_buffer.last_prediction:
                                        data["arrhythmia"] = signal_buffer.last_prediction
                                    else:
                                        data["arrhythmia"] = {
                                            "status": "Collecting data...",
                                            "samples_collected": len(signal_buffer.buffer),
                                            "samples_needed": REQUIRED_SAMPLES
                                        }
                        
                        # Send enhanced response to client
                        await websocket.send_json(data)
                        
                    except json.JSONDecodeError:
                        # If not JSON, forward as-is
                        await websocket.send_text(response)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info("External API connection closed")
            except Exception as e:
                logger.error(f"Error forwarding from external: {e}")
        
        # Run both tasks concurrently
        await asyncio.gather(
            forward_to_external(),
            forward_from_external()
        )
        
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        await websocket.send_json({
            "error": str(e),
            "message": "Failed to connect to external rPPG API"
        })
    finally:
        # Cleanup
        if external_ws:
            await external_ws.close()
        signal_buffer.clear()
        logger.info("🔌 Connection closed, buffer cleared")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
