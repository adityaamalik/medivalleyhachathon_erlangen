"""
Test script for Arrhythmia Detection API
"""

import asyncio
import json
import numpy as np
from main import ArrhythmiaDetector, PPGPreprocessor, SignalBuffer

def test_preprocessor():
    """Test PPG preprocessor"""
    print("\n" + "="*60)
    print("Testing PPG Preprocessor")
    print("="*60)
    
    preprocessor = PPGPreprocessor()
    
    # Generate synthetic PPG signal
    t = np.linspace(0, 10, 1000)  # 10 seconds, 100 Hz
    signal = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz (72 BPM)
    signal += 0.1 * np.random.randn(1000)  # Add noise
    
    print(f"✓ Generated synthetic signal: {len(signal)} samples")
    print(f"  Mean: {np.mean(signal):.4f}, Std: {np.std(signal):.4f}")
    
    # Preprocess
    processed = preprocessor.preprocess(signal)
    
    print(f"✓ Preprocessed signal")
    print(f"  Mean: {np.mean(processed):.4f}, Std: {np.std(processed):.4f}")
    
    assert len(processed) == len(signal), "Signal length changed!"
    assert abs(np.mean(processed)) < 0.1, "Mean not close to zero!"
    assert abs(np.std(processed) - 1.0) < 0.1, "Std not close to 1.0!"
    
    print("✅ Preprocessor test passed!")


def test_signal_buffer():
    """Test signal buffer"""
    print("\n" + "="*60)
    print("Testing Signal Buffer")
    print("="*60)
    
    buffer = SignalBuffer(max_samples=1000)
    
    print(f"✓ Created buffer (max_samples=1000)")
    print(f"  Ready: {buffer.is_ready()}")
    
    # Add samples in chunks
    for i in range(10):
        samples = list(np.random.randn(100))
        buffer.add_samples(samples)
        print(f"  Added 100 samples, total: {len(buffer.buffer)}, ready: {buffer.is_ready()}")
    
    assert buffer.is_ready(), "Buffer should be ready!"
    
    # Get segment
    segment = buffer.get_segment()
    assert len(segment) == 1000, f"Segment should be 1000 samples, got {len(segment)}"
    
    print("✅ Signal buffer test passed!")


def test_detector():
    """Test arrhythmia detector"""
    print("\n" + "="*60)
    print("Testing Arrhythmia Detector")
    print("="*60)
    
    try:
        detector = ArrhythmiaDetector("cnn_lstm_arrhythmia_detector.h5")
        print("✓ Model loaded successfully")
        print(f"  Input shape: {detector.model.input_shape}")
        
        # Generate synthetic signal
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(1000)
        
        print(f"✓ Generated test signal: {len(signal)} samples")
        
        # Run prediction
        result = detector.predict(signal.tolist())
        
        print(f"✓ Prediction completed:")
        print(f"  Status: {result['status']}")
        print(f"  Is Arrhythmic: {result.get('is_arrhythmic', 'N/A')}")
        print(f"  Probability: {result.get('probability', 'N/A'):.4f}")
        print(f"  Confidence: {result.get('confidence', 'N/A'):.4f}")
        print(f"  Samples Analyzed: {result.get('samples_analyzed', 'N/A')}")
        
        assert 'status' in result, "Result should contain status!"
        assert result.get('samples_analyzed') == 1000, "Should analyze 1000 samples!"
        
        print("✅ Detector test passed!")
        
    except Exception as e:
        print(f"❌ Detector test failed: {e}")
        raise


def test_resampling():
    """Test signal resampling"""
    print("\n" + "="*60)
    print("Testing Signal Resampling")
    print("="*60)
    
    detector = ArrhythmiaDetector("cnn_lstm_arrhythmia_detector.h5")
    
    # Test different input lengths
    test_lengths = [500, 800, 1000, 1200, 1500]
    
    for length in test_lengths:
        signal = list(np.random.randn(length))
        resampled = detector.resample_signal(signal, target_length=1000)
        
        print(f"✓ Resampled {length} → {len(resampled)} samples")
        assert len(resampled) == 1000, f"Expected 1000 samples, got {len(resampled)}"
    
    print("✅ Resampling test passed!")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ARRHYTHMIA DETECTION API - TEST SUITE")
    print("="*60)
    
    try:
        test_preprocessor()
        test_signal_buffer()
        test_resampling()
        test_detector()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now start the API server:")
        print("  python main.py")
        print("\nOr with uvicorn:")
        print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ TESTS FAILED: {e}")
        print("="*60)
        raise


if __name__ == "__main__":
    main()
