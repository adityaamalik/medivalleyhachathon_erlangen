#!/bin/bash

# Start Backend Server Script
# This script activates the virtual environment and starts the FastAPI backend

echo "🚀 Starting CAIRE Backend Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3.11 -m venv .venv"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Creating default .env file..."
    cat > .env << EOF
# External rPPG API Configuration
RPPG_API_KEY=your_api_key_here

# Model Configuration
MODEL_PATH=cnn_lstm_arrhythmia_detector.h5
EOF
    echo "✅ Created .env file. Please update RPPG_API_KEY with your actual API key."
fi

# Activate virtual environment and start server
echo "📦 Using Python 3.11 virtual environment..."
echo "🌐 Server will start at: http://localhost:8000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the server
.venv/bin/python main.py
