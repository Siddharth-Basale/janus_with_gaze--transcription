# ML Concentration Tracking Setup Guide

This guide explains how to set up and run the real-time concentration tracking system for your Janus meeting application.

## Architecture Overview

The system consists of three main components:

1. **Frontend (React/TypeScript)**: Extracts frames from video streams and sends them for processing
2. **Backend (Node.js)**: WebSocket server that forwards frames to Python ML service
3. **ML Service (Python/FastAPI)**: Processes frames using MediaPipe and returns concentration metrics

## Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- Janus Gateway running (Docker or native)
- Camera and microphone access

## Installation Steps

### 1. Install Python ML Service Dependencies

```bash
cd py-backend
pip install -r requirements.txt
```

### 2. Install Node.js Backend Dependencies

```bash
cd backend
npm install
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Configuration

### Python ML Service (`py-backend/config.py`)

Key configurable parameters:

- `FRAME_PROCESSING_FPS`: Frames per second to process (default: 8)
- `MAX_FRAME_WIDTH`: Maximum frame width (default: 640)
- `MAX_FRAME_HEIGHT`: Maximum frame height (default: 480)
- `JPEG_QUALITY`: JPEG compression quality (default: 70)
- `ML_SERVICE_PORT`: Port for ML service (default: 8000)

### Frontend (`frontend/src/config/mlConfig.ts`)

Key configurable parameters:

- `frameRate`: Frames per second to extract (default: 8)
- `jpegQuality`: JPEG compression quality (default: 70)
- `maxWidth`: Maximum frame width (default: 640)
- `maxHeight`: Maximum frame height (default: 480)
- `enableLocalProcessing`: Process local video (default: true)
- `enableRemoteProcessing`: Process remote videos (default: true)

## Running the Application

### 1. Start Python ML Service

```bash
cd py-backend
python ml_service.py
```

The service will start on `http://localhost:8000`

### 2. Start Node.js Backend

```bash
cd backend
npm run dev
```

The backend will start on `http://localhost:4000` with WebSocket at `ws://localhost:4000/ws/ml`

### 3. Start Frontend

```bash
cd frontend
npm run dev
```

The frontend will typically start on `http://localhost:5173`

## Usage

1. Open the frontend in your browser
2. Join a meeting room
3. Grant camera/microphone permissions
4. The system will automatically:
   - Extract frames from your video stream
   - Send them to the ML service for processing
   - Display concentration scores and status on video tiles

### Adjusting Settings in Real-Time

1. In the meeting room, open the "ML Settings" panel in the participants sidebar
2. Adjust parameters:
   - **Frame Rate**: Lower = less bandwidth, higher = more responsive (5-15 recommended)
   - **JPEG Quality**: Lower = smaller files, higher = better quality (50-90 recommended)
   - **Max Width/Height**: Smaller = faster processing, larger = better accuracy
3. Changes take effect immediately

## Performance Tuning

### For Lower Latency:
- Increase `frameRate` to 10-12 FPS
- Decrease `maxWidth`/`maxHeight` to 480x360
- Decrease `jpegQuality` to 60

### For Better Accuracy:
- Decrease `frameRate` to 5-6 FPS (less load)
- Increase `maxWidth`/`maxHeight` to 1280x720
- Increase `jpegQuality` to 80-90

### For Multiple Participants:
- Process only local video (`enableRemoteProcessing: false`)
- Lower `frameRate` to 5-6 FPS
- Use smaller frame dimensions

## Troubleshooting

### ML Service Not Responding
- Check if Python service is running: `curl http://localhost:8000/health`
- Check Python dependencies: `pip list | grep -E "fastapi|opencv|mediapipe"`

### WebSocket Connection Failed
- Verify backend is running on port 4000
- Check browser console for WebSocket errors
- Ensure CORS is properly configured

### No Concentration Results
- Check browser console for errors
- Verify video stream is active
- Check ML service logs for processing errors
- Ensure camera permissions are granted

### High CPU Usage
- Lower `frameRate` in config
- Reduce `maxWidth`/`maxHeight`
- Disable remote video processing if not needed

## Expected Latency

With default settings:
- Frame extraction: ~5-10ms
- Network transfer: ~20-50ms
- ML processing: ~50-100ms
- **Total: ~75-160ms** (acceptable for real-time)

## API Endpoints

### ML Service

- `POST /process-frame`: Process a single frame
  ```json
  {
    "session_id": "string",
    "frame_data": "base64_encoded_jpeg",
    "timestamp": 1234567890
  }
  ```

- `GET /health`: Health check
- `POST /reset-session/{session_id}`: Reset session calibration

### Backend

- `GET /health`: Health check
- `GET /api/config`: Get Janus configuration
- `WS /ws/ml`: WebSocket endpoint for frame processing

## Notes

- The system requires a 3-second calibration period per session
- Audio processing is disabled by default (set `ENABLE_AUDIO = True` in config.py to enable)
- Each video stream is processed independently
- Results are cached and smoothed over multiple frames for stability

