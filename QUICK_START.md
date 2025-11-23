# Quick Start Guide

## 🚀 Quick Setup (3 Steps)

### 1. Install Dependencies

```bash
# Python ML Service
cd py-backend
pip install -r requirements.txt

# Node.js Backend
cd ../backend
npm install

# Frontend
cd ../frontend
npm install
```

### 2. Start All Services

**Terminal 1 - Python ML Service:**
```bash
cd py-backend
python ml_service.py
```
✅ Should see: `Application startup complete` on `http://0.0.0.0:8000`

**Terminal 2 - Node.js Backend:**
```bash
cd backend
npm run dev
```
✅ Should see: `Backend listening on http://localhost:4000`

**Terminal 3 - Frontend:**
```bash
cd frontend
npm run dev
```
✅ Should see: `Local: http://localhost:5173`

### 3. Test It!

1. Open `http://localhost:5173` in your browser
2. Join a meeting room
3. Grant camera permissions
4. **Watch the concentration score appear on your video!** 🎯

## ⚙️ Adjust Settings in Real-Time

While in a meeting:

1. Look at the **"ML Settings"** panel in the right sidebar
2. Click to expand it
3. Adjust:
   - **Frame Rate**: 5-15 FPS (lower = less bandwidth)
   - **JPEG Quality**: 50-90 (lower = faster)
   - **Max Width/Height**: Smaller = faster processing

## 🎯 Expected Performance

- **Latency**: ~100-200ms (very responsive!)
- **CPU Usage**: Moderate (depends on frame rate)
- **Bandwidth**: ~50-200 KB/s per video stream

## 🔧 Troubleshooting

**No concentration scores showing?**
- Check browser console (F12) for errors
- Verify Python service is running: `curl http://localhost:8000/health`
- Ensure camera permissions are granted

**High latency?**
- Lower frame rate to 5-6 FPS
- Reduce max width/height to 480x360
- Lower JPEG quality to 60

**Service won't start?**
- Check if ports 4000 and 8000 are available
- Verify all dependencies are installed
- Check Python version: `python --version` (need 3.9+)

## 📊 Configuration Files

- **Python**: `py-backend/config.py` - ML processing settings
- **Frontend**: `frontend/src/config/mlConfig.ts` - Frame extraction settings

Both can be adjusted while the app is running via the UI!

