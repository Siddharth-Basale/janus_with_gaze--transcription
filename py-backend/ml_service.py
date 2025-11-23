"""
FastAPI service for real-time concentration tracking from video frames
Adapted from ml.py to process frames from HTTP/WebSocket requests
"""
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import time
from collections import deque, Counter
import threading
import config

app = FastAPI(title="ML Concentration Tracker Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    static_image_mode=False
)
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=config.FACE_DET_CONF
)

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = 468
RIGHT_IRIS = 473
# Eye corners for relative iris position calculation (eyeball movement detection)
LEFT_EYE_OUTER = 33   # Left eye outer corner
LEFT_EYE_INNER = 133  # Left eye inner corner
RIGHT_EYE_OUTER = 362 # Right eye outer corner
RIGHT_EYE_INNER = 263 # Right eye inner corner

# Per-session state (session_id -> state)
session_states: Dict[str, Dict] = {}
state_lock = threading.Lock()


def get_session_state(session_id: str) -> Dict:
    """Get or create session state"""
    with state_lock:
        if session_id not in session_states:
            session_states[session_id] = {
                'baseline_x': None,
                'baseline_y': None,
                'baseline_rel_x': None,  # Relative iris position baseline
                'baseline_rel_y': None,
                'calib_x': [],
                'calib_y': [],
                'calib_rel_x': [],
                'calib_rel_y': [],
                'calib_start': None,
                'calibrated': False,
                'score_buf': deque(maxlen=config.SCORE_SMOOTH),
        'blink_frames': 0,
        'last_blink_time': 0.0,
        'eyes_closed_start': None,
        'no_face_start': None,
        'frame_count': 0,
        'gaze_dir_buf': deque(maxlen=config.GAZE_SMOOTH_WINDOW if hasattr(config, 'GAZE_SMOOTH_WINDOW') else 5),
    }
        return session_states[session_id]


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """Calculate Eye Aspect Ratio"""
    try:
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        if C <= 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)
    except Exception:
        return 0.0


def eye_region_stats(gray, landmarks, eye_indices, w, h, pad=6):
    """Get eye region statistics for occlusion detection"""
    try:
        xs = [int(landmarks[i].x * w) for i in eye_indices]
        ys = [int(landmarks[i].y * h) for i in eye_indices]
        x1 = max(min(xs) - pad, 0)
        x2 = min(max(xs) + pad, w - 1)
        y1 = max(min(ys) - pad, 0)
        y2 = min(max(ys) + pad, h - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            return None
        return float(np.mean(region)), float(np.var(region))
    except Exception:
        return None


def get_iris_avg(landmarks):
    """Get average iris position (absolute coordinates)"""
    try:
        return (landmarks[LEFT_IRIS].x + landmarks[RIGHT_IRIS].x) / 2.0, \
               (landmarks[LEFT_IRIS].y + landmarks[RIGHT_IRIS].y) / 2.0
    except Exception:
        return None, None


def get_iris_relative_position(landmarks):
    """
    Calculate iris position relative to eye corners - detects actual EYEBALL movement
    independent of head position. Returns normalized position (0-1) where 0.5 is center.
    """
    try:
        # Left eye: iris position relative to corners
        left_outer_x = landmarks[LEFT_EYE_OUTER].x
        left_inner_x = landmarks[LEFT_EYE_INNER].x
        left_iris_x = landmarks[LEFT_IRIS].x
        
        # Right eye: iris position relative to corners
        right_outer_x = landmarks[RIGHT_EYE_OUTER].x
        right_inner_x = landmarks[RIGHT_EYE_INNER].x
        right_iris_x = landmarks[RIGHT_IRIS].x
        
        # Calculate normalized horizontal position (0 = outer corner, 1 = inner corner, 0.5 = center)
        # For left eye: outer is smaller x, inner is larger x
        left_eye_width = abs(left_inner_x - left_outer_x)
        if left_eye_width > 1e-6:
            left_rel_x = (left_iris_x - left_outer_x) / left_eye_width
        else:
            left_rel_x = 0.5
        
        # For right eye: outer is larger x, inner is smaller x (mirrored)
        right_eye_width = abs(right_outer_x - right_inner_x)
        if right_eye_width > 1e-6:
            right_rel_x = (right_outer_x - right_iris_x) / right_eye_width
        else:
            right_rel_x = 0.5
        
        # Average of both eyes
        avg_rel_x = (left_rel_x + right_rel_x) / 2.0
        
        # Vertical position: use eye top and bottom landmarks
        # Left eye top/bottom: 159 (top), 145 (bottom)
        # Right eye top/bottom: 386 (top), 374 (bottom)
        try:
            left_top_y = landmarks[159].y
            left_bottom_y = landmarks[145].y
            left_iris_y = landmarks[LEFT_IRIS].y
            left_eye_height = abs(left_bottom_y - left_top_y)
            if left_eye_height > 1e-6:
                left_rel_y = (left_iris_y - left_top_y) / left_eye_height
            else:
                left_rel_y = 0.5
        except:
            left_rel_y = 0.5
        
        try:
            right_top_y = landmarks[386].y
            right_bottom_y = landmarks[374].y
            right_iris_y = landmarks[RIGHT_IRIS].y
            right_eye_height = abs(right_bottom_y - right_top_y)
            if right_eye_height > 1e-6:
                right_rel_y = (right_iris_y - right_top_y) / right_eye_height
            else:
                right_rel_y = 0.5
        except:
            right_rel_y = 0.5
        
        avg_rel_y = (left_rel_y + right_rel_y) / 2.0
        
        return avg_rel_x, avg_rel_y
    except Exception:
        return None, None


def compute_gaze_on_screen_percentage(gaze_distance, max_deviation=0.2):
    """
    Compute percentage of how much the gaze is on-screen (0-100%)
    - gaze_distance: distance from calibrated center (0.0 = perfectly centered)
    - max_deviation: maximum expected deviation (normalized, ~0.2 for relative position)
    Returns: 0-100% where 100% = perfectly on screen, 0% = completely off screen
    """
    if gaze_distance is None or gaze_distance < 0:
        return 0
    
    # Normalize distance (0.0 to 1.0)
    normalized_dist = min(gaze_distance / max_deviation, 1.0)
    
    # Convert to percentage: 100% when centered (distance=0), 0% when at max deviation
    # Use smooth curve: 100% * (1 - normalized_dist)
    percentage = 100.0 * (1.0 - normalized_dist)
    
    return int(np.clip(percentage, 0, 100))


class FrameRequest(BaseModel):
    session_id: str
    frame_data: str  # Base64 encoded JPEG
    timestamp: Optional[float] = None


class FrameResponse(BaseModel):
    gaze_on_screen_percentage: int  # 0-100% of how much gaze is on screen
    status: str  # FOCUSED, PARTIAL, DISTRACTED, OFF_SCREEN, NO_FACE
    gaze_direction: str  # CENTER, LEFT, RIGHT, UP, DOWN, UNKNOWN, NO_FACE
    calibrated: bool
    smooth_score: int  # Same as gaze_on_screen_percentage (for compatibility)


def process_frame(frame_data: bytes, session_id: str) -> Dict:
    """Process a single frame and return results"""
    state = get_session_state(session_id)
    
    # Decode frame
    try:
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode frame")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {str(e)}")
    
    # Resize if needed
    h, w = frame.shape[:2]
    if w > config.MAX_FRAME_WIDTH or h > config.MAX_FRAME_HEIGHT:
        scale = min(config.MAX_FRAME_WIDTH / w, config.MAX_FRAME_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        h, w = new_h, new_w
    
    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Process with MediaPipe
    det = face_detection.process(rgb)
    mesh = face_mesh.process(rgb)
    
    face_conf = 0.0
    if det.detections:
        face_conf = max([d.score[0] for d in det.detections])
    
    # Simplified: Only track gaze, nothing else
    gaze_on_screen_percentage = 0
    gaze_dir = "UNKNOWN"
    gaze_distance = None
    
    state['frame_count'] += 1
    now = time.time()
    
    if mesh.multi_face_landmarks and face_conf >= config.FACE_DET_CONF:
        # Reset no-face timer
        state['no_face_start'] = None
        
        lm = mesh.multi_face_landmarks[0].landmark
        
        # ONLY GAZE TRACKING - ignore everything else
        avgx, avgy = get_iris_avg(lm)
        if avgx is not None:
            # Get relative iris position for eyeball-specific detection
            rel_x, rel_y = get_iris_relative_position(lm)
            
            if not state['calibrated']:
                # Collect calibration data (both absolute and relative)
                state['calib_x'].append(avgx)
                state['calib_y'].append(avgy)
                if rel_x is not None and rel_y is not None:
                    state['calib_rel_x'].append(rel_x)
                    state['calib_rel_y'].append(rel_y)
                if state['calib_start'] is None:
                    state['calib_start'] = now
                
                # Check if calibration complete
                if now - state['calib_start'] >= config.CALIB_SECONDS and len(state['calib_x']) > 0:
                    state['baseline_x'] = float(np.mean(state['calib_x']))
                    state['baseline_y'] = float(np.mean(state['calib_y']))
                    if len(state['calib_rel_x']) > 0:
                        state['baseline_rel_x'] = float(np.mean(state['calib_rel_x']))
                        state['baseline_rel_y'] = float(np.mean(state['calib_rel_y']))
                    else:
                        # Fallback if relative calibration failed
                        state['baseline_rel_x'] = 0.5
                        state['baseline_rel_y'] = 0.5
                    state['calibrated'] = True
                    print(f"Session {session_id} calibrated: baseline=({state['baseline_x']:.3f}, {state['baseline_y']:.3f})")
                    print(f"Relative baseline=({state['baseline_rel_x']:.3f}, {state['baseline_rel_y']:.3f}) [0.5 = centered]")
            else:
                # EYEBALL-SPECIFIC DETECTION: Use relative iris position (independent of head movement)
                if rel_x is None or rel_y is None or state['baseline_rel_x'] is None or state['baseline_rel_y'] is None:
                    # Fallback to absolute position if relative fails
                    dx = avgx - state['baseline_x']
                    dy = avgy - state['baseline_y']
                    gaze_distance = float(np.sqrt(dx**2 + dy**2))
                    if abs(dx) <= config.GAZE_X_DELTA and abs(dy) <= config.GAZE_Y_DELTA:
                        gaze_dir = "CENTER"
                    elif abs(dx) > abs(dy):
                        gaze_dir = "LEFT" if dx < 0 else "RIGHT"
                    else:
                        gaze_dir = "UP" if dy < 0 else "DOWN"
                else:
                    # Method 1: Relative position (eyeball-specific) - PRIMARY METHOD
                    rel_dx = rel_x - state['baseline_rel_x']
                    rel_dy = rel_y - state['baseline_rel_y']
                    
                    # Method 2: Absolute position (backup/combined)
                    abs_dx = avgx - state['baseline_x']
                    abs_dy = avgy - state['baseline_y']
                    
                    # Use BOTH methods for maximum accuracy
                    combined_dx = 0.7 * rel_dx + 0.3 * abs_dx
                    combined_dy = 0.7 * rel_dy + 0.3 * abs_dy
                    
                    # Calculate distance from center
                    gaze_distance = float(np.sqrt(combined_dx**2 + combined_dy**2))
                    
                    # Ultra-sensitive thresholds for detecting even slight eyeball shifts
                    rel_threshold = getattr(config, 'EYEBALL_RELATIVE_THRESHOLD', 0.15)
                    
                    # Check if eyeball has shifted
                    if abs(rel_dx) <= rel_threshold and abs(rel_dy) <= rel_threshold:
                        # Also check absolute position as secondary check
                        if abs(combined_dx) <= config.GAZE_X_DELTA and abs(combined_dy) <= config.GAZE_Y_DELTA:
                            gaze_dir = "CENTER"
                        else:
                            # Small deviation detected - be more lenient
                            if abs(combined_dx) > abs(combined_dy):
                                gaze_dir = "LEFT" if combined_dx < 0 else "RIGHT"
                            else:
                                gaze_dir = "UP" if combined_dy < 0 else "DOWN"
                    else:
                        # Eyeball has shifted significantly
                        if abs(rel_dx) > abs(rel_dy):
                            gaze_dir = "LEFT" if rel_dx < 0 else "RIGHT"
                        else:
                            gaze_dir = "UP" if rel_dy < 0 else "DOWN"
                    
                    # Add to smoothing buffer
                    state['gaze_dir_buf'].append(gaze_dir)
                    
                    # Apply temporal smoothing - only if we have enough consistency
                    if len(state['gaze_dir_buf']) >= 2:
                        most_common = Counter(state['gaze_dir_buf']).most_common(1)[0][0]
                        count = Counter(state['gaze_dir_buf']).most_common(1)[0][1]
                        # Only smooth if majority agrees (at least 2 out of 3)
                        if count >= 2:
                            gaze_dir = most_common
        
        # Compute gaze on-screen percentage
        if gaze_distance is not None:
            gaze_on_screen_percentage = compute_gaze_on_screen_percentage(gaze_distance)
        else:
            gaze_on_screen_percentage = 0
    else:
        # No face detected
        gaze_on_screen_percentage = 0
        gaze_dir = "NO FACE"
        if state['no_face_start'] is None:
            state['no_face_start'] = now
        elif now - state['no_face_start'] >= config.NO_FACE_SECONDS:
            # Could signal to stop processing this session
            pass
    
    # Update score buffer with gaze percentage
    state['score_buf'].append(gaze_on_screen_percentage)
    smooth_percentage = int(np.mean(state['score_buf'])) if len(state['score_buf']) > 0 else gaze_on_screen_percentage
    
    # Simple status based on gaze percentage
    if not mesh.multi_face_landmarks or face_conf < config.FACE_DET_CONF:
        status = "NO FACE"
    elif smooth_percentage >= 80:
        status = "FOCUSED"  # Looking at screen (80-100%)
    elif smooth_percentage >= 50:
        status = "PARTIAL"  # Partially looking (50-79%)
    elif smooth_percentage >= 20:
        status = "DISTRACTED"  # Looking away (20-49%)
    else:
        status = "OFF_SCREEN"  # Not looking at screen (0-19%)
    
    return {
        'gaze_on_screen_percentage': smooth_percentage,  # Main metric: 0-100%
        'status': status,
        'gaze_direction': gaze_dir,
        'calibrated': state['calibrated'],
        'smooth_score': smooth_percentage,  # For compatibility
    }


@app.post("/process-frame", response_model=FrameResponse)
async def process_frame_endpoint(request: FrameRequest):
    """Process a single frame and return concentration metrics"""
    try:
        # Decode base64 frame
        frame_data = base64.b64decode(request.frame_data)
        result = process_frame(frame_data, request.session_id)
        return FrameResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset-session/{session_id}")
async def reset_session(session_id: str):
    """Reset calibration and state for a session"""
    with state_lock:
        if session_id in session_states:
            del session_states[session_id]
    return {"status": "reset", "session_id": session_id}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "active_sessions": len(session_states),
        "config": {
            "fps": config.FRAME_PROCESSING_FPS,
            "max_width": config.MAX_FRAME_WIDTH,
            "max_height": config.MAX_FRAME_HEIGHT,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.ML_SERVICE_HOST,
        port=config.ML_SERVICE_PORT,
        workers=1  # MediaPipe models are not thread-safe
    )

