# Configuration file for ML processing service
# Adjust these values to experiment with different settings

# Frame Processing Configuration
FRAME_PROCESSING_FPS = 15  # Frames per second to process (5-15 recommended for balance)
MAX_FRAME_WIDTH = 640  # Resize frames to this width (smaller = faster, less accurate)
MAX_FRAME_HEIGHT = 480  # Resize frames to this height
JPEG_QUALITY = 70  # JPEG compression quality (0-100, lower = smaller but less quality)

# ML Model Configuration - EXACTLY matching ml.py
EAR_BLINK_THRESHOLD = 0.18
BLINK_CONSEC_FRAMES = 2
FACE_DET_CONF = 0.45
EYE_VARIANCE_THRESHOLD = 200.0   # occlusion check: variance too low => likely covered
EYE_MEAN_DARK = 45.0             # occlusion check: mean too dark => covered
GAZE_X_DELTA = 0.025             # threshold around calibrated center for LEFT/RIGHT (ultra-sensitive for eyeball detection)
GAZE_Y_DELTA = 0.02              # threshold around calibrated center for UP/DOWN (ultra-sensitive for eyeball detection)
GAZE_SMOOTH_WINDOW = 3           # frames to smooth gaze direction (reduced for faster response)
EYEBALL_RELATIVE_THRESHOLD = 0.15  # threshold for iris position relative to eye corners (detects actual eyeball movement)
SCORE_SMOOTH = 6
NOISE_SENSITIVITY = 2.0          # how much above baseline RMS counts as noise
EYES_CLOSED_SECONDS = 3.0        # display eyes closed if closed for this long
NO_FACE_SECONDS = 100.0           # exit if no face detected this many seconds
BLINK_MIN_SEP = 0.35             # seconds between blink events

# Calibration Configuration - EXACTLY matching ml.py
CALIB_SECONDS = 3.0
AUDIO_CALIB_SECONDS = 1.0  # Seconds to calibrate audio baseline
AUDIO_SR = 22050  # Audio sample rate
AUDIO_BLOCKSIZE = 1024  # Audio block size

# Server Configuration
ML_SERVICE_HOST = "0.0.0.0"  # Host to bind the ML service
ML_SERVICE_PORT = 8000  # Port for the ML service
MAX_WORKERS = 4  # Maximum concurrent workers for processing
ENABLE_AUDIO = False  # Enable audio processing (requires microphone access)

# Performance Tuning
ENABLE_GPU = False  # Enable GPU acceleration if available
BATCH_SIZE = 1  # Batch size for processing (1 = sequential)
SKIP_FRAMES_IF_BUSY = True  # Skip frames if previous processing not done

