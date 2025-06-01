import numpy as np

# Constants
CONFIDENCE_THRESHOLD = 0.7      # Threshold to determine if a detection is valid
MAX_HISTORY = 10                # Maximum number entries in history
REMOVE_TIME_FRAME = 8           # Number of frames to remove a vehicle
SPEED_THRESHOLD = 5.0           # Speed threshold to determine if a vehicle is fast
TTC_THRESHOLD = 2.0             # Time to collision threshold to determine if a vehicle is dangerous
IOU_THRESHOLD = 0.3             # Threshold to determine if a detection is a match
WARNING_STICKY_TIME_FRAME = 10  # Number of frames to show a warning
METRIC_HISTORY_GAP = 2          # Number of frames to skip for metrics
ROI_MIN = 0.15                  # Minimum ROI (Region-of-Interest) for vehicle detection
ROI_MAX = 0.85                  # Maximum ROI (Region-of-Interest) for vehicle detection

# Kalman filter parameters
KALMAN_STATE_TRANSITION_MATRIX = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
    [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
    [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
    [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
    [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx
    [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy
    [0, 0, 0, 0, 0, 0, 1, 0],  # dw = dw
    [0, 0, 0, 0, 0, 0, 0, 1]   # dh = dh
], np.float32)

KALMAN_MEASUREMENT_MATRIX = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],  # x
    [0, 1, 0, 0, 0, 0, 0, 0],  # y
    [0, 0, 1, 0, 0, 0, 0, 0],  # w
    [0, 0, 0, 1, 0, 0, 0, 0]   # h
], np.float32)

# Very low noise for width-related states to strongly prevent sudden changes
# Higher noise for position to allow movement
KALMAN_PROCESS_NOISE_COV = np.array([
    [0.01, 0, 0, 0, 0, 0, 0, 0],    # x
    [0, 0.03, 0, 0, 0, 0, 0, 0],    # y
    [0, 0, 0.01, 0, 0, 0, 0, 0],   # w
    [0, 0, 0, 0.02, 0, 0, 0, 0],    # h
    [0, 0, 0, 0, 0.01, 0, 0, 0],    # dx
    [0, 0, 0, 0, 0, 0.03, 0, 0],    # dy
    [0, 0, 0, 0, 0, 0, 0.01, 0],   # dw
    [0, 0, 0, 0, 0, 0, 0, 0.02]     # dh
], np.float32)

# Very high noise for width measurement to be extremely conservative with changes
# Lower noise for position to trust position measurements
KALMAN_MEASUREMENT_NOISE_COV = np.array([
    [0.7, 0, 0, 0],  # x
    [0, 0.1, 0, 0],  # y
    [0, 0, 0.7, 0],    # w
    [0, 0, 0, 0.2]   # h
], np.float32)

# Camera dimensions
CAMERA_PREVIEW_DIM = (640, 640)
# CAMERA_PREVIEW_DIM = (960, 960)
# CAMERA_PREVIEW_DIM = (1280, 1280)

# Labels for detected objects
LABELS = ["Vehicle"]

# Training parameters
EPOCHS      = 200
BATCH_SIZE  = 32
IMAGE_SIZE  = 640
CONFIG      = './dataset/dataset.yaml'
MODEL       = 'yolov8n.pt'
OUTPUT_DIR  = './runs'

# Downloader parameters
TRAIN_SAMPLES = 6000
VAL_SAMPLES   = 1500