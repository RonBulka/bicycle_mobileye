# Constants
CONFIDENCE_THRESHOLD = 0.7      # Threshold to determine if a detection is valid
MAX_HISTORY = 10                # Maximum number entries in history
REMOVE_TIME_FRAME = 8           # Number of frames to remove a vehicle
SPEED_THRESHOLD = 5.0           # Speed threshold to determine if a vehicle is fast
TTC_THRESHOLD = 2.0             # Time to collision threshold to determine if a vehicle is dangerous
IOU_THRESHOLD = 0.3             # Threshold to determine if a detection is a match
WARNING_STICKY_TIME_FRAME = 10  # Number of frames to show a warning
METRIC_HISTORY_GAP = 2          # Number of frames to skip for metrics

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