# Constants
CONFIDENCE_THRESHOLD = 0.7
MAX_HISTORY = 10
SIZE_THRESHOLD = 0.1

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