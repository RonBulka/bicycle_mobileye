"""
Constants Configuration Module

This module contains all configuration constants used throughout the vehicle tracking system.
It centralizes parameters for detection thresholds, tracking behavior, Kalman filtering,
and system configuration to enable easy tuning and maintenance.

The constants are organized into logical groups:
- Detection and tracking thresholds
- Kalman filter matrices and parameters
- Camera and model configuration
- Training parameters
- Dataset configuration
"""

import numpy as np

# =============================================================================
# DETECTION AND TRACKING PARAMETERS
# =============================================================================
CONFIDENCE_THRESHOLD = 0.7      # Minimum confidence score for valid detections
MAX_HISTORY = 10                # Maximum number of historical entries per vehicle
REMOVE_TIME_FRAME = 8           # Frames to wait before removing inactive vehicles
SPEED_THRESHOLD = 5.0           # Speed threshold (percentage) for warning triggers
TTC_THRESHOLD = 2.0             # Time-to-collision threshold (seconds) for warnings
IOU_THRESHOLD = 0.3             # Intersection-over-Union threshold for track matching
WARNING_STICKY_TIME_FRAME = 10  # Frames to maintain warning state after trigger
METRIC_HISTORY_GAP = 2          # Frame gap for calculating speed and TTC metrics
ROI_MIN = 0.15                  # Minimum ROI boundary (normalized 0-1)
ROI_MAX = 0.85                  # Maximum ROI boundary (normalized 0-1)

# =============================================================================
# KALMAN FILTER PARAMETERS
# =============================================================================
# State transition matrix A: [x, y, w, h, dx, dy, dw, dh]
# Models constant velocity with position and size components
KALMAN_STATE_TRANSITION_MATRIX = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
    [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
    [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
    [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
    [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx (constant velocity)
    [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy (constant velocity)
    [0, 0, 0, 0, 0, 0, 1, 0],  # dw = dw (constant size change)
    [0, 0, 0, 0, 0, 0, 0, 1]   # dh = dh (constant size change)
], np.float32)

# Measurement matrix H: Maps state to measurements [x, y, w, h]
KALMAN_MEASUREMENT_MATRIX = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],  # x measurement
    [0, 1, 0, 0, 0, 0, 0, 0],  # y measurement
    [0, 0, 1, 0, 0, 0, 0, 0],  # w measurement
    [0, 0, 0, 1, 0, 0, 0, 0]   # h measurement
], np.float32)

# Process noise covariance Q: Controls prediction uncertainty
# Lower values = more trust in predictions, higher values = more responsive to measurements
KALMAN_PROCESS_NOISE_COV = np.array([
    [0.01, 0, 0, 0, 0, 0, 0, 0],    # x position noise
    [0, 0.03, 0, 0, 0, 0, 0, 0],    # y position noise
    [0, 0, 0.01, 0, 0, 0, 0, 0],    # width noise (low for stability)
    [0, 0, 0, 0.02, 0, 0, 0, 0],    # height noise
    [0, 0, 0, 0, 0.01, 0, 0, 0],    # x velocity noise
    [0, 0, 0, 0, 0, 0.03, 0, 0],    # y velocity noise
    [0, 0, 0, 0, 0, 0, 0.01, 0],    # width velocity noise (low)
    [0, 0, 0, 0, 0, 0, 0, 0.02]     # height velocity noise
], np.float32)

# Measurement noise covariance R: Controls measurement trust
# Higher values = less trust in measurements, lower values = more responsive to measurements
KALMAN_MEASUREMENT_NOISE_COV = np.array([
    [0.7, 0, 0, 0],  # x measurement noise (high - conservative)
    [0, 0.1, 0, 0],  # y measurement noise (low - trust vertical position)
    [0, 0, 0.7, 0],  # width measurement noise (high - conservative for size)
    [0, 0, 0, 0.2]   # height measurement noise (moderate)
], np.float32)

# =============================================================================
# CAMERA AND MODEL CONFIGURATION
# =============================================================================
CAMERA_PREVIEW_DIM = (640, 640)  # Camera preview dimensions (width, height)
# Alternative resolutions for different use cases:
# CAMERA_PREVIEW_DIM = (960, 960)   # Higher resolution
# CAMERA_PREVIEW_DIM = (1280, 1280) # Maximum resolution

# Object detection labels
LABELS = ["Vehicle"]

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
EPOCHS      = 200    # Number of training epochs
BATCH_SIZE  = 32     # Training batch size
IMAGE_SIZE  = 640    # Input image size for training
CONFIG      = './dataset/dataset.yaml'  # Dataset configuration file
MODEL       = 'yolov8n.pt'             # Base model for transfer learning
OUTPUT_DIR  = './runs'                  # Training output directory

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
TRAIN_SAMPLES = 6000  # Number of training samples to download
VAL_SAMPLES   = 1500   # Number of validation samples to download