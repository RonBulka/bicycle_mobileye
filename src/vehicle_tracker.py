"""
Vehicle Tracker Module

This module implements a real-time vehicle tracking system with collision warning capabilities.
It uses YOLO object detection combined with Kalman filtering for robust vehicle tracking.

Key Features:
- Multi-object tracking with unique vehicle IDs
- Kalman filter-based prediction and state estimation
- Time-to-collision (TTC) calculation for safety warnings
- Speed estimation based on vehicle size changes
- Audio warning system for approaching vehicles
- Region-of-interest (ROI) filtering for side vehicles
- Adaptive noise filtering for improved tracking stability

Classes:
- TrackedVehicle: Represents a tracked vehicle with state and history
- WarningStateManager: Manages audio warnings in a separate thread
- VehicleTracker: Main tracking system that manages all vehicles

The system is designed for bicycle safety applications, detecting and warning about
approaching vehicles that pose collision risks.
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Protocol
from collections import deque
import numpy as np
import cv2
import pygame
import time
import os
import threading
try:
    from .constants import  CONFIDENCE_THRESHOLD, \
                            MAX_HISTORY, \
                            REMOVE_TIME_FRAME, \
                            SPEED_THRESHOLD, \
                            TTC_THRESHOLD, \
                            IOU_THRESHOLD, \
                            WARNING_STICKY_TIME_FRAME, \
                            METRIC_HISTORY_GAP, \
                            ROI_MIN, \
                            ROI_MAX, \
                            KALMAN_STATE_TRANSITION_MATRIX, \
                            KALMAN_MEASUREMENT_MATRIX, \
                            KALMAN_PROCESS_NOISE_COV, \
                            KALMAN_MEASUREMENT_NOISE_COV
except ImportError:
    from constants import  CONFIDENCE_THRESHOLD, \
                            MAX_HISTORY, \
                            REMOVE_TIME_FRAME, \
                            SPEED_THRESHOLD, \
                            TTC_THRESHOLD, \
                            IOU_THRESHOLD, \
                            WARNING_STICKY_TIME_FRAME, \
                            METRIC_HISTORY_GAP, \
                            ROI_MIN, \
                            ROI_MAX, \
                            KALMAN_STATE_TRANSITION_MATRIX, \
                            KALMAN_MEASUREMENT_MATRIX, \
                            KALMAN_PROCESS_NOISE_COV, \
                            KALMAN_MEASUREMENT_NOISE_COV

class DetectionProtocol(Protocol):
    """Protocol defining the required attributes for detection objects.
    
    Any object passed to the tracker must have these attributes:
    - xmin: float - normalized x coordinate of top-left corner (0-1)
    - ymin: float - normalized y coordinate of top-left corner (0-1)
    - xmax: float - normalized x coordinate of bottom-right corner (0-1)
    - ymax: float - normalized y coordinate of bottom-right corner (0-1)
    - confidence: float - detection confidence score (0-1)
    - label: int - class label of the detection
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    label: int

def get_adaptive_noise(change_ratio):
    """Get adaptive noise based on change ratio."""
    if change_ratio < 0.05:  # Small change
        return 0.1  # Trust small changes
    elif change_ratio > 0.20:  # Large change
        return 2.0  # Be very skeptical of large changes
    else:  # Medium change
        # Linear interpolation between small and large noise
        return 0.1 + (change_ratio - 0.05) * (2.0 - 0.1) / (0.20 - 0.05)

@dataclass
class TrackedVehicle:
    id: int
    bbox: Tuple[int, int, int, int]     # xmin, ymin, xmax, ymax
    confidence: float
    width_history: deque                # Store last N frame widths
    momentary_ttc_history: deque        # Store last N TTCs values
    last_update: float
    speed: float = 0.0
    ttc: float = float('inf')           # Time to collision in seconds
    warning: bool = False
    count_warning: int = 0
    ignore_warning: bool = False        # For ignoring side-entering vehicles
    kalman: cv2.KalmanFilter = None     # Kalman filter for tracking

    def __post_init__(self):
        # Initialize Kalman filter
        self.kalman = cv2.KalmanFilter(8, 4)  # 8 state variables, 4 measurements

        # State transition matrix (A)
        self.kalman.transitionMatrix    = KALMAN_STATE_TRANSITION_MATRIX

        # Measurement matrix (H)
        self.kalman.measurementMatrix   = KALMAN_MEASUREMENT_MATRIX

        # Process noise covariance (Q)
        self.kalman.processNoiseCov     = KALMAN_PROCESS_NOISE_COV

        # Measurement noise covariance (R)
        self.kalman.measurementNoiseCov = KALMAN_MEASUREMENT_NOISE_COV

        # Initialize state
        x, y, w, h = self._bbox_to_state(self.bbox)
        self.kalman.statePre    = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], np.float32)
        self.kalman.statePost   = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], np.float32)

    def _bbox_to_state(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        """Convert bounding box to state representation (center x, y, width, height)."""
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return x, y, w, h

    def _state_to_bbox(self, state: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert state representation to bounding box."""
        x, y, w, h = state[:4].flatten()
        return (
            int(x - (w / 2)),
            int(y - (h / 2)),
            int(x + (w / 2)),
            int(y + (h / 2))
        )

    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next state using Kalman filter."""
        prediction = self.kalman.predict()
        return self._state_to_bbox(prediction)

    def update(self, bbox: Tuple[int, int, int, int]):
        """Update Kalman filter with new measurement using adaptive noise."""
        x, y, w, h = self._bbox_to_state(bbox)
        measurement = np.array([[x], [y], [w], [h]], np.float32)

        # # Get current state estimate
        # current_state = self.kalman.statePost

        # # Calculate relative changes
        # current_width = current_state[2, 0]
        # new_width = measurement[2, 0]
        # width_change_ratio = abs(new_width - current_width) / current_width

        # current_x = current_state[0, 0]
        # new_x = measurement[0, 0]
        # x_change_ratio = abs(new_x - current_x) / current_x if current_x != 0 else 0

        # # Update measurement noise matrix with adaptive noise for both width and x
        # self.kalman.measurementNoiseCov[0, 0] = get_adaptive_noise(x_change_ratio)
        # self.kalman.measurementNoiseCov[2, 2] = get_adaptive_noise(width_change_ratio)

        # Update Kalman filter
        self.kalman.correct(measurement)
        self.bbox = self._state_to_bbox(self.kalman.statePost)

class WarningStateManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._has_warning = False
        self._stop_event = threading.Event()
        self._audio_thread = None

    def update_warning_state(self, tracked_vehicles: List[TrackedVehicle]):
        with self._lock:
            self._has_warning = any(vehicle.warning for vehicle in tracked_vehicles)

    def has_warning(self) -> bool:
        with self._lock:
            return self._has_warning

    def start_audio_thread(self):
        if self._audio_thread is None or not self._audio_thread.is_alive():
            self._stop_event.clear()
            self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._audio_thread.start()

    def stop_audio_thread(self):
        self._stop_event.set()
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join()

    def _audio_loop(self):
        """Audio loop that runs in a separate thread."""
        file_path = "audio/clock-alarm-8762.mp3"
        if not os.path.exists(file_path):
            print(f"Audio file {file_path} not found.")
            return

        try:
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            
            while not self._stop_event.is_set():
                if self.has_warning():
                    pygame.mixer.music.play()
                    # Wait for the audio to finish or until warning state changes
                    while pygame.mixer.music.get_busy() and self.has_warning() and not self._stop_event.is_set():
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)  # Small sleep to prevent busy waiting

        except Exception as e:
            print(f"Error in audio thread: {e}")
        finally:
            pygame.mixer.quit()

class VehicleTracker:
    def __init__(self, max_history: int = MAX_HISTORY):
        self.tracked_vehicles: Dict[int, TrackedVehicle] = {}
        self.next_id = 0
        self.max_history = max_history
        self.frame_count = 0
        self.warning_manager = WarningStateManager()
        self.warning_manager.start_audio_thread()

    def _update_warning_state(self, vehicle: TrackedVehicle, current_frame: int):
        """Update warning state for a vehicle."""
        # Warnings if this track is not flagged ignore_warning
        if (not vehicle.ignore_warning):
            # Check for warning condition
            if vehicle.speed > SPEED_THRESHOLD and vehicle.ttc < TTC_THRESHOLD:
                if not vehicle.warning:
                    vehicle.warning = True
                    vehicle.count_warning = current_frame
                else: # Reset warning counter
                    vehicle.count_warning = current_frame

            # Check for sticky warning
            if vehicle.warning:
                if current_frame - vehicle.count_warning > WARNING_STICKY_TIME_FRAME:
                    vehicle.warning = False
                    vehicle.count_warning = 0

    def calculate_vehicle_metrics(self, vehicle: TrackedVehicle, fps: float) -> Tuple[float, float]:
        """Calculate speed and time to collision for a vehicle."""
        if len(vehicle.width_history) < (METRIC_HISTORY_GAP + 2):
            return 0.0, float('inf')

        # Calculate relative width change
        current_width, current_frame = vehicle.width_history[-1]
        prev_width, prev_frame = vehicle.width_history[-(METRIC_HISTORY_GAP + 2)]
        width_change = (current_width - prev_width) / prev_width
        width_relative_change = (current_width / prev_width)
        delta_time = (current_frame - prev_frame) / fps

        # Calculate speed based on width change
        speed = width_change * 100  # Convert to percentage

        # Calculate time to collision (TTC)
        if width_relative_change > 1:
            momentary_ttc = (delta_time / (width_relative_change - 1))
        else:
            momentary_ttc = float('inf')
        return speed, momentary_ttc

    # def calculate_ttc(self, vehicle: TrackedVehicle) -> float:
    #     """Calculate the time to collision for a vehicle."""
    #     if len(vehicle.momentary_ttc_history) < (METRIC_HISTORY_GAP + 2):
    #         return float('inf')

    #     current_ttc, current_frame = vehicle.momentary_ttc_history[-1]
    #     prev_ttc, prev_frame = vehicle.momentary_ttc_history[-(METRIC_HISTORY_GAP + 2)]
    #     momentary_ttc_derivative = (current_ttc - prev_ttc) / (current_frame - prev_frame)
    #     variable_C = momentary_ttc_derivative + 1
    #     ttc = current_ttc * (1 - np.sqrt(1 + 2 * variable_C)) / variable_C
    #     return ttc

    def update(self, detections: List[DetectionProtocol], frame_shape: np.ndarray, frame_number: int, fps: float) -> List[TrackedVehicle]:
        """Update tracked vehicles with new detections.
        
        Args:
            detections: List of detection objects. Each detection must have:
                - xmin, ymin, xmax, ymax: normalized coordinates (0-1)
                - confidence: detection confidence score (0-1)
                - label: class label
            frame_shape: Tuple of (height, width) of the frame
            frame_number: Current frame number
            fps: Frames per second
        Returns:
            List of currently tracked vehicles
        """
        current_frame = frame_number
        self.frame_count += 1

        # Predict next state for all existing tracks
        for vehicle in self.tracked_vehicles.values():
            predicted_bbox = vehicle.predict()
            vehicle.bbox = predicted_bbox

        # Update existing tracks
        for vehicle_id, vehicle in list(self.tracked_vehicles.items()):
            # Remove old tracks
            if current_frame - vehicle.last_update > REMOVE_TIME_FRAME:
                del self.tracked_vehicles[vehicle_id]
                continue

            # Calculate metrics
            speed, momentary_ttc = self.calculate_vehicle_metrics(vehicle, fps)
            vehicle.speed = speed
            vehicle.ttc = momentary_ttc

        # Match new detections to existing tracks
        for detection in detections:
            if detection.confidence < CONFIDENCE_THRESHOLD:
                continue

            # ROI‐based ignore flag
            cx_norm = (detection.xmin + detection.xmax) / 2
            ignore_for_warning = (cx_norm < ROI_MIN) or (cx_norm > ROI_MAX)

            # Convert normalized coordinates to pixel coordinates
            bbox = frame_norm(frame_shape, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

            # Try to match with existing tracks
            best_iou = 0
            best_match = None

            for vehicle in self.tracked_vehicles.values():
                iou = self.calculate_iou(vehicle.bbox, bbox)
                if iou > best_iou and iou > IOU_THRESHOLD:
                    best_iou = iou
                    best_match = vehicle

            if best_match:               # Update existing track
                best_match.update(bbox)  # Update Kalman filter
                width = best_match.bbox[2] - best_match.bbox[0]
                best_match.confidence = detection.confidence
                best_match.width_history.append((width, current_frame))
                best_match.last_update = current_frame
                best_match.ignore_warning |= ignore_for_warning
                best_match.warning = best_match.warning and not best_match.ignore_warning

                # Update width history
                width = bbox[2] - bbox[0]
                best_match.width_history.append((width, current_frame))
                if len(best_match.width_history) > self.max_history:
                    best_match.width_history.popleft()

                # Calculate metrics
                speed, momentary_ttc = self.calculate_vehicle_metrics(best_match, fps)
                best_match.speed = speed
                best_match.momentary_ttc_history.append((momentary_ttc, current_frame))
                if len(best_match.momentary_ttc_history) > self.max_history:
                    best_match.momentary_ttc_history.popleft()

                # Update warning state
                self._update_warning_state(best_match, current_frame)

            else:
                width = bbox[2] - bbox[0]
                # Create new track
                new_vehicle = TrackedVehicle(
                    id=self.next_id,
                    bbox=bbox,
                    confidence=detection.confidence,
                    width_history=deque([(width, current_frame)], maxlen=self.max_history),
                    momentary_ttc_history=deque([(float('inf'), current_frame)], maxlen=self.max_history),
                    last_update=current_frame,
                    ignore_warning=ignore_for_warning
                )
                self.tracked_vehicles[self.next_id] = new_vehicle
                self.next_id += 1

        # Update warning manager
        self.warning_manager.update_warning_state(list(self.tracked_vehicles.values()))

        return list(self.tracked_vehicles.values())

    @staticmethod
    def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

def frame_norm(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Normalizes bounding box coordinates to frame dimensions."""
    # Handle both frame objects and frame shape arrays
    if hasattr(frame, 'shape') and len(frame.shape) >= 2:
        # frame is an actual frame object
        height, width = frame.shape[0], frame.shape[1]
    else:
        # frame is a shape array [height, width, channels]
        height, width = frame[0], frame[1]
    
    # Create normalization values: [width, height, width, height] for [xmin, ymin, xmax, ymax]
    norm_vals = np.array([width, height, width, height])
    return tuple((np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int))

def annotate_frame(frame: np.ndarray, tracked_vehicles: List[TrackedVehicle], fps: float) -> np.ndarray:
    """Annotates a frame with tracked vehicles and their metrics."""
    color = (0, 0, 255)
    for vehicle in tracked_vehicles:
        bbox = vehicle.bbox
        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Compose info text
        info_text = f"ID: {vehicle.id} | Speed: {vehicle.speed:.1f}% | TTC: {vehicle.ttc:.1f}s"
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_TRIPLEX, 0.3, 1)
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (bbox[0], bbox[1] - text_height - baseline - 4),
            (bbox[0] + text_width, bbox[1]),
            (255, 255, 255),
            -1
        )
        # Draw info text above the bbox
        cv2.putText(
            frame,
            info_text,
            (bbox[0], bbox[1] - 4),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA
        )

        # Add warning if vehicle is approaching too fast
        if vehicle.warning:
            warning_text = "WARNING: Approaching vehicle!"
            # Calculate center of bbox
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            # Get text size
            (warn_width, warn_height), warn_baseline = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 2)
            # Calculate bottom-left corner for centered text
            warn_x = center_x - warn_width // 2
            warn_y = center_y + warn_height // 2
            # Draw warning text
            cv2.putText(
                frame,
                warning_text,
                (warn_x, warn_y),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

    # Annotate the frame with the FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)
    return frame