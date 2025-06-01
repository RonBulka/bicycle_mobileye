from dataclasses import dataclass
from typing import Tuple, List, Dict, Protocol
from collections import deque
import numpy as np
import cv2
from constants import   CONFIDENCE_THRESHOLD, \
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

class VehicleTracker:
    def __init__(self, max_history: int = MAX_HISTORY):
        self.tracked_vehicles: Dict[int, TrackedVehicle] = {}
        self.next_id = 0
        self.max_history = max_history
        self.frame_count = 0
        
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
            
            bbox = frame_norm(frame_shape, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            width = bbox[2] - bbox[0]

            # Find best matching existing track
            best_match = None
            best_iou = 0.0

            for vehicle in self.tracked_vehicles.values():
                prev_bbox = vehicle.bbox
                iou = self.calculate_iou(bbox, prev_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = vehicle

            if best_match and best_iou > IOU_THRESHOLD:  # Update existing track
                best_match.update(bbox)  # Update Kalman filter
                width = best_match.bbox[2] - best_match.bbox[0]
                best_match.confidence = detection.confidence
                best_match.width_history.append((width, current_frame))
                best_match.last_update = current_frame
                best_match.ignore_warning |= ignore_for_warning

                # Keep history at fixed size
                if len(best_match.width_history) > self.max_history:
                    best_match.width_history.popleft()
                if len(best_match.momentary_ttc_history) > self.max_history:
                    best_match.momentary_ttc_history.popleft()

                # Warnings if this track is not flagged ignore_warning
                if (not best_match.ignore_warning):
                    # Check for warning condition
                    if best_match.speed > SPEED_THRESHOLD and best_match.ttc < TTC_THRESHOLD:
                        if not best_match.warning:
                            best_match.warning = True
                            best_match.count_warning = current_frame
                        else: # Reset warning counter
                            best_match.count_warning = current_frame

                    # Check for sticky warning
                    if best_match.warning:
                        if current_frame - best_match.count_warning > WARNING_STICKY_TIME_FRAME:
                            best_match.warning = False
                            best_match.count_warning = 0

            else:  # Create new track
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
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

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