from dataclasses import dataclass
from typing import Tuple, List, Dict, Protocol
from collections import deque
import time
import numpy as np
import cv2
from constansts import  CONFIDENCE_THRESHOLD, \
                        MAX_HISTORY, \
                        SIZE_THRESHOLD, \
                        REMOVE_TIME_FRAME, \
                        SPEED_THRESHOLD, \
                        TTC_THRESHOLD, \
                        IOU_THRESHOLD, \
                        WARNING_STICKY_TIME_FRAME

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

@dataclass
class TrackedVehicle:
    id: int
    bbox: Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax
    confidence: float
    size_history: deque  # Store last N frame sizes
    position_history: deque  # Store last N frame positions - doesnt seem to be used
    last_update: float
    speed: float = 0.0
    ttc: float = float('inf')  # Time to collision in seconds
    warning: bool = False
    count_warning: int = 0

class VehicleTracker:
    def __init__(self, max_history: int = MAX_HISTORY, size_threshold: float = SIZE_THRESHOLD):
        self.tracked_vehicles: Dict[int, TrackedVehicle] = {}
        self.next_id = 0
        self.max_history = max_history
        self.size_threshold = size_threshold
        self.frame_count = 0
        
    def calculate_vehicle_metrics(self, vehicle: TrackedVehicle) -> Tuple[float, float]:
        """Calculate speed and time to collision for a vehicle."""
        if len(vehicle.size_history) < 2:
            return 0.0, float('inf')

        # Calculate size change rate
        current_size = vehicle.size_history[-1]
        prev_size = vehicle.size_history[-2]
        size_change = (current_size - prev_size) / prev_size

        if size_change < 0:
            size_change = 0

        # Calculate speed based on size change
        speed = size_change * 100  # Convert to percentage

        # Calculate time to collision (TTC)
        # TTC = current_size / (size_change_rate * current_size)
        if abs(size_change) > 0.001:  # Avoid division by very small numbers
            ttc = 1.0 / abs(size_change)
        else:
            ttc = float('inf')
        return speed, ttc

    def update(self, detections: List[DetectionProtocol], frame_shape: np.ndarray, frame_number: int) -> List[TrackedVehicle]:
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
        # TODO: Change to use frame number instead of time
        current_frame = frame_number
        self.frame_count += 1

        # Update existing tracks
        for vehicle_id, vehicle in list(self.tracked_vehicles.items()):
            # Remove old tracks
            if current_frame - vehicle.last_update > REMOVE_TIME_FRAME:
                del self.tracked_vehicles[vehicle_id]
                continue

            # Calculate metrics
            speed, ttc = self.calculate_vehicle_metrics(vehicle)
            vehicle.speed = speed
            vehicle.ttc = ttc

        # Match new detections to existing tracks
        for detection in detections:
            if detection.confidence < CONFIDENCE_THRESHOLD:
                continue

            bbox = frame_norm(frame_shape, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            position = (bbox[0] + bbox[2]) / 2  # Center x position - maybe add also center y position

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
                best_match.bbox = bbox
                best_match.confidence = detection.confidence
                best_match.size_history.append(size)
                best_match.position_history.append(position)
                best_match.last_update = current_frame

                # Keep history at fixed size
                if len(best_match.size_history) > self.max_history:
                    best_match.size_history.popleft()
                if len(best_match.position_history) > self.max_history:
                    best_match.position_history.popleft()

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
                    size_history=deque([size], maxlen=self.max_history),
                    position_history=deque([position], maxlen=self.max_history),
                    last_update=current_frame
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