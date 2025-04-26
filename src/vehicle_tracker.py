from dataclasses import dataclass
from typing import Tuple, List, Dict
from collections import deque
import time
import numpy as np
import cv2
from constansts import CONFIDENCE_THRESHOLD, MAX_HISTORY, SIZE_THRESHOLD

@dataclass
class TrackedVehicle:
    id: int
    bbox: Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax
    confidence: float
    size_history: deque  # Store last N frame sizes
    position_history: deque  # Store last N frame positions
    last_update: float
    speed: float = 0.0
    ttc: float = float('inf')  # Time to collision in seconds

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
        
        # Calculate speed based on size change
        speed = size_change * 100  # Convert to percentage
        
        # Calculate time to collision (TTC)
        # TTC = current_size / (size_change_rate * current_size)
        if abs(size_change) > 0.001:  # Avoid division by very small numbers
            ttc = 1.0 / abs(size_change)
        else:
            ttc = float('inf')
            
        return speed, ttc
        
    def update(self, detections: List, frame_shape: Tuple[int, int]) -> List[TrackedVehicle]:
        """Update tracked vehicles with new detections."""
        current_time = time.time()
        self.frame_count += 1
        
        # Update existing tracks
        for vehicle_id, vehicle in list(self.tracked_vehicles.items()):
            # Remove old tracks
            if current_time - vehicle.last_update > 1.0:  # Remove if not updated for 1 second
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
            position = (bbox[0] + bbox[2]) / 2  # Center x position
            
            # Find best matching existing track
            best_match = None
            best_iou = 0.0
            
            for vehicle in self.tracked_vehicles.values():
                prev_bbox = vehicle.bbox
                iou = self.calculate_iou(bbox, prev_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = vehicle
            
            if best_match and best_iou > 0.3:  # Update existing track
                best_match.bbox = bbox
                best_match.confidence = detection.confidence
                best_match.size_history.append(size)
                best_match.position_history.append(position)
                best_match.last_update = current_time
                
                # Keep history at fixed size
                if len(best_match.size_history) > self.max_history:
                    best_match.size_history.popleft()
                if len(best_match.position_history) > self.max_history:
                    best_match.position_history.popleft()
            else:  # Create new track
                new_vehicle = TrackedVehicle(
                    id=self.next_id,
                    bbox=bbox,
                    confidence=detection.confidence,
                    size_history=deque([size], maxlen=self.max_history),
                    position_history=deque([position], maxlen=self.max_history),
                    last_update=current_time
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

def frame_norm(frame, bbox):
    """Normalizes bounding box coordinates to frame dimensions."""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

def annotate_frame(frame, tracked_vehicles, fps):
    """Annotates a frame with tracked vehicles and their metrics."""
    color = (0, 0, 255)
    for vehicle in tracked_vehicles:
        bbox = vehicle.bbox
        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Add vehicle ID and metrics
        text = f"ID: {vehicle.id}"
        cv2.putText(frame, text, (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        
        # Add speed and TTC
        speed_text = f"Speed: {vehicle.speed:.1f}%"
        ttc_text = f"TTC: {vehicle.ttc:.1f}s"
        cv2.putText(frame, speed_text, (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.putText(frame, ttc_text, (bbox[0] + 10, bbox[1] + 95), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        
        # Add warning if vehicle is approaching too fast
        if vehicle.ttc < 3.0 and vehicle.speed > 5.0:
            warning_text = "WARNING: Approaching vehicle!"
            cv2.putText(frame, warning_text, (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

    # Annotate the frame with the FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame