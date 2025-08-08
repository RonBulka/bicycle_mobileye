"""
OAK-D Camera Deployment Script

This script deploys trained YOLO models on OAK-D Lite cameras for real-time vehicle detection
and tracking. It provides a complete edge computing solution for bicycle safety applications.

Key Features:
- Real-time video processing on OAK-D Lite hardware
- YOLO model deployment using DepthAI framework
- Vehicle tracking with collision warning system
- Audio warning integration for dangerous vehicles
- Video recording with annotated output
- Test mode with live visualization
- Optimized for edge computing performance

Hardware Requirements:
- OAK-D Lite camera
- Compatible host system (Linux/Windows)
- DepthAI SDK installation

Deployment Process:
1. Loads trained YOLO model in blob format
2. Configures OAK-D camera pipeline
3. Sets up real-time detection and tracking
4. Processes video stream with vehicle tracking
5. Generates warnings for approaching vehicles
6. Records annotated video output

The script is designed for production deployment on bicycle-mounted systems
for real-time collision avoidance and safety monitoring.
"""

#!/usr/bin/env python
import os
import json
import cv2
import depthai as dai
import time
import argparse
from vehicle_tracker import VehicleTracker, annotate_frame
from constants import CAMERA_PREVIEW_DIM, CONFIDENCE_THRESHOLD

# Current working directory
cwd = os.getcwd()

# Define paths to the model, test data directory, and results
YOLO_MODEL = os.path.join(cwd, "luxonis_output/last_openvino_2022.1_6shave.blob")

# Input and output video paths
OUTPUT_VIDEO = "vid_result/test_video.mp4"

def create_camera_pipeline(model_path):
    """Creates a pipeline for the camera and YOLO detection using DepthAI v3 API."""
    # Create pipeline
    pipeline = dai.Pipeline()

    # Create nodes
    cam = pipeline.create(dai.node.ColorCamera)
    detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_video = pipeline.create(dai.node.XLinkOut)

    # Set output streams
    xout_nn.setStreamName("nn")
    xout_video.setStreamName("video")

    # Configure camera
    cam.setPreviewSize(CAMERA_PREVIEW_DIM[0], CAMERA_PREVIEW_DIM[1])  # Set preview to match model input
    cam.setVideoSize(CAMERA_PREVIEW_DIM[0], CAMERA_PREVIEW_DIM[1])    # Set video output to match model input
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Configure detection network
    detection_nn.setBlobPath(model_path)
    detection_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    detection_nn.setNumClasses(1)  # For vehicle detection
    detection_nn.setCoordinateSize(4)
    detection_nn.setIouThreshold(0.5)
    detection_nn.setNumInferenceThreads(2)
    detection_nn.input.setBlocking(False)

    # Linking
    cam.preview.link(detection_nn.input)
    cam.video.link(xout_video.input)
    detection_nn.out.link(xout_nn.input)

    return pipeline

def load_config(config_path):
    """Loads configuration from a JSON file."""
    with open(config_path) as f:
        return json.load(f)

def main(args):
    # Create pipeline
    pipeline = create_camera_pipeline(YOLO_MODEL)

    # Initialize vehicle tracker
    vehicle_tracker = VehicleTracker()

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Get output queues
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)

        # Video writer setup
        fps = 15
        frame_width, frame_height = CAMERA_PREVIEW_DIM
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

        start_time = time.time()
        frame_count = 0

        try:
            while True:
                # Get detection results
                in_nn = q_nn.get()
                detections = in_nn.detections if in_nn is not None else []

                # Get video frame
                in_video = q_video.get()
                if in_video is not None:
                    frame = in_video.getCvFrame()

                    # Update vehicle tracking with detections
                    tracked_vehicles = vehicle_tracker.update(
                        detections=detections,
                        frame_shape=frame,
                        frame_number=frame_count,
                        fps=fps
                    )

                    # Annotate the frame with tracked vehicles
                    annotated_frame = annotate_frame(frame, tracked_vehicles, fps)

                    # Write the annotated frame to the output video
                    out.write(annotated_frame)

                    # Update frame count and calculate FPS
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    
                    if args.test:
                        # Display the annotated frame
                        cv2.imshow("Frame", annotated_frame)
                        # Exit on key press
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        # In production mode, just continue processing
                        # User can stop with Ctrl+C
                        pass

        except KeyboardInterrupt:
            print("\nStopping video capture...")
        finally:
            # Release resources
            out.release()
            vehicle_tracker.warning_manager.stop_audio_thread()
            if args.test:
                cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy model on OAK-D lite camera")
    parser.add_argument(
        "--test", "-t",
        action='store_true',
        help="Flag to run in test mode"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
