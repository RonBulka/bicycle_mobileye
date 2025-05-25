#!/usr/bin/env python
import os
import argparse
import cv2
import torch
from ultralytics import YOLO
from vehicle_tracker import VehicleTracker, annotate_frame
from constansts import CONFIDENCE_THRESHOLD
import time

# COCO dataset vehicle class IDs
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    1: 'bicycle'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run YOLO object detection on video')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolov8n.pt',  # Changed default to use pretrained model
        help='Path to YOLO model weights'
    )
    parser.add_argument(
        '--input_dir', '-id',
        type=str,
        default='./evaluation_vids/input/',
        help='Path to input video directory'
    )
    parser.add_argument(
        '--output_dir', '-od',
        type=str,
        default='./evaluation_vids/output/',
        help='Path to output video directory'
    )
    parser.add_argument(
        '--input_name', '-in',
        type=str,
        default='input1.mp4',
        help='Name of input video'
    )
    parser.add_argument(
        '--output_name', '-on',
        type=str,
        default='out1.mp4',
        help='Name of output video'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help='Confidence threshold for object detection'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Flag to run in test mode with visualization'
    )
    parser.add_argument(
        '--play', '-p',
        action='store_true',
        help='Play the output video after processing'
    )
    return parser.parse_args()

class YOLODetection:
    """Wrapper class to make YOLO detection results compatible with our tracker"""
    def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float, class_id: int):
        self.xmin = x1
        self.ymin = y1
        self.xmax = x2
        self.ymax = y2
        self.confidence = confidence
        self.label = class_id

def play_video(video_path: str):
    """Play a video file using OpenCV with X11 forwarding support."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000/fps)  # Delay between frames in milliseconds

    print("\nVideo playback controls:")
    print("Press 'q' to quit")
    print("Press 'space' to pause/resume")
    print("Press 'r' to restart video")
    print("Press 'f' to toggle fullscreen")
    print("\nPlaying video...")

    paused = False
    fullscreen = False
    window_name = 'Processed Video'

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Restart video when it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize frame for better performance over X11
            height, width = frame.shape[:2]
            scale = min(1.0, 1280/width)  # Scale down if width > 1280
            if scale < 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Show frame
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

def main(args):
    # Load the YOLOv8 model using command line argument
    model = YOLO(args.model)

    # Load the video file
    cwd = os.getcwd()
    input_video_path = os.path.join(cwd, args.input_dir, args.input_name)
    output_video_path = os.path.join(cwd, args.output_dir, args.output_name)

    # Open the video using OpenCV
    video_capture = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec for better compatibility
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Check if VideoWriter was initialized successfully
    if not out_video.isOpened():
        print(f"Error: Could not create output video file at {output_video_path}")
        print("Please check if the output directory exists and you have write permissions")
        return

    # Initialize vehicle tracker
    vehicle_tracker = VehicleTracker()

    # Iterate over each frame
    frame_count = 0
    start_time = time.time()
    while video_capture.isOpened():
        ret, frame = video_capture.read()  # Read a frame
        if not ret:
            break

        # Apply YOLOv8 object detection
        results = model(frame)[0]

        # Convert YOLO detections to our format
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result[:6]
            class_id = int(cls)
            # Only process vehicle detections
            if conf > args.confidence and class_id in VEHICLE_CLASSES:
                # Normalize coordinates to [0,1] range
                x1, x2 = x1 / frame_width, x2 / frame_width
                y1, y2 = y1 / frame_height, y2 / frame_height
                detections.append(YOLODetection(x1, y1, x2, y2, conf, class_id))

        # Update vehicle tracking
        tracked_vehicles = vehicle_tracker.update(detections, frame, frame_count, fps)

        # Annotate the frame with tracked vehicles
        annotated_frame = annotate_frame(frame, tracked_vehicles, fps)

        # Write the processed frame to the output video
        out_video.write(annotated_frame)

        # Display frame if in test mode
        if args.test:
            cv2.imshow('Frame', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print progress
        frame_count += 1
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time
        fps = current_fps
        print(f'Processed frame {frame_count}/{total_frames} (FPS: {current_fps:.2f})')

    # Release resources
    video_capture.release()
    out_video.release()
    if args.test:
        cv2.destroyAllWindows()

    # Verify the output file exists and has content
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path)
        if file_size > 0:
            print(f'Output video saved successfully to {output_video_path} (Size: {file_size/1024/1024:.2f} MB)')
        else:
            print(f'Warning: Output video file was created but is empty (0 bytes)')
    else:
        print(f'Error: Output video file was not created at {output_video_path}')

    # Play the output video if requested
    if args.play:
        play_video(output_video_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
