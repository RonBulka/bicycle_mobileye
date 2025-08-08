#!/usr/bin/env python
"""
YOLO Video Prediction Script

This script performs real-time vehicle detection and tracking on video files using a trained YOLO model.
It integrates with the vehicle tracker to provide collision warnings and safety alerts for bicycle riders.

Key Features:
- Video processing with customizable input/output paths
- Real-time vehicle detection using trained YOLO models
- Vehicle tracking with collision warning system
- Audio warning integration for approaching vehicles
- Video playback with interactive controls
- Support for both custom-trained and pretrained models
- Batch processing of multiple video files
- Performance metrics and FPS monitoring

Usage:
    python predict.py --model path/to/model.pt --input_name video.mp4 --output_name output.mp4

The script processes videos through the complete pipeline:
1. YOLO object detection
2. Vehicle tracking with Kalman filtering
3. Speed and TTC calculation
4. Warning generation for dangerous vehicles
5. Video annotation and output generation
"""

import os
import sys
import argparse
import cv2
import torch
from ultralytics import YOLO

# Add the project root to Python path to enable imports from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.vehicle_tracker import VehicleTracker, annotate_frame
from src.constants import CONFIDENCE_THRESHOLD
import time
import numpy as np
from pydub import AudioSegment
import tempfile
import subprocess
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Run YOLO object detection on video')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='./runs/yolov8n_640sz_6000t_1500v/weights/last.pt',
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

def get_video_audio(input_video_path: str) -> AudioSegment:
    """Return empty audio segment."""
    # try:
    #     return AudioSegment.from_file(input_video_path)
    # except:
    #     # Return empty audio segment if no audio track exists
    return AudioSegment.silent(duration=0)

def create_warning_audio(warning_manager, duration_ms: int) -> AudioSegment:
    """Create audio segment from warning sound."""
    warning_sound_path = "audio/clock-alarm-8762.mp3"
    if not os.path.exists(warning_sound_path):
        return AudioSegment.silent(duration=duration_ms)
    
    warning_sound = AudioSegment.from_file(warning_sound_path)
    # Repeat the warning sound to match video duration
    repeats = int(np.ceil(duration_ms / len(warning_sound)))
    return warning_sound * repeats

def combine_audio_video(video_path: str, audio_path: str, output_path: str):
    """Combine video and audio using ffmpeg."""
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    try:
        # Use ffmpeg to combine video and audio
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            temp_output
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Copy the file instead of moving it
        shutil.copy2(temp_output, output_path)
    finally:
        if os.path.exists(temp_output):
            os.unlink(temp_output)

def get_video_properties(video_path: str) -> dict:
    """Get comprehensive video properties including FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return {
            'fps': 30.0,
            'width': 640,
            'height': 480,
            'frame_count': 0,
            'duration': 0
        }
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps if fps > 0 else 30.0,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration': duration
    }

def main(args):
    # Load the YOLOv8 model using command line argument
    model = YOLO(args.model)

    # Load the video file
    cwd = os.getcwd()
    input_video_path = os.path.join(cwd, args.input_dir, args.input_name)
    output_video_path = os.path.join(cwd, args.output_dir, args.output_name)
    temp_video_path = os.path.join(cwd, args.output_dir, 'temp_' + args.output_name)

    # Open the video using OpenCV
    video_capture = cv2.VideoCapture(input_video_path)

    # Get video properties
    video_props = get_video_properties(input_video_path)
    frame_width = video_props['width']
    frame_height = video_props['height']
    fps = video_props['fps']
    total_frames = video_props['frame_count']
    video_duration_ms = int((total_frames / fps) * 1000)
    
    print(f"Video properties: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames, {video_duration_ms/1000:.2f}s duration")

    # Define the codec and create VideoWriter object to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Check if VideoWriter was initialized successfully
    if not out_video.isOpened():
        print(f"Error: Could not create output video file at {temp_video_path}")
        print("Please check if the output directory exists and you have write permissions")
        return

    # Initialize vehicle tracker
    vehicle_tracker = VehicleTracker()

    # Track warning states for audio generation
    warning_states = []

    # Iterate over each frame
    frame_count = 0
    start_time = time.time()
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Apply YOLOv8 object detection
        results = model(frame)[0]

        # Convert YOLO detections to our format
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result[:6]
            if conf > args.confidence:
                x1, x2 = x1 / frame_width, x2 / frame_width
                y1, y2 = y1 / frame_height, y2 / frame_height
                detections.append(YOLODetection(x1, y1, x2, y2, conf, int(cls)))

        # Update vehicle tracking
        tracked_vehicles = vehicle_tracker.update(detections, frame, frame_count, fps)
        
        # Record warning state for this frame
        warning_states.append(any(vehicle.warning for vehicle in tracked_vehicles))

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
    vehicle_tracker.warning_manager.stop_audio_thread()
    if args.test:
        cv2.destroyAllWindows()

    # Create audio tracks
    print("Creating audio tracks...")
    input_audio = get_video_audio(input_video_path)
    warning_audio = create_warning_audio(vehicle_tracker.warning_manager, video_duration_ms)
    
    # Create warning audio track based on recorded states
    warning_track = AudioSegment.silent(duration=video_duration_ms)
    frame_duration_ms = int(1000 / video_props['fps'])
    
    for i, has_warning in enumerate(warning_states):
        if has_warning:
            start_ms = i * frame_duration_ms
            end_ms = start_ms + frame_duration_ms
            warning_track = warning_track.overlay(
                warning_audio[start_ms:end_ms],
                position=start_ms
            )

    # Mix input audio with warning audio
    if len(input_audio) > 0:
        final_audio = input_audio.overlay(warning_track)
    else:
        final_audio = warning_track

    # Save audio to temporary file
    temp_audio_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    final_audio.export(temp_audio_path, format='wav')

    # Combine video and audio
    print("Combining video and audio...")
    combine_audio_video(temp_video_path, temp_audio_path, output_video_path)

    # Clean up temporary files
    os.unlink(temp_video_path)
    os.unlink(temp_audio_path)


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
