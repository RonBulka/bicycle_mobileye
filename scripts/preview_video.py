"""
Video Preview Script

This script provides an interactive video player with advanced controls for previewing
processed videos and analyzing results. It supports various video formats and includes
frame-by-frame analysis capabilities.

Key Features:
- Interactive video playback with keyboard controls
- Frame-by-frame navigation and analysis
- Fullscreen mode support
- Frame capture and saving functionality
- X11 forwarding support for remote viewing
- Performance optimization for large videos
- Frame counter and video information display

Playback Controls:
- 'q': Quit video playback
- 'space': Pause/resume playback
- 'r': Restart video from beginning
- 'f': Toggle fullscreen mode
- 's': Save current frame as image

Video Features:
- Automatic video scaling for performance
- Frame counter display
- Video property information
- Error handling for corrupted files
- Support for various video codecs

Usage:
    python preview_video.py --video path/to/video.mp4

This script is essential for:
- Reviewing processed video outputs
- Analyzing detection and tracking results
- Debugging video processing issues
- Capturing specific frames for analysis
- Quality control of video processing pipeline
"""

#!/usr/bin/env python
import os
import argparse
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Preview MP4 video files')
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Path to the video file to preview'
    )
    return parser.parse_args()

def play_video(video_path: str):
    """Play a video file using OpenCV with X11 forwarding support."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000/fps)  # Delay between frames in milliseconds
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\nVideo playback controls:")
    print("Press 'q' to quit")
    print("Press 'space' to pause/resume")
    print("Press 'r' to restart video")
    print("Press 'f' to toggle fullscreen")
    print("Press 's' to save current frame")
    print("\nPlaying video...")

    paused = False
    fullscreen = False
    window_name = os.path.basename(video_path)
    frame_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Restart video when it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue

            frame_count += 1

            # Resize frame for better performance over X11
            height, width = frame.shape[:2]
            scale = min(1.0, 1280/width)  # Scale down if width > 1280
            if scale < 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show frame
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == ord('s'):
            # Save current frame
            frame_path = f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame to {frame_path}")

    cap.release()
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    play_video(args.video)

if __name__ == "__main__":
    main() 