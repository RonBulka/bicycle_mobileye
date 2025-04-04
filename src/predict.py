#!/usr/bin/env python
import os
import argparse
import cv2
import torch
from ultralytics import  YOLO

CONFIDENCE_THRESHOLD = 0.7

def parse_args():
    parser = argparse.ArgumentParser(description='Run YOLO object detection on video')
    parser.add_argument('--model', '-m', type=str, default='./runs/train2/weights/last.pt',
                        help='Path to YOLO model weights')
    parser.add_argument('--input_dir', '-id', type=str, default='./evaluation_vids/input/',
                        help='Path to input video directory')
    parser.add_argument('--output_dir', '-od', type=str, default='./evaluation_vids/output/',
                        help='Path to output video directory')
    parser.add_argument('--input_name', '-in', type=str, default='4p9Zk12iE8s.mp4',
                        help='Name of input video')
    parser.add_argument('--output_name', '-on', type=str, default='out.mp4',
                        help='Name of output video')
    parser.add_argument('--confidence', '-c', type=float, default=CONFIDENCE_THRESHOLD,
                        help='Confidence threshold for object detection')
    return parser.parse_args()

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Iterate over each frame
    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()  # Read a frame
        if not ret:
            break

        # Apply YOLOv8 object detection
        results = model(frame)[0]

        # Iterate through the detections and draw bounding boxes
        for result in results.boxes.data.tolist():  # Each detection in the format [x1, y1, x2, y2, conf, class]
            x1, y1, x2, y2, conf, cls = result[:6]
            label = f'{model.names[cls]} {conf:.2f}'

            # Draw bounding box and label on the frame
            if conf > args.confidence: 
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Bounding box
            # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write the processed frame to the output video
        out_video.write(frame)

        # Print progress
        frame_count += 1
        print(f'Processed frame {frame_count}/{total_frames}')

    # Release resources
    video_capture.release()
    out_video.release()
    cv2.destroyAllWindows()

    print(f'Output video saved to {output_video_path}')

if __name__ == "__main__":
    args = parse_args()
    main(args)
