#!/usr/bin/env python
import os
import json
import cv2
import depthai as dai
import time
import argparse
from vehicle_tracker import VehicleTracker, annotate_frame
from constants import CAMERA_PREVIEW_DIM

# Current working directory
cwd = os.getcwd()

# Define paths to the model, test data directory, and results
YOLO_MODEL = os.path.join(cwd, "luxonis_output/last_openvino_2022.1_6shave.blob")
YOLO_CONFIG = os.path.join(cwd, "luxonis_output/last.json")

# Input and output video paths
OUTPUT_VIDEO = "vid_result/test_video.mp4"
def create_camera_pipeline(config_path, model_path):
    """Creates a pipeline for the camera and YOLO detection."""
    pipeline = dai.Pipeline()
    model_config = load_config(config_path)
    nnConfig = model_config.get("nn_config", {})
    metadata = nnConfig.get("NN_specific_metadata", {})
    classes = metadata.get("classes", {})
    coordinates = metadata.get("coordinates", {})
    anchors = metadata.get("anchors", {})
    anchorMasks = metadata.get("anchor_masks", {})
    iouThreshold = metadata.get("iou_threshold", {})
    confidenceThreshold = metadata.get("confidence_threshold", {})

    # Create camera node
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(CAMERA_PREVIEW_DIM[0], CAMERA_PREVIEW_DIM[1])
    camRgb.setVideoSize(CAMERA_PREVIEW_DIM[0], CAMERA_PREVIEW_DIM[1])
    camRgb.setInterleaved(False)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Use CAM_A instead of RGB
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    # Create detection network node
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    nnOut = pipeline.create(dai.node.XLinkOut)
    videoOut = pipeline.create(dai.node.XLinkOut)

    nnOut.setStreamName("nn")
    videoOut.setStreamName("video")

    detectionNetwork.setConfidenceThreshold(confidenceThreshold)
    detectionNetwork.setNumClasses(classes)
    detectionNetwork.setCoordinateSize(coordinates)
    detectionNetwork.setAnchors(anchors)
    detectionNetwork.setAnchorMasks(anchorMasks)
    detectionNetwork.setIouThreshold(iouThreshold)
    detectionNetwork.setBlobPath(model_path)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # Linking
    camRgb.preview.link(detectionNetwork.input)
    camRgb.video.link(videoOut.input)  # Link video output
    detectionNetwork.out.link(nnOut.input)

    return pipeline

def load_config(config_path):
    """Loads configuration from a JSON file."""
    with open(config_path) as f:
        return json.load(f)

def main(args):
    # Create pipeline
    pipeline = create_camera_pipeline(YOLO_CONFIG, YOLO_MODEL)

    # Initialize vehicle tracker
    vehicle_tracker = VehicleTracker()

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Define the queues that will be used to receive the neural network output and video frames
        detectionNN = device.getOutputQueue("nn", maxSize=4, blocking=False)
        videoQueue = device.getOutputQueue("video", maxSize=4, blocking=False)

        # Video writer to save the output video
        fps = 15  # Assuming 15 FPS for the OAK-D camera
        frame_width, frame_height = CAMERA_PREVIEW_DIM

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

        start_time = time.time()
        frame_count = 0

        try:
            while True:
                inDet = detectionNN.get()
                detections = []
                if inDet is not None:
                    detections = inDet.detections

                # Retrieve the frame from the camera preview
                inVideo = videoQueue.get()
                if inVideo is not None:
                    frame = inVideo.getCvFrame()

                    # Update vehicle tracking
                    tracked_vehicles = vehicle_tracker.update(detections, frame, frame_count, fps)

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
