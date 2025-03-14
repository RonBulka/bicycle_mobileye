#!/usr/bin/env python
import os
import json
import cv2
import depthai as dai
import numpy as np
import time

# Current working directory
cwd = os.getcwd()

# Define paths to the model, test data directory, and results
YOLO_MODEL = os.path.join(cwd, "luxonis_output/last_openvino_2022.1_6shave.blob")
YOLO_CONFIG = os.path.join(cwd, "luxonis_output/last.json")

# Input and output video paths
OUTPUT_VIDEO = "vid_result/test_video.mp4"

# Camera preview dimensions
CAMERA_PREVIEW_DIM = (640, 640)

# Labels for detected objects
LABELS = ["Vehicle"]

def load_config(config_path):
    """Loads configuration from a JSON file."""
    with open(config_path) as f:
        return json.load(f)

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

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """Resizes an array to a specified shape and transposes it."""
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1)

def frame_norm(frame, bbox):
    """Normalizes bounding box coordinates to frame dimensions."""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

def annotate_frame(frame, detections, fps):
    """Annotates a frame with detections and FPS."""
    color = (0, 0, 255)
    for detection in detections:
        bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, LABELS[detection.label], (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # Annotate the frame with the FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Create pipeline
pipeline = create_camera_pipeline(YOLO_CONFIG, YOLO_MODEL)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Define the queues that will be used to receive the neural network output and video frames
    detectionNN = device.getOutputQueue("nn", maxSize=4, blocking=False)
    videoQueue = device.getOutputQueue("video", maxSize=4, blocking=False)

    # Video writer to save the output video
    fps = 30  # Assuming 30 FPS for the OAK-D camera
    frame_width, frame_height = CAMERA_PREVIEW_DIM
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    start_time = time.time()
    frame_count = 0

    while True:
        inDet = detectionNN.get()
        detections = []
        if inDet is not None:
            detections = inDet.detections
            print("Detections", detections)

        # Retrieve the frame from the camera preview
        inVideo = videoQueue.get()
        if inVideo is not None:
            frame = inVideo.getCvFrame()  # Use getCvFrame for OpenCV format

            # Annotate the frame with detections
            annotated_frame = annotate_frame(frame, detections, fps)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Update frame count and calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            # Display the annotated frame
            cv2.imshow("Frame", annotated_frame)

            # Exit on key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    out.release()
    cv2.destroyAllWindows()