"""
Model Conversion Script for OAK-D Deployment

This script converts trained YOLO models to blob format for deployment on OAK-D Lite cameras.
It handles the complete conversion pipeline from PyTorch to OpenVINO to blob format.

Key Features:
- Converts PyTorch YOLO models to ONNX format
- Optimizes ONNX models using onnx-simplifier
- Converts optimized ONNX to blob format for OAK-D
- Generates configuration JSON files for DepthAI
- Supports FP16 precision for edge computing
- Configurable shave count for OAK-D hardware

Conversion Pipeline:
1. Load PyTorch model (.pt file)
2. Export to ONNX format
3. Simplify ONNX model for optimization
4. Convert to blob format using blobconverter
5. Generate configuration JSON for DepthAI

Hardware Requirements:
- OAK-D Lite camera compatibility
- OpenVINO 2022.1 support
- 6 shave configuration

Output Files:
- last_simplified.onnx: Optimized ONNX model
- last_openvino_2022.1_6shave.blob: Deployable blob file
- last.json: DepthAI configuration file

This script is essential for deploying trained models on edge devices for real-time
vehicle detection and tracking applications.
"""

#!/usr/bin/env python
import os
import blobconverter
import json
from ultralytics import YOLO
from constants import IMAGE_SIZE
import onnx

# Image size
img_size = IMAGE_SIZE

# Define paths
model_name = "last"
model_dir = os.path.join(os.getcwd(), 'runs/yolov8n_640sz_6000t_1500v/weights')
model_path = os.path.join(model_dir, f'{model_name}.pt')

# Load the model
print("=========================================================================================================================")
print(f"Loading YOLO model from {model_path}")
model = YOLO(model_path)

# Export to ONNX first
print("========================================================================================================================")
print(f"Exporting model to ONNX format...")
onnx_path = model_path.replace('.pt', '.onnx')
success = model.export(format="onnx", imgsz=img_size, opset=10)

# Optimize the model
print("========================================================================================================================")
print(f"Optimizing model...")
onnx_path = onnx_path.replace('.onnx', '_simplified.onnx')
onnx.save(model, onnx_path)

# Convert to blob
print("========================================================================================================================")
print("Converting to blob format...")
blob_path = blobconverter.from_onnx(
    model=onnx_path,
    output_dir=model_path.replace('.pt', '_blob'),
    data_type="FP16",
    shaves=6,
    use_cache=False,
    optimizer_params=[]
)

print(f"Blob file saved to: {blob_path}")

# Generate configuration JSON
config = {
    "nn_config": {
        "output_format": "detection",
        "NN_family": "YOLO",
        "input_size": f"{img_size}x{img_size}",
        "NN_specific_metadata": {
            "classes": 1,
            "coordinates": 4,
            "anchors": [],
            "anchor_masks": {},
            "iou_threshold": 0.5,
            "confidence_threshold": 0.5
        }
    },
    "mappings": {
        "labels": [
            "vehicle"
        ]
    },
    "version": 1
}

# Save the configuration JSON
config_path = os.path.join(os.path.dirname(blob_path), f"{model_name}.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"Configuration JSON saved to: {config_path}")
