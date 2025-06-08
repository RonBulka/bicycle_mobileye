#!/usr/bin/env python
import os
import blobconverter
import json
from ultralytics import YOLO
from constants import IMAGE_SIZE
import onnx
from onnxsim import simplify

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
onnx_model = onnx.load(onnx_path)
model_simplified, check = simplify(onnx_model)
onnx_path = onnx_path.replace('.onnx', '_simplified.onnx')
onnx.save(model_simplified, onnx_path)

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
