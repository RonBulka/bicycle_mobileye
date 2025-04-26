#!/usr/bin/env python
import os
import blobconverter
from ultralytics import YOLO

# Image size
img_size = 640

# Define paths
model_dir = os.path.join(os.getcwd(), 'runs/yolov8n_640sz_6000t_1500v/weights')
model_path = os.path.join(model_dir, 'last.pt')

# Load the model
print(f"Loading YOLO model from {model_path}")
model = YOLO(model_path)

# Export directly to OpenVINO format
print(f"Exporting model to OpenVINO format...")
success = model.export(format="openvino", imgsz=img_size)

# The export function returns the path to the exported model
openvino_dir = model_path.replace('.pt', '_openvino_model')
xml_path = os.path.join(openvino_dir, "model.xml")
bin_path = os.path.join(openvino_dir, "model.bin")

print(f"OpenVINO model exported to: {openvino_dir}")

# Convert to blob
print("Converting to blob format...")
blob_path = blobconverter.from_openvino(
    xml=xml_path,
    bin=bin_path,
    data_type="FP16",
    shaves=6,
    version="2022.1",
    use_cache=False
)

print(f"Blob file saved to: {blob_path}")

