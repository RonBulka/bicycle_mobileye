#!/usr/bin/env python
import os
import blobconverter
import subprocess
from ultralytics import YOLO
from constants import IMAGE_SIZE

# Image size
img_size = IMAGE_SIZE

# Define paths
model_name = "last"
model_dir = os.path.join(os.getcwd(), 'runs/yolov8n_640sz_6000t_1500v/weights')
model_path = os.path.join(model_dir, f'{model_name}.pt')

# Load the model
print(f"Loading YOLO model from {model_path}")
model = YOLO(model_path)

# Export to ONNX first
print(f"Exporting model to ONNX format...")
onnx_path = model_path.replace('.pt', '.onnx')
success = model.export(format="onnx", imgsz=img_size, opset=10)

# Convert ONNX to OpenVINO using model optimizer
print("================================================================================================")
print("Converting ONNX to OpenVINO format...")
openvino_dir = model_path.replace('.pt', '_openvino_model')
xml_path = os.path.join(openvino_dir, f"{model_name}.xml")
bin_path = os.path.join(openvino_dir, f"{model_name}.bin")

# Create directory if it doesn't exist
os.makedirs(openvino_dir, exist_ok=True)

# Use OpenVINO model optimizer
ovc_command = [
    "ovc",
    onnx_path,
    "--output_model", os.path.join(openvino_dir, f"{model_name}.xml"),
    "--compress_to_fp16",
    "--input", f"images[1,3,{img_size},{img_size}]",
    "--output", "output0"
]

subprocess.run(ovc_command, check=True)

print(f"OpenVINO model exported to: {openvino_dir}")

# Convert to blob
print("================================================================================================")
print("Converting to blob format...")
blob_path = blobconverter.from_openvino(
    xml=xml_path,
    bin=bin_path,
    data_type="FP16",
    shaves=6,
    version="2021.4",
    use_cache=False
)

print(f"Blob file saved to: {blob_path}")

