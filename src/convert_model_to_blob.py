#!/usr/bin/env python
import os
import subprocess
import cv2
from ultralytics import YOLO
import torch
import openvino as ov
import blobconverter

# use https://tools.luxonis.com/ to convert the model to blob
onnx_name = 'model.onnx'
img_size = 640  # Image size

# Define paths
model_dir = os.path.join(os.getcwd(), 'runs/yolov8n_640sz_6000t_1500v')
onnx_path = os.path.join(model_dir, onnx_name)
output_dir = os.path.join(model_dir, "openvino")
xml_path = os.path.join(output_dir, "model.xml")
bin_path = os.path.join(output_dir, "model.bin")

# Change directory
os.chdir(model_dir)

# Load the model
model = YOLO(os.path.join(model_dir, 'last.pt'))

dummy_input = torch.randn(1, 3, img_size, img_size)
 
torch.onnx.export(
           model.model,
           dummy_input,
           str(onnx_path),  # Convert Path to str
           opset_version = 11)

 
ir_path = onnx_path.replace('.onnx', '.xml')
 
if not os.path.exists(ir_path):
   print("Exporting ONNX model to IR... This may take a few minutes.")
   ov_model = ov.convert_model(onnx_path)
   ov.save_model(ov_model, ir_path)
else:
   print(f"IR model {ir_path} already exists.")

# # Export to ONNX with opset10
# torch.onnx.export(model.model, torch.randn(1, 3, img_size, img_size), onnx_name, opset_version=9)


# # Create output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Convert ONNX to OpenVINO IR using openvino.convert_model
# ov_model = ov.convert_model(onnx_path)
# ov.save_model(model=ov_model, output_model=xml_path, compress_to_fp16=True)

# Convert OpenVINO IR to blob
# blob_path = blobconverter.from_openvino(
#     xml=xml_path,
#     bin=bin_path,
#     data_type="FP16",
#     shaves=6
# )

# print(f"Blob file saved to: {blob_path}")

