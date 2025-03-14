#!/usr/bin/env python
import os
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Training will run on the CPU.")

    config_path = './dataset/dataset.yaml'

    # Load a pre trained model
    model = YOLO("yolov8n.pt")

    # Set the device to the ROCm-supported GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # train the model
    model.train(data=config_path, epochs=200, batch=32, imgsz=640, device=device, project="./runs")