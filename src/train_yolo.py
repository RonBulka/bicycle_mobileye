#!/usr/bin/env python
import torch
import argparse
from ultralytics import YOLO

EPOCHS      = 200
BATCH_SIZE  = 32
IMAGE_SIZE  = 640
CONFIG      = './dataset/dataset.yaml'
MODEL       = 'yolov8n.pt'
OUTPUT_DIR  = './runs'

# add args to the script
def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=MODEL,
        help=f'Path to the pre-trained model, default is {MODEL}'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=EPOCHS,
        help=f'Number of epochs to train the model, default is {EPOCHS}'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size for training, default is {BATCH_SIZE}'
    )
    parser.add_argument(
        '--imgsz', '-i',
        type=int,
        default=IMAGE_SIZE,
        help=f'Image size for training, default is {IMAGE_SIZE}'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=CONFIG,
        help=f'Path to the dataset configuration file, default is {CONFIG}'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory for the training results, default is {OUTPUT_DIR}'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Training will run on the CPU.")

    # Load a pre trained model
    model = YOLO(args.model)

    # Set the device to the ROCm-supported GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # train the model
    model.train(data=args.config,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=device,
                project=args.output
    )
