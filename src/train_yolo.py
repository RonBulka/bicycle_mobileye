#!/usr/bin/env python
"""
YOLO Model Training Script

This script trains YOLO models for vehicle detection using transfer learning from pretrained weights.
It supports custom datasets and provides comprehensive training configuration options.

Key Features:
- Transfer learning from pretrained YOLO models (YOLOv8, YOLOv10, YOLOv11)
- Customizable training parameters (epochs, batch size, image size)
- GPU/CPU detection and automatic device selection
- Support for custom dataset configurations
- Training progress monitoring and logging
- Model checkpointing and best weight saving

Training Process:
1. Loads pretrained model weights
2. Configures training parameters from constants or command line
3. Trains on custom vehicle dataset
4. Saves best model weights and training artifacts
5. Generates training metrics and visualizations

The trained models are optimized for bicycle safety applications, detecting
vehicles that pose collision risks to cyclists.
"""

import torch
import argparse
from ultralytics import YOLO
from .constants import EPOCHS, \
                       BATCH_SIZE, \
                       IMAGE_SIZE, \
                       CONFIG, \
                       MODEL, \
                       OUTPUT_DIR

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

    # train the model
    model.train(
        data=args.config,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.output
    )
