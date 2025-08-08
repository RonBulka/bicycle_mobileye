"""
GPU Detection Script

This script checks for CUDA GPU availability and provides information about available
graphics processing units for deep learning training and inference.

Key Features:
- CUDA availability detection
- GPU device enumeration and information
- Device name and capability reporting
- Fallback to CPU when no GPU is available
- PyTorch CUDA integration verification

Output Information:
- CUDA availability status
- Number of available GPUs
- Device names and specifications
- Training device recommendations

Usage:
    python check_gpu.py

This script is essential for:
- Verifying GPU setup for training
- Debugging CUDA installation issues
- Determining optimal training configuration
- System compatibility checking
- Performance optimization planning

The script helps ensure optimal performance for YOLO model training
and inference by identifying available computational resources.
"""

#!/usr/bin/env python
import torch

if __name__ == '__main__':
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU detected. Training will run on the CPU.")