#!/usr/bin/env python
import torch

if __name__ == '__main__':
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Training will run on the CPU.")