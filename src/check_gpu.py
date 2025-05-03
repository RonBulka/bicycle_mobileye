#!/usr/bin/env python
import torch

if __name__ == '__main__':
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU detected. Training will run on the CPU.")