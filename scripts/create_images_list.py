"""
Image List Creation Script

This script creates training and validation image lists for YOLO training by scanning
directories and generating text files with image paths. It's used to prepare dataset
configuration files for YOLO training.

Key Features:
- Scans directories for image files (jpg, jpeg, png, bmp)
- Generates train/validation image lists
- Supports relative path formatting for YOLO training
- Automatic file sorting for consistent output
- Validation of input directories and file types
- Command-line interface with customizable parameters

Supported Image Formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

Output Format:
- Text file with one image path per line
- Relative paths with train/val prefix
- Example: train/image001.jpg

Usage:
    python create_images_list.py --input-dir dataset/images/train --output-type train --output-dir dataset

This script is essential for preparing YOLO dataset configuration files that specify
which images to use for training and validation phases.
"""

#!/usr/bin/env python
import os
import argparse
from pathlib import Path

def create_images_list(input_dir: str, output_type: str, output_dir: str):
    # Validate output type
    if output_type not in ['train', 'valid']:
        raise ValueError("Output type must be either 'train' or 'valid'")
    
    # Set the prefix based on output type
    if output_type == 'train':
        prefix = "train"
    else:
        prefix = "val"

    # Get all image files from the directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    
    for file in os.listdir(input_dir):
        if file.lower().endswith(image_extensions):
            image_files.append(f"{prefix}/{file}")
    
    # Sort files to ensure consistent output
    image_files.sort()
    
    # Create output filename
    output_file = f"{output_dir}/{output_type}.txt"
    
    # Write to file
    with open(output_file, 'w') as f:
        for image_path in image_files:
            f.write(f"{image_path}\n")
    
    print(f"Created {output_file} with {len(image_files)} images")

def main():
    parser = argparse.ArgumentParser(description='Create a list of images for training or validation')
    parser.add_argument("--input-dir", "-i", type=str, help='Path to directory containing images')
    parser.add_argument("--output-type", "-o", type=str, choices=['train', 'valid'], 
                      help='Type of output file (train or valid)')
    parser.add_argument("--output-dir", "-d", type=str, help='Path to directory containing output files')
    
    args = parser.parse_args()
    
    # Convert input path to absolute path
    input_dir = os.path.abspath(args.input_dir)
    
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    create_images_list(input_dir, args.output_type, args.output_dir)

if __name__ == "__main__":
    main()
