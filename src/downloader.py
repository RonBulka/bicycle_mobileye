#!/usr/bin/env python
import fiftyone as fo
import os
import argparse
from constansts import TRAIN_SAMPLES, VAL_SAMPLES

def export_only_images(cwd, label_type, classes, train_samples, val_samples):
    train_output_dir = os.path.join(cwd, "images/train")
    val_output_dir = os.path.join(cwd, "images/val")
    if not os.path.isdir(train_output_dir):
        os.makedirs(train_output_dir, exist_ok=True)
    if not os.path.isdir(val_output_dir):
        os.makedirs(val_output_dir, exist_ok=True)

    # Load the datasets
    dataset_train = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=[label_type],
        classes=classes,
        max_samples=train_samples,
    )

    dataset_val = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=[label_type],
        classes=classes,
        max_samples=val_samples,
    )

    # Export the images to the folder
    dataset_train.export(
        export_dir=train_output_dir,
        dataset_type=fo.types.ImageDirectory,
        label_field=None,  # Optional: Exclude labels if not needed
    )

    dataset_val.export(
        export_dir=val_output_dir,
        dataset_type=fo.types.ImageDirectory,
        label_field=None,  # Optional: Exclude labels if not needed
    )

def export_images_and_labels(export_dir, label_type, splits, classes, train_samples, val_samples):
    dataset = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        splits=["train"],
        label_types=[label_type],
        classes=classes,
        max_samples=train_samples,
    )
    val_dataset = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        splits=["validation"],
        label_types=[label_type],
        classes=classes,
        max_samples=val_samples,
    )

    # Combine or process each dataset as needed
    dataset.add_samples(val_dataset)

    # Convert all class labels to "vehicle"
    for sample in dataset:
        detections = sample.ground_truth.detections
        for detection in detections:
            if detection.label in classes:
                detection.label = "vehicle"
        sample.save()

    # Export separately per split
    for split in splits:
        split_view = dataset.match_tags([split])
        split_name = "train" if split == "train" else "val"
        max_samples = train_samples if split == "train" else val_samples
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            split=split_name,
            label_field=None,
            classes=["vehicle"],
            max_samples=max_samples,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Download and export datasets.")
    parser.add_argument(
        "--export_labels", "-el",
        action='store_true',
        help="Flag to also export labels"
    )
    parser.add_argument(
        "--label_type", "-l",
        type=str,
        default="detections",
        help="Label type for exporting images and labels (default: 'detections')"
    )
    parser.add_argument(
        "--splits", "-s",
        nargs='+',
        default=["train", "validation"],
        help="Dataset splits to export (default: ['train', 'validation'])"
    )
    parser.add_argument(
        "--classes", "-c",
        nargs='+',
        default=["Car", "Truck", "Bus", "Motorcycle", "Van", "Ambulance"],
        help="Classes to include in the export (default: ['Car', 'Truck', 'Bus', 'Motorcycle', 'Van', 'Ambulance'])"
    )
    parser.add_argument(
        "--train_samples", "-ts",
        type=int,
        default=TRAIN_SAMPLES,
        help=f"Number of training samples to export (default: {TRAIN_SAMPLES})"
    )
    parser.add_argument(
        "--val_samples", "-vs",
        type=int,
        default=VAL_SAMPLES,
        help=f"Number of validation samples to export (default: {VAL_SAMPLES})"
    )
    parser.add_argument(
        "--export_dir", "-ed",
        type=str,
        default="dataset",
        help="Directory to export the dataset (default: [cwd]/dataset)"
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    export_dir = os.path.join(os.getcwd(), args.export_dir)
    if not args.export_labels:
        export_only_images(export_dir, args.label_type, args.classes, args.train_samples, args.val_samples)
    elif args.export_labels:
        export_images_and_labels(export_dir, args.label_type, args.splits, args.classes, args.train_samples, args.val_samples)
