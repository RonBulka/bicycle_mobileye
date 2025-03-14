#!/usr/bin/env python
import fiftyone as fo
import os
import argparse

def export_only_images(cwd, classes):
    train_output_dir = os.path.join(cwd, "images/train")
    val_output_dir = os.path.join(cwd, "images/val")
    if not os.path.isdir(train_output_dir) or not os.path.isdir(val_output_dir):
        print("Output directories do not exist. Please create them first.")
        return

    # Load the datasets
    dataset_train = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=classes,
        max_samples=1000,
    )

    dataset_val = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=classes,
        max_samples=200,
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

def export_images_and_labels(export_dir, label_type, splits, classes):
    dataset = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        splits=splits,
        label_types=[label_type],
        classes=classes,
        max_samples=4000,
    )

    # Convert all class labels to "vehicle"
    for sample in dataset:
        # print(sample)
        detections = sample.ground_truth.detections
        for detection in detections:
            if detection.label in classes:
                detection.label = "vehicle"
        sample.save()

    # Export with single class
    for split in splits:
        split_view = dataset.match_tags([split])
        split_name = "train" if split == "train" else "val"
        max_samples = 2000 if split == "validation" else 4000
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            split=split_name,
            label_field=None,
            classes=["vehicle"],  # Single class definition
            max_samples=max_samples,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Download and export datasets.")
    parser.add_argument(
        "--export-type", "-t",
        choices=["images", "images_and_labels"],
        required=True,
        help="Type of export: 'images' or 'images_and_labels'"
    )
    parser.add_argument(
        "--label_type", "-l",
        type=str,
        default="detections",
        help="Label type for exporting images and labels"
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
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    export_dir = os.path.join(os.getcwd(), "datatest")
    if args.export_type == "images":
        export_only_images(export_dir, args.classes)
    elif args.export_type == "images_and_labels":
        export_images_and_labels(export_dir, args.label_type, args.splits, args.classes)