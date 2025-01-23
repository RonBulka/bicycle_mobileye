#!/usr/bin/env python
import fiftyone as fo
import os

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

def export_images_and_labels(export_dir, label_field, splits, classes):
    dataset = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split=None,
        label_types=["detections"],
        classes=classes,
        max_samples=3000,)

    # Export the splits
    for split in splits:
        split_view = dataset.match_tags(split)
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            split=split,
            classes=classes,
        )

if __name__ == '__main__':
    # Define the output directories
    cwd = os.getcwd()
    label_field = "vehicles"
    splits = ["train", "val"]
    classes = ["Car", "Truck", "Bus", "Motorcycle"]

    export_only_images(cwd, classes)
    # export_images_and_labels(cwd, label_field, splits, classes)