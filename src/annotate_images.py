#!/usr/bin/env python
from ultralytics import YOLO
import os

def is_vehicle(box):
    if int(box.cls)   == 2:  # COCO: class 2 = car
        return True
    elif int(box.cls) == 3:  # COCO: class 3 = motorcycle
        return True
    elif int(box.cls) == 5:  # COCO: class 5 = bus
        return True
    elif int(box.cls) == 7:  # COCO: class 7 = truck
        return True
    else:
        return False

def annotate_images(images_dir, output_dir, model):
    results = model(source=images_dir, stream=True, conf=0.5)

    for result in results:
        img_path = result.path
        base_name = os.path.basename(img_path)
        name, _ = os.path.splitext(base_name)
        txt_path = os.path.join(output_dir, name + '.txt')

        width, height = result.orig_shape[1], result.orig_shape[0]

        filtered_boxes = []

        for box in result.boxes:
            if is_vehicle(box):
                # YOLO format: cls x_center y_center width height (all relative)
                coords = box.xywh[0].tolist()
                x_center = coords[0] / width
                y_center = coords[1] / height
                w = coords[2] / width
                h = coords[3] / height
                filtered_boxes.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if filtered_boxes:
            with open(txt_path, 'w') as f:
                f.write('\n'.join(filtered_boxes))

# Load the model
model = YOLO("yolo11x.pt")
images_train_dir = "dataset/images/train"
output_train_dir = "dataset/labels/train"
os.makedirs(output_train_dir, exist_ok=True)
images_val_dir = "dataset/images/val"
output_val_dir = "dataset/labels/val"
os.makedirs(output_val_dir, exist_ok=True)

annotate_images(images_train_dir, output_train_dir, model)
annotate_images(images_val_dir, output_val_dir, model)
