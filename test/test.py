import fiftyone as fo
import fiftyone.zoo as foz

# 1. Download only 500 images containing 'person' from COCO 2017
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",               # Using validation split for fast download
    label_types=["detections"],       # Fetch bounding box annotations
    classes=["person"],              # Filter ONLY for person class
    max_samples=2000,                  # Limit to 500 images
    only_matching=True,               # Exclude other object classes present in the images
)

# 2. Export dataset directly to YOLO format
output_dir = "./coco_person_yolo"

dataset.export(
    export_dir=output_dir,
    dataset_type=fo.types.YOLOv5Dataset,  # Standard YOLO text label format
    label_field="ground_truth",
    classes=["person"],                   # Maps 'person' to class ID 0
)

print(f"\nSuccessfully downloaded and converted 500 person images to YOLO format in: {output_dir}")
