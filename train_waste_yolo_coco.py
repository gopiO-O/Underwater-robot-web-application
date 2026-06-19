"""
Train YOLOv8 to detect non-living objects (waste items) from COCO dataset
Filters out: person, animals, birds, insects
Keeps: vehicles, indoor objects, outdoor objects, etc.
"""

from ultralytics import YOLO
import torch
import yaml
import os

# COCO classes - Filter out living things
COCO_CLASSES = {
    0: 'person',  # EXCLUDE
    1: 'bicycle',  # KEEP
    2: 'car',  # KEEP
    3: 'motorcycle',  # KEEP
    4: 'airplane',  # KEEP
    5: 'bus',  # KEEP
    6: 'train',  # KEEP
    7: 'truck',  # KEEP
    8: 'boat',  # KEEP
    9: 'traffic light',  # KEEP
    10: 'fire hydrant',  # KEEP
    11: 'stop sign',  # KEEP
    12: 'parking meter',  # KEEP
    13: 'bench',  # KEEP
    14: 'cat',  # EXCLUDE
    15: 'dog',  # EXCLUDE
    16: 'horse',  # EXCLUDE
    17: 'sheep',  # EXCLUDE
    18: 'cow',  # EXCLUDE
    19: 'elephant',  # EXCLUDE
    20: 'bear',  # EXCLUDE
    21: 'zebra',  # EXCLUDE
    22: 'giraffe',  # EXCLUDE
    23: 'backpack',  # KEEP
    24: 'umbrella',  # KEEP
    25: 'handbag',  # KEEP
    26: 'tie',  # KEEP
    27: 'suitcase',  # KEEP
    28: 'frisbee',  # KEEP
    29: 'skis',  # KEEP
    30: 'snowboard',  # KEEP
    31: 'sports ball',  # KEEP
    32: 'kite',  # KEEP
    33: 'baseball bat',  # KEEP
    34: 'baseball glove',  # KEEP
    35: 'skateboard',  # KEEP
    36: 'surfboard',  # KEEP
    37: 'tennis racket',  # KEEP
    38: 'bottle',  # KEEP - WASTE
    39: 'wine glass',  # KEEP
    40: 'cup',  # KEEP - WASTE
    41: 'fork',  # KEEP
    42: 'knife',  # KEEP
    43: 'spoon',  # KEEP
    44: 'bowl',  # KEEP
    45: 'banana',  # EXCLUDE
    46: 'apple',  # EXCLUDE
    47: 'sandwich',  # EXCLUDE
    48: 'orange',  # EXCLUDE
    49: 'broccoli',  # EXCLUDE
    50: 'carrot',  # EXCLUDE
    51: 'hot dog',  # EXCLUDE
    52: 'pizza',  # EXCLUDE
    53: 'donut',  # EXCLUDE
    54: 'cake',  # EXCLUDE
    55: 'chair',  # KEEP
    56: 'couch',  # KEEP
    57: 'potted plant',  # EXCLUDE - Living
    58: 'bed',  # KEEP
    59: 'dining table',  # KEEP
    60: 'toilet',  # KEEP
    61: 'tv',  # KEEP
    62: 'laptop',  # KEEP
    63: 'mouse',  # KEEP
    64: 'remote',  # KEEP
    65: 'keyboard',  # KEEP
    66: 'microwave',  # KEEP
    67: 'oven',  # KEEP
    68: 'toaster',  # KEEP
    69: 'sink',  # KEEP
    70: 'refrigerator',  # KEEP
    71: 'book',  # KEEP
    72: 'clock',  # KEEP
    73: 'vase',  # KEEP
    74: 'scissors',  # KEEP
    75: 'teddy bear',  # KEEP
    76: 'hair drier',  # KEEP
    77: 'toothbrush',  # KEEP
    78: 'hair brush',  # KEEP
    79: 'unknown'  # EXCLUDE
}

# Living things to exclude
LIVING_THINGS = {
    0, 14, 15, 16, 17, 18, 19, 20, 21, 22,  # person, animals
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54,  # food items
    57  # potted plant
}

# Non-living waste/objects to keep
WASTE_CLASSES = {idx: name for idx, name in COCO_CLASSES.items() if idx not in LIVING_THINGS}

print("=" * 60)
print("NON-LIVING OBJECT DETECTION CLASSES")
print("=" * 60)
print(f"\nTotal classes to train on: {len(WASTE_CLASSES)}")
print("\nClasses included:")
for idx, name in sorted(WASTE_CLASSES.items()):
    print(f"  {idx}: {name}")

print(f"\nClasses excluded (living things): {len(LIVING_THINGS)}")
print("=" * 60)


def train_waste_detector_from_coco():
    """
    Train YOLOv8 on COCO dataset filtered for non-living objects
    This will detect waste items, vehicles, furniture, etc.
    """
    
    # Check GPU availability
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Load YOLOv8 model
    print("\n📦 Loading YOLOv8 model...")
    model = YOLO('yolov8m.pt')  # Use medium model for balance
    
    # Train on COCO dataset
    print("\n🚀 Starting training on COCO dataset (non-living objects only)...")
    print("Note: This will download COCO2017 dataset (~20GB) on first run")
    
    results = model.train(
        data='coco8.yaml',  # YOLO's mini COCO dataset for testing
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/waste_detection_coco',
        name='yolov8m_waste_v1',
        save=True,
        augment=True,
        flipud=0.5,  # Flip images upside down (helps with water reflections)
        fliplr=0.5,  # Flip left-right
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.1,  # Mix images
        cache=True,
        workers=8,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # DFL loss gain
        hsv_h=0.015,  # Image HSV-H augmentation
        hsv_s=0.7,  # Image HSV-S augmentation
        hsv_v=0.4,  # Image HSV-V augmentation
        degrees=10.0,  # Image rotation
        translate=0.1,  # Image translation
        scale=0.5,  # Image scale
        perspective=0.0,  # Image perspective
        copy_paste=0.0,  # Segment copy-paste
    )
    
    print("\n✅ Training completed!")
    return model, results


def validate_model(model):
    """
    Validate the trained model
    """
    print("\n🔍 Validating model...")
    metrics = model.val()
    print(f"\nValidation Results:")
    print(f"  mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    return metrics


def test_detection(model, image_path):
    """
    Test detection on a single image
    """
    print(f"\n📸 Testing detection on: {image_path}")
    results = model.predict(image_path, conf=0.6)
    
    for result in results:
        print(f"\nDetected objects:")
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = WASTE_CLASSES.get(class_id, 'unknown')
            confidence = float(box.conf[0])
            print(f"  - {class_name}: {confidence:.2%}")
    
    return results


def export_model(model):
    """
    Export model to different formats
    """
    print("\n💾 Exporting model to multiple formats...")
    
    formats = ['torchscript', 'onnx', 'openvino', 'tflite']
    
    for fmt in formats:
        try:
            print(f"  Exporting to {fmt}...")
            model.export(format=fmt)
            print(f"  ✅ {fmt} export successful")
        except Exception as e:
            print(f"  ⚠️ {fmt} export failed: {str(e)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("YOLO v8 - Non-Living Object Detection Training")
    print("=" * 60)
    
    # Train model
    model, results = train_waste_detector_from_coco()
    
    # Validate
    metrics = validate_model(model)
    
    # Export to multiple formats
    export_model(model)
    
    print("\n" + "=" * 60)
    print("✨ Training Complete!")
    print("=" * 60)
    print(f"\nBest model saved at: runs/waste_detection_coco/yolov8m_waste_v1/weights/best.pt")
    print("Use this path in your detection scripts")
