import argparse
import sys
from pathlib import Path
import os
try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ultralytics is required. pip install ultralytics") from exc


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data.yaml"
DEFAULT_WEIGHTS = ROOT.parent / "yolov8n.pt"
DEFAULT_PROJECT = ROOT / "runs_garbage"

# Auto-select device: GPU id '0' if available else 'cpu'
DEFAULT_DEVICE = "0" if (torch and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"


def train_model(
    weights: Path = DEFAULT_WEIGHTS,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = DEFAULT_DEVICE,
    workers: int = 8,
    run_name: str = "yolo8n-floating-waste",
) -> None:
    """Train YOLO model on floating waste dataset with optimized parameters for real-time detection."""
    if not DATA_YAML.exists():
        raise SystemExit(f"data.yaml missing at {DATA_YAML}")

    if not weights.exists():
        raise SystemExit(f"Weights not found at {weights}")

    print(f"\n{'='*60}")
    print(f"  FLOATING WASTE DETECTION - TRAINING")
    print(f"{'='*60}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Base weights: {weights.name}")
    print(f"Epochs: {epochs} | Image size: {imgsz} | Batch: {batch}")
    print(f"Device: {device} | Workers: {workers}")
    print(f"{'='*60}\n")

    model = YOLO(str(weights))

    # Optimized training parameters for real-time detection
    model.train(
        data=str(DATA_YAML),
        project=str(DEFAULT_PROJECT),
        name=run_name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        
        # Optimization for real-time performance
        patience=20,              # Early stopping patience
        save=True,                # Save checkpoints
        save_period=10,           # Save every 10 epochs
        cache=True,               # Cache images for faster training
        
        # Augmentation (helps with varied water/lighting conditions)
        hsv_h=0.015,              # HSV-Hue augmentation
        hsv_s=0.7,                # HSV-Saturation
        hsv_v=0.4,                # HSV-Value
        degrees=10.0,             # Rotation (+/- deg)
        translate=0.1,            # Translation (+/- fraction)
        scale=0.5,                # Scale (+/- gain)
        shear=0.0,                # Shear (+/- deg)
        perspective=0.0,          # Perspective (+/- fraction)
        flipud=0.0,               # Vertical flip probability
        fliplr=0.5,               # Horizontal flip probability
        mosaic=1.0,               # Mosaic augmentation
        mixup=0.0,                # Mixup augmentation
        
        # Optimizer settings
        optimizer='auto',         # Auto-select optimizer (SGD/Adam)
        lr0=0.01,                 # Initial learning rate
        lrf=0.01,                 # Final learning rate factor
        momentum=0.937,           # Momentum
        weight_decay=0.0005,      # Weight decay
        warmup_epochs=3.0,        # Warmup epochs
        warmup_momentum=0.8,      # Warmup momentum
        warmup_bias_lr=0.1,       # Warmup bias learning rate
        
        # Loss weights
        box=7.5,                  # Box loss weight
        cls=0.5,                  # Class loss weight
        dfl=1.5,                  # DFL loss weight
        
        # Validation & visualization
        val=True,                 # Validate during training
        plots=True,               # Save training plots
        exist_ok=True,
        
        # Performance
        amp=True,                 # Automatic Mixed Precision for speed
        fraction=1.0,             # Use 100% of dataset
        
        # For real-time: prefer speed over marginal accuracy
        dropout=0.0,              # No dropout for faster inference
    )
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best model saved at: {DEFAULT_PROJECT}/{run_name}/weights/best.pt")
    print(f"Last model saved at: {DEFAULT_PROJECT}/{run_name}/weights/last.pt")
    print(f"\nTo run real-time detection:")
    print(f'  python train_floating_garbage.py realtime --model "{DEFAULT_PROJECT}/{run_name}/weights/best.pt" --source 0')
    print(f"{'='*60}\n")


def run_realtime(
    model_path: Path,
    source: str = "0",
    conf: float = 0.4,
    iou: float = 0.5,
    device: str = DEFAULT_DEVICE,
    imgsz: int = 640,
    save: bool = False,
) -> None:
    """Run real-time detection from webcam or video file with optimized parameters."""
    if not model_path.exists():
        raise SystemExit(f"Model not found at {model_path}")

    print(f"\n{'='*60}")
    print(f"  FLOATING WASTE DETECTION - REAL-TIME")
    print(f"{'='*60}")
    print(f"Model: {model_path.name}")
    print(f"Source: {source} | Confidence: {conf} | IoU: {iou}")
    print(f"Device: {device} | Image size: {imgsz}")
    print(f"{'='*60}\n")

    model = YOLO(str(model_path))
    
    # Real-time detection with optimized settings
    results = model.predict(
        source=source,
        conf=conf,                    # Confidence threshold (0.4 for balance)
        iou=iou,                      # IoU threshold for NMS
        device=device,
        imgsz=imgsz,
        stream=True,                  # Enable streaming for real-time
        show=True,                    # Display results
        save=save,                    # Save predictions to file
        
        # Real-time optimization
        vid_stride=1,                 # Process every frame
        visualize=False,              # No feature visualization (faster)
        augment=False,                # No test-time augmentation (faster)
        agnostic_nms=False,           # Class-specific NMS
        max_det=50,                   # Max 50 detections per image
        
        # Display settings
        line_width=2,                 # Bounding box thickness
        show_labels=True,             # Show class labels
        show_conf=True,               # Show confidence scores
        
        # Performance
        half=False,                   # FP16 inference (use True if GPU supports)
        verbose=True,                 # Print results
    )
    
    # Process streaming results
    for r in results:
        # Results are displayed in real-time window
        # Press 'q' to quit
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run floating waste detector")
    # 'required' not enforced to allow default 'train' when no subcommand provided
    sub = parser.add_subparsers(dest="cmd")

    train_p = sub.add_parser("train", help="Train YOLO on floating waste dataset with optimized parameters")
    train_p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Base weights to fine-tune (default: yolov8n.pt)")
    train_p.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    train_p.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    train_p.add_argument("--batch", type=int, default=16, help="Batch size (default: 16, reduce if GPU OOM)")
    train_p.add_argument("--device", type=str, default=DEFAULT_DEVICE, help=f"GPU id or 'cpu' (default: {DEFAULT_DEVICE})")
    train_p.add_argument("--workers", type=int, default=8, help="Dataloader workers (default: 8)")
    train_p.add_argument("--run-name", type=str, default="yolo8n-floating-waste", help="Run name for output folder")

    rt_p = sub.add_parser("realtime", help="Run real-time floating waste detection")
    rt_p.add_argument("--model", type=Path, required=True, help="Trained model path (e.g., runs_garbage/.../weights/best.pt)")
    rt_p.add_argument("--source", type=str, default="0", help="Camera index (0,1,2) or video file path")
    rt_p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    rt_p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS (default: 0.5)")
    rt_p.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    rt_p.add_argument("--device", type=str, default=DEFAULT_DEVICE, help=f"GPU id or 'cpu' (default: {DEFAULT_DEVICE})")
    rt_p.add_argument("--save", action="store_true", help="Save detection results to file")

    # If no arguments provided, default to 'train' command
    if len(sys.argv) == 1:
        # Build a namespace with defaults of 'train' parser
        args = parser.parse_args(["train"])  # use subparser defaults
        args.cmd = "train"
        return args
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "train":
        train_model(
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            run_name=args.run_name,
        )
    elif args.cmd == "realtime":
        run_realtime(
            model_path=args.model,
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save=args.save,
        )
    else:  # pragma: no cover
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
