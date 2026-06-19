"""
Drowning Detection - Real-time Video Inference
Detects drowning incidents in video streams using trained YOLOv8 model
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime

# Install requirements if needed
def check_requirements():
    required = ['ultralytics', 'opencv-python', 'numpy']
    import subprocess
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

check_requirements()

import cv2
import numpy as np
from ultralytics import YOLO

# Paths
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR / "models"
DEFAULT_MODEL = MODEL_DIR / "drowning_detector_best.pt"

# Detection classes
CLASS_NAMES = {
    0: 'person_in_water',
    1: 'person_struggling',
    2: 'person_floating',
    3: 'person_submerged',
    4: 'out_of_water',
    5: 'drowning',
    6: 'swimming'
}

# Alert classes (trigger emergency)
ALERT_CLASSES = ['drowning', 'person_struggling', 'person_submerged']

# Colors for visualization (BGR format)
CLASS_COLORS = {
    'drowning': (0, 0, 255),          # Red - DANGER
    'person_struggling': (0, 69, 255), # Orange-Red
    'person_submerged': (0, 165, 255), # Orange
    'person_floating': (0, 255, 255),  # Yellow
    'person_in_water': (255, 255, 0),  # Cyan
    'swimming': (0, 255, 0),           # Green - SAFE
    'out_of_water': (0, 255, 0)        # Green - SAFE
}


class DrowningDetector:
    def __init__(self, model_path=None, conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize Drowning Detector
        
        Args:
            model_path: Path to trained YOLOv8 model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.alert_active = False
        self.alert_start_time = None
        self.detection_history = []
        
        # Load model
        if model_path is None:
            model_path = DEFAULT_MODEL
        
        if not Path(model_path).exists():
            print(f"⚠️ Model not found at {model_path}")
            print("📥 Using pretrained YOLOv8 model for demo...")
            print("   Train your model first with: python train_model.py --mode quick")
            self.model = YOLO('yolov8n.pt')
            self.custom_model = False
        else:
            print(f"✅ Loading model: {model_path}")
            self.model = YOLO(str(model_path))
            self.custom_model = True
    
    def detect(self, frame):
        """
        Run detection on a single frame
        
        Returns:
            detections: List of detection dicts
            annotated_frame: Frame with annotations
        """
        # Run inference
        results = self.model(
            frame, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        detections = []
        annotated_frame = frame.copy()
        
        # Process detections
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                
                # Get class name
                if self.custom_model and cls in CLASS_NAMES:
                    class_name = CLASS_NAMES[cls]
                else:
                    class_name = results.names[cls] if cls in results.names else f"class_{cls}"
                
                # Create detection dict
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(conf),
                    'class': class_name,
                    'class_id': cls,
                    'is_alert': class_name in ALERT_CLASSES
                }
                detections.append(detection)
                
                # Draw on frame
                color = CLASS_COLORS.get(class_name, (255, 255, 255))
                thickness = 3 if detection['is_alert'] else 2
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label background
                label = f"{class_name}: {conf:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w + 5, y1),
                    color, -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame, label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2
                )
                
                # Add alert indicator for dangerous detections
                if detection['is_alert']:
                    cv2.putText(
                        annotated_frame, "⚠️ ALERT",
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2
                    )
        
        # Check for alerts
        alert_detections = [d for d in detections if d['is_alert']]
        if alert_detections:
            self.trigger_alert(alert_detections)
        else:
            self.clear_alert()
        
        # Add status overlay
        annotated_frame = self.add_status_overlay(annotated_frame, detections)
        
        return detections, annotated_frame
    
    def trigger_alert(self, alert_detections):
        """Trigger drowning alert"""
        if not self.alert_active:
            self.alert_active = True
            self.alert_start_time = time.time()
            print(f"\n🚨 DROWNING ALERT! Detected: {[d['class'] for d in alert_detections]}")
    
    def clear_alert(self):
        """Clear alert if no dangerous detections"""
        if self.alert_active:
            self.alert_active = False
            self.alert_start_time = None
    
    def add_status_overlay(self, frame, detections):
        """Add status information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Status text
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"🕐 {timestamp}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Detection count
        total = len(detections)
        alerts = sum(1 for d in detections if d['is_alert'])
        
        status_text = f"Detections: {total} | Alerts: {alerts}"
        cv2.putText(frame, status_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Alert status
        if self.alert_active:
            alert_duration = time.time() - self.alert_start_time
            # Flashing effect
            if int(alert_duration * 3) % 2 == 0:
                cv2.putText(frame, "🚨 DROWNING ALERT!", (w - 250, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Alert: {alert_duration:.1f}s", (w - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "✅ ALL CLEAR", (w - 180, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add legend at bottom
        legend_y = h - 20
        legend_items = [
            ("DROWNING", (0, 0, 255)),
            ("Swimming", (0, 255, 0)),
            ("In Water", (255, 255, 0))
        ]
        x_offset = 10
        for label, color in legend_items:
            cv2.circle(frame, (x_offset, legend_y), 8, color, -1)
            cv2.putText(frame, label, (x_offset + 15, legend_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            x_offset += 120
        
        return frame


def detect_video(video_source, model_path=None, output_path=None, show=True):
    """
    Run drowning detection on video
    
    Args:
        video_source: Video file path, webcam index (0), or RTSP URL
        model_path: Path to trained model
        output_path: Path to save output video (optional)
        show: Show video window
    """
    print("\n" + "="*60)
    print("🏊 DROWNING DETECTION SYSTEM")
    print("="*60)
    
    # Initialize detector
    detector = DrowningDetector(model_path)
    
    # Open video source
    if isinstance(video_source, int) or video_source.isdigit():
        video_source = int(video_source)
        print(f"📹 Opening webcam: {video_source}")
    else:
        print(f"📹 Opening video: {video_source}")
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {video_source}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video: {width}x{height} @ {fps}fps")
    if total_frames > 0:
        print(f"📊 Total frames: {total_frames}")
    
    # Setup video writer if output path specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Output will be saved to: {output_path}")
    
    print("\n🚀 Starting detection...")
    print("Press 'q' to quit, 's' to save screenshot\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            detections, annotated_frame = detector.detect(frame)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add FPS to frame
            cv2.putText(
                annotated_frame, f"FPS: {current_fps:.1f}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            
            # Progress for video files
            if total_frames > 0:
                progress = (frame_count / total_frames) * 100
                cv2.putText(
                    annotated_frame, f"Progress: {progress:.1f}%",
                    (width - 150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
            
            # Write to output video
            if writer:
                writer.write(annotated_frame)
            
            # Show frame
            if show:
                cv2.imshow('Drowning Detection', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️ Stopped by user")
                    break
                elif key == ord('s'):
                    screenshot_path = SCRIPT_DIR / f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(str(screenshot_path), annotated_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"\n💾 Output saved to: {output_path}")
        cv2.destroyAllWindows()
    
    # Summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print("\n" + "="*60)
    print("📊 DETECTION SUMMARY")
    print("="*60)
    print(f"   Frames processed: {frame_count}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average FPS: {avg_fps:.1f}")
    print("="*60)


def detect_image(image_path, model_path=None, output_path=None, show=True):
    """
    Run drowning detection on a single image
    """
    print(f"\n🖼️ Processing image: {image_path}")
    
    detector = DrowningDetector(model_path)
    
    # Read image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Failed to read image: {image_path}")
        return
    
    # Run detection
    detections, annotated_frame = detector.detect(frame)
    
    # Print detections
    print(f"\n📊 Found {len(detections)} detections:")
    for i, det in enumerate(detections):
        status = "⚠️ ALERT" if det['is_alert'] else "✅"
        print(f"   {i+1}. {status} {det['class']}: {det['confidence']:.2f}")
    
    # Save output
    if output_path:
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"\n💾 Saved to: {output_path}")
    
    # Show
    if show:
        cv2.imshow('Drowning Detection', annotated_frame)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detections


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Drowning Detection Inference')
    parser.add_argument('source', nargs='?', default='0',
                        help='Video file path, webcam index (0), or image path')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not show video window')
    
    args = parser.parse_args()
    
    # Determine if source is image or video
    source = args.source
    
    if Path(source).exists() and Path(source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        detect_image(
            source,
            model_path=args.model,
            output_path=args.output,
            show=not args.no_show
        )
    else:
        detect_video(
            source,
            model_path=args.model,
            output_path=args.output,
            show=not args.no_show
        )
