"""
Real-time waste detection using trained YOLOv8 model
Detects non-living objects: bottles, cups, debris, vehicles, furniture, etc.
"""

import cv2
import torch
from ultralytics import YOLO
from datetime import datetime
import numpy as np
import json

# Non-living object classes (filtered from COCO)
WASTE_CLASSES = {
    1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 23: 'backpack',
    24: 'umbrella', 25: 'handbag', 26: 'tie', 27: 'suitcase', 28: 'frisbee',
    29: 'skis', 30: 'snowboard', 31: 'sports ball', 32: 'kite', 33: 'baseball bat',
    34: 'baseball glove', 35: 'skateboard', 36: 'surfboard', 37: 'tennis racket',
    38: 'bottle', 39: 'wine glass', 40: 'cup', 41: 'fork', 42: 'knife',
    43: 'spoon', 44: 'bowl', 55: 'chair', 56: 'couch', 58: 'bed',
    59: 'dining table', 60: 'toilet', 61: 'tv', 62: 'laptop', 63: 'mouse',
    64: 'remote', 65: 'keyboard', 66: 'microwave', 67: 'oven', 68: 'toaster',
    69: 'sink', 70: 'refrigerator', 71: 'book', 72: 'clock', 73: 'vase',
    74: 'scissors', 75: 'teddy bear', 76: 'hair drier', 77: 'toothbrush', 78: 'hair brush'
}

# Waste severity mapping
WASTE_SEVERITY = {
    # High priority floating waste
    'bottle': 'HIGH',
    'cup': 'HIGH',
    'plastic bag': 'HIGH',
    'backpack': 'HIGH',
    'handbag': 'HIGH',
    'suitcase': 'HIGH',
    'frisbee': 'MEDIUM',
    'sports ball': 'MEDIUM',
    'skateboard': 'MEDIUM',
    'surfboard': 'MEDIUM',
    
    # Structural/other objects
    'boat': 'LOW',
    'bicycle': 'MEDIUM',
    'car': 'LOW',
    'motorcycle': 'LOW',
    'truck': 'LOW',
    'chair': 'MEDIUM',
    'bench': 'LOW',
    'bed': 'MEDIUM',
    'toilet': 'HIGH',
    'refrigerator': 'MEDIUM',
}


class WasteDetector:
    def __init__(self, model_path, confidence_threshold=0.6):
        """
        Initialize waste detector
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence_threshold: Minimum confidence for detections
        """
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        print(f"Model loaded. Confidence threshold: {confidence_threshold}")
    
    def detect(self, frame):
        """
        Run detection on a frame
        
        Args:
            frame: OpenCV frame/image
            
        Returns:
            detections: List of detected objects
            results: YOLO results object
        """
        results = self.model(frame, conf=self.confidence_threshold, device=self.device)
        detections = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                
                # Only include non-living objects
                if class_id not in WASTE_CLASSES:
                    continue
                
                detection = {
                    'class_id': class_id,
                    'class_name': WASTE_CLASSES.get(class_id, 'unknown'),
                    'confidence': float(box.conf[0]),
                    'bbox': [float(x) for x in box.xyxy[0].tolist()],
                    'area_pixels': float((box.xyxy[0][2] - box.xyxy[0][0]) * 
                                        (box.xyxy[0][3] - box.xyxy[0][1])),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add severity if available
                severity = WASTE_SEVERITY.get(detection['class_name'], 'MEDIUM')
                detection['severity'] = severity
                
                detections.append(detection)
        
        return detections, results
    
    def visualize(self, frame, results):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Original frame
            results: YOLO results
            
        Returns:
            annotated_frame: Frame with annotations
        """
        annotated_frame = results[0].plot()
        return annotated_frame
    
    def filter_detections(self, detections, min_confidence=0.7):
        """
        Filter detections by confidence
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold
            
        Returns:
            filtered_detections: Filtered list
        """
        return [d for d in detections if d['confidence'] >= min_confidence]
    
    def get_waste_summary(self, detections):
        """
        Summarize detected waste
        
        Args:
            detections: List of detections
            
        Returns:
            summary: Dictionary with waste summary
        """
        if not detections:
            return {
                'total_objects': 0,
                'critical_alerts': 0,
                'waste_types': {},
                'average_confidence': 0.0
            }
        
        summary = {
            'total_objects': len(detections),
            'critical_alerts': sum(1 for d in detections if d['severity'] == 'HIGH'),
            'waste_types': {},
            'average_confidence': np.mean([d['confidence'] for d in detections]),
            'detections': detections
        }
        
        # Count by type
        for detection in detections:
            class_name = detection['class_name']
            if class_name not in summary['waste_types']:
                summary['waste_types'][class_name] = 0
            summary['waste_types'][class_name] += 1
        
        return summary


def process_camera_stream(model_path, camera_source=0, output_video=None):
    """
    Process live camera stream with waste detection
    
    Args:
        model_path: Path to trained model
        camera_source: Camera index or RTSP URL
        output_video: Optional path to save output video
    """
    detector = WasteDetector(model_path)
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera source {camera_source}")
        return
    
    # Video writer setup
    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        print(f"Writing output to {output_video}")
    
    frame_count = 0
    
    print("\nStarting waste detection... Press 'q' to quit, 's' to save screenshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize for faster processing
        frame_resized = cv2.resize(frame, (640, 480))
        
        # Run detection
        detections, results = detector.detect(frame_resized)
        
        # Filter low confidence
        filtered_detections = detector.filter_detections(detections, min_confidence=0.6)
        
        # Get summary
        summary = detector.get_waste_summary(filtered_detections)
        
        # Visualize
        annotated_frame = detector.visualize(frame_resized, results)
        
        # Add statistics overlay
        cv2.putText(annotated_frame, f"Objects: {summary['total_objects']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Alerts: {summary['critical_alerts']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Confidence: {summary['average_confidence']:.2%}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {frame_count}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Print detections periodically
        if frame_count % 30 == 0:  # Every 30 frames (~1 sec at 30fps)
            print(f"\n[Frame {frame_count}] {summary['total_objects']} objects detected")
            for class_name, count in summary['waste_types'].items():
                print(f"  - {class_name}: {count}")
        
        # Write to output video
        if writer:
            writer.write(cv2.resize(annotated_frame, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))))
        
        # Display
        cv2.imshow('Waste Detection', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"waste_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved: {filename}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Total frames: {frame_count}")


def process_image(model_path, image_path):
    """
    Process single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to image file
    """
    detector = WasteDetector(model_path)
    
    print(f"Processing image: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    detections, results = detector.detect(frame)
    filtered_detections = detector.filter_detections(detections, min_confidence=0.6)
    summary = detector.get_waste_summary(filtered_detections)
    
    print(f"\nDetection Results:")
    print(f"  Total objects: {summary['total_objects']}")
    print(f"  Critical alerts: {summary['critical_alerts']}")
    print(f"  Average confidence: {summary['average_confidence']:.2%}")
    print(f"\nDetected waste types:")
    for class_name, count in summary['waste_types'].items():
        print(f"  - {class_name}: {count}")
    
    # Save annotated image
    annotated_frame = detector.visualize(frame, results)
    output_path = f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"\nAnnotated image saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Waste Detection using YOLOv8')
    parser.add_argument('--model', type=str, default='runs/waste_detection_coco/yolov8m_waste_v1/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--source', type=str, default='0',
                       help='Camera index (0, 1, etc.) or video file path or RTSP URL')
    parser.add_argument('--image', type=str, help='Process single image')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--conf', type=float, default=0.6,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    if args.image:
        process_image(args.model, args.image)
    else:
        # Try to convert source to int (camera index)
        try:
            source = int(args.source)
        except:
            source = args.source
        
        process_camera_stream(args.model, camera_source=source, output_video=args.output)
