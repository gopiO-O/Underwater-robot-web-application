"""
Crowd Detection using YOLOv8 Pretrained Model
This script detects and counts people in images or videos using a pretrained YOLO model.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse


class CrowdDetector:
    """Crowd detection using YOLOv8 pretrained model."""
    
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5, multi_scale=True, realtime_mode=False):
        """
        Initialize the crowd detector.
        
        Args:
            model_name: Name of the YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
            confidence_threshold: Minimum confidence for detections
            multi_scale: Use multi-scale detection for better accuracy
            realtime_mode: Optimize for speed over accuracy
        """
        print(f"Loading {model_name} model...")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0  # COCO dataset class ID for 'person'
        self.multi_scale = multi_scale
        self.realtime_mode = realtime_mode
        
        # Adjust scales based on mode
        if realtime_mode:
            self.scales = [640]  # Single scale for speed
        elif multi_scale:
            self.scales = [640, 960]  # Reduced scales for balance
        else:
            self.scales = [640]
    
    def _nms(self, detections, iou_threshold=0.3):
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(detections) == 0:
            return []
        
        boxes = np.array(detections)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _detect_multi_scale(self, img):
        """Detect people at multiple scales and combine results."""
        all_detections = []
        
        # Use optimized settings for real-time
        augment = False if self.realtime_mode else True
        max_det = 300 if self.realtime_mode else 1000
        
        for scale in self.scales:
            results = self.model(
                img,
                conf=self.confidence_threshold,
                iou=0.5,
                classes=[0],  # Person class only
                verbose=False,
                imgsz=scale,
                augment=augment,
                max_det=max_det
            )
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == self.person_class_id:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        all_detections.append([x1, y1, x2, y2, conf])
        
        # Apply NMS to remove duplicates (only if needed)
        if len(all_detections) > 0 and not self.realtime_mode:
            keep_indices = self._nms(all_detections, iou_threshold=0.4)
            return [all_detections[i] for i in keep_indices]
        
        return all_detections
        
    def detect_crowd(self, image_path, save_output=True, output_dir='output'):
        """
        Detect people in a single image.
        
        Args:
            image_path: Path to the input image
            save_output: Whether to save annotated image
            output_dir: Directory to save output
            
        Returns:
            count: Number of people detected
            annotated_image: Image with bounding boxes
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run multi-scale inference
        detections = self._detect_multi_scale(img)
        people_count = len(detections)
        
        # Draw bounding boxes
        annotated_img = img.copy()
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f'Person {conf:.2f}'
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add crowd count to image
        count_text = f'Crowd Count: {people_count}'
        cv2.putText(annotated_img, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Save output
        if save_output:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            output_file = output_path / f"detected_{Path(image_path).name}"
            cv2.imwrite(str(output_file), annotated_img)
            print(f"Saved annotated image to: {output_file}")
        
        return people_count, annotated_img
    
    def detect_crowd_video(self, video_path, save_output=True, output_dir='output'):
        """
        Detect people in a video.
        
        Args:
            video_path: Path to the input video
            save_output: Whether to save annotated video
            output_dir: Directory to save output
            
        Returns:
            average_count: Average crowd count across frames
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        if save_output:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            output_file = output_path / f"detected_{Path(video_path).name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        frame_count = 0
        total_people = 0
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run multi-scale inference
            detections = self._detect_multi_scale(frame)
            people_count = len(detections)
            
            # Draw bounding boxes
            for detection in detections:
                x1, y1, x2, y2, conf = detection
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            total_people += people_count
            
            # Add count to frame
            count_text = f'Count: {people_count}'
            cv2.putText(frame, count_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Write frame
            if save_output:
                out.write(frame)
            
            # Progress update
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        if save_output:
            out.release()
            print(f"Saved annotated video to: {output_file}")
        
        average_count = total_people / frame_count if frame_count > 0 else 0
        print(f"Average crowd count: {average_count:.2f}")
        
        return average_count
    
    def detect_realtime(self, camera_id=0):
        """
        Real-time crowd detection from webcam.
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
        """
        import time
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        # Enable real-time optimizations
        self.realtime_mode = True
        self.scales = [640]  # Force single scale
        
        print("Starting real-time detection. Press 'q' to quit.")
        print(f"Model: {self.model.model_name}")
        
        fps_list = []
        frame_skip = 0  # Process every frame for accuracy
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Direct YOLO inference with optimized settings
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=0.45,  # Balanced IoU for accuracy
                classes=[0],
                verbose=False,
                imgsz=640,
                augment=False,
                max_det=500,  # Increased for yolov8x
                device='0' if self.model.device.type == 'cuda' else 'cpu'
            )
            
            # Draw boxes directly
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == self.person_class_id:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Red box with thin thickness
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            
            cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Crowd Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Crowd Detection using YOLOv8')
    parser.add_argument('--input', type=str, help='Path to input image or video')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'webcam'],
                       default='image', help='Detection mode')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera ID for webcam mode')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = CrowdDetector(model_name=args.model, 
                            confidence_threshold=args.confidence)
    
    # Run detection based on mode
    if args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            return
        count, _ = detector.detect_crowd(args.input, output_dir=args.output_dir)
        print(f"Detected {count} people in the image")
        
    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            return
        avg_count = detector.detect_crowd_video(args.input, output_dir=args.output_dir)
        print(f"Average crowd count: {avg_count:.2f}")
        
    elif args.mode == 'webcam':
        detector.detect_realtime(camera_id=args.camera_id)

if __name__ == '__main__':
    main()
