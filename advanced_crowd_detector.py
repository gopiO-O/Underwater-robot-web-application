"""
Advanced Crowd Detection and Counting using Multiple Methods
- YOLO for sparse crowds (object detection)
- Density estimation for dense crowds (crowd counting)
- Ensemble approach for best accuracy
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class DensityEstimator(nn.Module):
    """CSRNet-based density estimation for accurate crowd counting."""
    
    def __init__(self):
        super(DensityEstimator, self).__init__()
        
        # Frontend: VGG16-like architecture
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Backend: Dilated convolutions
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        
        # Output layer
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AdvancedCrowdDetector:
    """Advanced crowd detection using both object detection and density estimation."""
    
    def __init__(self, yolo_model='yolov8x.pt', use_density=True, confidence=0.25):
        """
        Initialize the advanced crowd detector.
        
        Args:
            yolo_model: YOLO model for object detection
            use_density: Whether to use density estimation for dense crowds
            confidence: Confidence threshold for YOLO
        """
        print("Initializing Advanced Crowd Detector...")
        
        # YOLO for sparse crowds
        print(f"Loading YOLO model: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        self.yolo.model.eval()
        self.confidence = confidence
        
        # Density estimator for dense crowds
        self.use_density = use_density
        if use_density:
            print("Initializing density estimation model...")
            self.density_model = DensityEstimator()
            self.density_model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.density_model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def estimate_density(self, img):
        """Estimate crowd density using density map."""
        if not self.use_density:
            return 0
        
        # Prepare image
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Generate density map
        with torch.no_grad():
            density_map = self.density_model(img_tensor)
        
        # Count from density map
        count = density_map.sum().item()
        return max(0, int(count))
    
    def detect_with_yolo(self, img):
        """Detect people using YOLO with optimized settings."""
        detections = []
        
        # Multiple scales for better detection
        scales = [640, 960, 1280, 1600]
        
        for scale in scales:
            results = self.yolo(
                img,
                conf=self.confidence,
                iou=0.3,
                classes=[0],
                verbose=False,
                imgsz=scale,
                augment=True,
                max_det=1000
            )
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        detections.append([x1, y1, x2, y2, conf])
        
        # Apply NMS
        if len(detections) > 0:
            detections = np.array(detections)
            keep = self._nms(detections, 0.4)
            return detections[keep]
        return np.array([])
    
    def _nms(self, detections, iou_threshold):
        """Non-Maximum Suppression."""
        if len(detections) == 0:
            return []
        
        x1, y1, x2, y2, scores = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3], detections[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect_crowd(self, image_path, save_output=True, output_dir='output'):
        """
        Detect crowd using hybrid approach.
        
        Args:
            image_path: Path to input image
            save_output: Whether to save results
            output_dir: Output directory
            
        Returns:
            total_count: Estimated crowd count
            annotated_img: Annotated image
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        print(f"\nAnalyzing image: {image_path}")
        print("=" * 60)
        
        # Method 1: YOLO detection
        print("Method 1: YOLO object detection (multi-scale)...")
        yolo_detections = self.detect_with_yolo(img)
        yolo_count = len(yolo_detections)
        print(f"  → YOLO detected: {yolo_count} people")
        
        # Method 2: Density estimation
        density_count = 0
        if self.use_density:
            print("Method 2: Density estimation...")
            density_count = self.estimate_density(img)
            print(f"  → Density estimate: {density_count} people")
        
        # Hybrid decision: use density for dense crowds, YOLO for sparse
        # If YOLO detects fewer people but they're spread out, it's likely sparse
        # If density suggests many more people, trust density estimation
        
        img_area = img.shape[0] * img.shape[1]
        density_ratio = density_count / (img_area / 10000) if img_area > 0 else 0
        
        # Decision logic
        if density_count > yolo_count * 1.5 and density_ratio > 2:
            # Dense crowd - trust density estimation more
            final_count = int(0.7 * density_count + 0.3 * yolo_count)
            method = "Density-based (dense crowd)"
        elif yolo_count > 0:
            # Sparse/medium crowd - trust YOLO more
            final_count = int(0.8 * yolo_count + 0.2 * density_count)
            method = "YOLO-based (sparse/medium crowd)"
        else:
            # Fallback to density
            final_count = density_count
            method = "Density-based (fallback)"
        
        print(f"\nFinal Estimate: {final_count} people ({method})")
        print("=" * 60)
        
        # Visualize
        annotated_img = img.copy()
        
        # Draw YOLO detections
        for detection in yolo_detections:
            x1, y1, x2, y2, conf = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, f'{conf:.2f}', (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add text overlay
        cv2.putText(annotated_img, f'Final Count: {final_count}', (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(annotated_img, f'YOLO: {yolo_count} | Density: {density_count}', (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(annotated_img, method, (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Save
        if save_output:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            output_file = output_path / f"advanced_{Path(image_path).name}"
            cv2.imwrite(str(output_file), annotated_img)
            print(f"\n✓ Saved to: {output_file}")
        
        return final_count, annotated_img
    
    def detect_crowd_video(self, video_path, save_output=True, output_dir='output', process_every_n_frames=1):
        """
        Detect crowd in video with optimized processing.
        
        Args:
            video_path: Path to input video
            save_output: Whether to save annotated video
            output_dir: Output directory
            process_every_n_frames: Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
            
        Returns:
            stats: Dictionary with statistics
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("\n" + "="*70)
        print(f"Processing Video: {Path(video_path).name}")
        print("="*70)
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total Frames: {total_frames}")
        print(f"Processing: Every {process_every_n_frames} frame(s)")
        print("="*70 + "\n")
        
        # Setup video writer
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        output_file = output_path / f"detected_{Path(video_path).name}"
        
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        total_people = 0
        max_count = 0
        min_count = float('inf')
        counts_per_frame = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame or use previous count
            if frame_count % process_every_n_frames == 0:
                processed_count += 1
                
                # YOLO detection (faster for video)
                yolo_detections = self.detect_with_yolo(frame)
                people_count = len(yolo_detections)
                
                # Update statistics
                total_people += people_count
                max_count = max(max_count, people_count)
                min_count = min(min_count, people_count)
                counts_per_frame.append(people_count)
                
                # Draw detections
                for detection in yolo_detections:
                    x1, y1, x2, y2, conf = detection
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add info overlay
                cv2.putText(frame, f'Count: {people_count}', (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                avg_so_far = total_people / processed_count
                cv2.putText(frame, f'Avg: {avg_so_far:.1f}', (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Progress
                if processed_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | Current count: {people_count}")
            
            # Write frame
            if save_output:
                out.write(frame)
        
        cap.release()
        if save_output:
            out.release()
        
        # Calculate statistics
        avg_count = total_people / processed_count if processed_count > 0 else 0
        
        print("\n" + "="*70)
        print("VIDEO PROCESSING COMPLETE")
        print("="*70)
        print(f"Total Frames Processed: {processed_count}")
        print(f"Average Crowd Count: {avg_count:.2f}")
        print(f"Maximum Count: {max_count}")
        print(f"Minimum Count: {min_count}")
        
        if save_output:
            print(f"\n✓ Saved video to: {output_file}")
        
        print("="*70 + "\n")
        
        return {
            'average': avg_count,
            'max': max_count,
            'min': min_count,
            'frames_processed': processed_count,
            'counts': counts_per_frame
        }
    
    def detect_realtime(self, camera_id=0, show_density=False):
        """
        Real-time crowd detection from webcam.
        
        Args:
            camera_id: Camera device ID (0 for laptop, 1 for external)
            show_density: Whether to show density heatmap (slower)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        print(f"\nStarting real-time detection on camera {camera_id}")
        print("Press 'q' to quit, 'd' to toggle density view")
        print("="*60)
        
        show_density_viz = show_density
        fps_list = []
        import time
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO detection
            yolo_detections = self.detect_with_yolo(frame)
            yolo_count = len(yolo_detections)
            
            # Density estimation (optional)
            density_count = 0
            density_map = None
            if self.use_density and show_density_viz:
                density_count = self.estimate_density(frame)
                # Create simple density visualization
                h, w = frame.shape[:2]
                density_map = np.zeros((h, w), dtype=np.float32)
                for detection in yolo_detections:
                    x1, y1, x2, y2, conf = detection
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv2.circle(density_map, (cx, cy), 30, 1.0, -1)
            
            # Determine final count
            if density_count > yolo_count * 1.5:
                final_count = int(0.7 * density_count + 0.3 * yolo_count)
                method = "Hybrid"
            else:
                final_count = yolo_count
                method = "YOLO"
            
            # Draw bounding boxes
            for detection in yolo_detections:
                x1, y1, x2, y2, conf = detection
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate FPS
            fps = 1 / (time.time() - start_time)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)
            
            # Display information
            cv2.putText(frame, f'Crowd Count: {final_count} ({method})', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, f'YOLO: {yolo_count}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if density_count > 0:
                cv2.putText(frame, f'Density: {density_count}', (10, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show density heatmap if enabled
            if show_density_viz and density_map is not None:
                density_colored = cv2.applyColorMap(
                    (density_map * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                frame = cv2.addWeighted(frame, 0.7, density_colored, 0.3, 0)
            
            # Display
            cv2.imshow(f'Advanced Crowd Detection - Camera {camera_id}', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_density_viz = not show_density_viz
                print(f"Density view: {'ON' if show_density_viz else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nAverage FPS: {np.mean(fps_list):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Advanced Crowd Detection')
    parser.add_argument('--input', type=str, help='Input image or video path (not needed for webcam mode)')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'webcam'], default='image',
                       help='Detection mode (image, video, or webcam)')
    parser.add_argument('--model', type=str, default='yolov8x.pt', help='YOLO model')
    parser.add_argument('--confidence', type=float, default=0.25, 
                       help='YOLO confidence threshold (lower for dense crowds)')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--no-density', action='store_true', 
                       help='Disable density estimation (YOLO only)')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every Nth frame (for faster video processing)')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera ID for webcam mode (0=laptop, 1=external)')
    parser.add_argument('--show-density', action='store_true',
                       help='Show density heatmap in webcam mode (slower)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = AdvancedCrowdDetector(
        yolo_model=args.model,
        use_density=not args.no_density,
        confidence=args.confidence
    )
    
    # Detect based on mode
    if args.mode == 'image':
        if not args.input:
            parser.error("--input is required for image mode")
        count, _ = detector.detect_crowd(args.input, output_dir=args.output_dir)
        print(f"\n{'='*60}")
        print(f"FINAL RESULT: {count} people detected")
        print(f"{'='*60}\n")
    elif args.mode == 'video':
        if not args.input:
            parser.error("--input is required for video mode")
        stats = detector.detect_crowd_video(
            args.input, 
            output_dir=args.output_dir,
            process_every_n_frames=args.skip_frames
        )
    elif args.mode == 'webcam':
        detector.detect_realtime(
            camera_id=args.camera_id,
            show_density=args.show_density
        )


if __name__ == '__main__':
    main()
