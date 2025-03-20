import cv2
import numpy as np
import time
from ultralytics import YOLO
import argparse
import os
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description='Universal YOLOv8/YOLO11 Detection Demo')
    parser.add_argument('--source', type=str, default='0', 
                        help='Source for detection: path to video file or webcam index (default: 0)')
    parser.add_argument('--mode', type=str, default='yolov8n.pt', 
                        help='Path to any YOLOv8/YOLO11 model file (e.g., yolov8n.pt, yolo11n-obb.pt, yolov8n-seg.pt)')
    parser.add_argument('--conf', type=float, default=0.5, 
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Path to output video file (default: None)')
    parser.add_argument('--color', type=str, default='random',
                        help='Box color: "random" or R,G,B values (default: random)')
    parser.add_argument('--info', action='store_true',
                        help='Show detailed model information at startup')
    parser.add_argument('--verbose', action='store_true',
                        help='Show verbose detection details in terminal')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency value for segmentation masks (default: 0.5)')
    
    return parser.parse_args()

def generate_colors(num_classes):
    """Generate distinct colors for different classes"""
    colors = []
    for i in range(num_classes):
        # Generate evenly distributed hues, then convert to BGR
        hue = i / num_classes * 180
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors

def draw_detections(frame, result, colors, default_color, use_random_colors, model, alpha=0.5):
    """Draw detections based on model type (regular, segmentation, or OBB)"""
    class_counts = {}
    annotated_frame = frame.copy()
    
    # Check model types
    is_obb_model = hasattr(result, 'obb') and result.obb is not None
    has_masks = hasattr(result, 'masks') and result.masks is not None
    has_boxes = hasattr(result, 'boxes') and result.boxes is not None
    
    # Process segmentation masks first (if available)
    if has_masks:
        # Create a mask overlay image
        mask_overlay = annotated_frame.copy()
        
        # Draw each instance mask
        for i, mask in enumerate(result.masks):
            # Get class information
            class_id = 0
            class_name = "Object"
            
            if has_boxes and i < len(result.boxes):
                box = result.boxes[i]
                if hasattr(box, 'cls') and box.cls is not None and len(box.cls) > 0:
                    class_id = int(box.cls[0])
                    if hasattr(model, 'names') and class_id in model.names:
                        class_name = model.names[class_id]
                    else:
                        class_name = f"Class {class_id}"
            
            # Update class counts
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
            
            # Get color for this class
            if use_random_colors and class_id < len(colors):
                color = colors[class_id]
            else:
                color = default_color
            
            # Get mask as a numpy array
            mask_array = mask.data.cpu().numpy().astype(np.uint8)
            
            # Reshape if necessary
            if len(mask_array.shape) == 3 and mask_array.shape[0] == 1:
                mask_array = mask_array[0]
            
            # Create mask based on mask_array
            if mask_array.shape[:2] != frame.shape[:2]:
                # Resize mask if dimensions don't match
                mask_array = cv2.resize(mask_array, (frame.shape[1], frame.shape[0]))
            
            # Apply color to the mask area
            mask_overlay[mask_array > 0] = color
        
        # Blend the mask overlay with the original frame
        cv2.addWeighted(mask_overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
    
    # Process OBB results
    if is_obb_model:
        # For OBB models (oriented bounding boxes)
        if hasattr(result, 'obb') and result.obb is not None:
            for i, det in enumerate(result.obb):
                # Get class information
                if hasattr(det, 'cls') and det.cls is not None:
                    class_id = int(det.cls.item())
                    if hasattr(model, 'names') and class_id in model.names:
                        class_name = model.names[class_id]
                    else:
                        class_name = f"Class {class_id}"
                else:
                    class_id = 0
                    class_name = "Object"
                
                # Update class counts
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
                
                # Get color for this class
                if use_random_colors and class_id < len(colors):
                    color = colors[class_id]
                else:
                    color = default_color
                
                # Get confidence
                confidence = float(det.conf.item()) if hasattr(det, 'conf') else 0.0
                
                # Get the OBB polygon points
                if hasattr(det, 'xyxyxyxy'):
                    # YOLO11-OBB format with 4 points (x1,y1,x2,y2,x3,y3,x4,y4)
                    points = det.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2).astype(np.int32)[0]
                elif hasattr(det, 'xywhr'):
                    # xywhr format (center_x, center_y, width, height, rotation_radians)
                    cx, cy, w, h, angle = det.xywhr.cpu().numpy()[0]
                    # Convert to polygon points
                    rect = ((cx, cy), (w, h), angle * 180 / np.pi)  # OpenCV uses degrees
                    points = cv2.boxPoints(rect).astype(np.int32)
                else:
                    # Try to find any format that might work
                    for attr in dir(det):
                        if attr.startswith('xy') and not callable(getattr(det, attr)):
                            try:
                                points_data = getattr(det, attr).cpu().numpy()
                                if points_data.size >= 8:  # At least 4 points (x,y)
                                    points = points_data.reshape(-1, 4, 2).astype(np.int32)[0]
                                    break
                            except:
                                continue
                    else:
                        # Fallback if no suitable format found
                        print(f"Warning: Could not determine OBB format for {class_name}")
                        continue
                
                # Draw oriented bounding box as a polygon
                cv2.polylines(annotated_frame, [points], True, color, 2)
                
                # Get top-left point for label placement (use the minimum x and y from points)
                min_x = min(points[:, 0])
                min_y = min(points[:, 1])
                
                # Display label and confidence
                label = f"{class_name}: {confidence:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (min_x, min_y - text_size[1] - 5), (min_x + text_size[0], min_y), color, -1)
                cv2.putText(annotated_frame, label, (min_x, min_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Process standard bounding boxes
    elif has_boxes:
        for i, box in enumerate(result.boxes):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = float(box.conf[0])
            
            # Get class ID and name if available
            if hasattr(box, 'cls') and box.cls is not None and len(box.cls) > 0:
                class_id = int(box.cls[0])
                if hasattr(model, 'names') and class_id in model.names:
                    class_name = model.names[class_id]
                else:
                    class_name = f"Class {class_id}"
                
                # Update class counts
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
                
                # Get color for this class
                if use_random_colors and class_id < len(colors):
                    color = colors[class_id]
                else:
                    color = default_color
            else:
                class_id = 0
                class_name = "Object"
                color = default_color
                
                # Update generic object count
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
            
            # Draw bounding box (only if not already drawn for segmentation)
            if not has_masks:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            else:
                # For segmentation models, just draw a thin border
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
            
            # Display label and confidence
            label = f"{class_name}: {confidence:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Draw landmarks if available
    has_landmarks = hasattr(result, 'keypoints') and result.keypoints is not None
    if has_landmarks:
        try:
            kpts = result.keypoints.data
            for i in range(len(kpts)):
                landmarks = kpts[i].cpu().numpy()
                for lm in landmarks:
                    x, y = int(lm[0]), int(lm[1])
                    if x > 0 and y > 0 and (len(lm) < 3 or lm[2] > 0.2):  # Only draw if valid
                        cv2.circle(annotated_frame, (x, y), 4, (0, 255, 255), -1)  # Yellow dots
        except Exception as e:
            print(f"Error drawing landmarks: {e}")
    
    return annotated_frame, class_counts

def detect_model_type(model_path, model):
    """Detect the type of YOLO model based on name and attributes"""
    model_name = os.path.basename(model_path).lower()
    
    # Check model name for clues
    if 'seg' in model_name:
        return 'segmentation'
    elif 'obb' in model_name:
        return 'obb'
    elif 'pose' in model_name:
        return 'pose'
    
    # Check model task attribute if available
    if hasattr(model, 'task'):
        task = model.task.lower() if hasattr(model.task, 'lower') else str(model.task).lower()
        if 'segment' in task:
            return 'segmentation'
        elif 'pose' in task:
            return 'pose'
        elif 'obb' in task:
            return 'obb'
        elif 'detect' in task:
            return 'detection'
    
    # Default to detection
    return 'detection'

def main():
    print("Starting Universal YOLO Detection Demo")
    # Parse command line arguments
    args = parse_arguments()
    print(f"Arguments parsed: mode={args.mode}, source={args.source}")
    
    # Ensure the mode has .pt extension if not provided
    if not args.mode.endswith('.pt'):
        args.mode += '.pt'
    
    # Parse color setting
    if args.color == 'random':
        use_random_colors = True
        default_color = (0, 255, 255)  # Default yellow if no classes
    else:
        use_random_colors = False
        try:
            default_color = tuple(map(int, args.color.split(',')))
        except:
            print(f"Invalid color format: {args.color}. Using default yellow.")
            default_color = (0, 255, 255)
    
    # Load the model with full error traceback
    try:
        print(f"Attempting to load model: {args.mode}")
        if not os.path.exists(args.mode):
            print(f"❌ Model file not found: {args.mode}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in directory: {os.listdir('.')}")
            return
            
        model = YOLO(args.mode)
        print(f"✅ Model {args.mode} loaded successfully")
        
        # Detect model type
        model_type = detect_model_type(args.mode, model)
        print(f"🔍 Detected model type: {model_type}")
        
        # Print model information if requested
        if args.info or args.verbose:
            print("\n📊 Model Information:")
            print(f"  - Task: {model.task if hasattr(model, 'task') else 'Unknown'}")
            if hasattr(model, 'names') and model.names:
                print(f"  - Classes ({len(model.names)}):")
                for idx, name in model.names.items():
                    print(f"    {idx}: {name}")
            else:
                print("  - No class names found in model")
            
            print(f"  - Model type: {type(model).__name__}")
            if args.verbose:
                print(f"  - Available attributes: {dir(model)[:10]}...")
                print(f"  - Supports keypoints: {'keypoints' in dir(model)}")
    except Exception as e:
        import traceback
        print(f"❌ Error loading model: {e}")
        print(f"Detailed traceback:")
        traceback.print_exc()
        print(f"Make sure the model file {args.mode} exists and is a valid YOLO model.")
        return
    
    # Generate colors for classes if the model has class names
    if hasattr(model, 'names') and model.names:
        num_classes = max(model.names.keys()) + 1
        colors = generate_colors(num_classes)
    else:
        colors = [default_color]  # Just use the default color
    
    # Open the video source
    try:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise Exception(f"Could not open video source: {args.source}")
    except Exception as e:
        print(f"❌ Error opening video source: {e}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video source: {args.source}, Resolution: {width}x{height}, FPS: {fps:.1f}")
    
    # Initialize video writer if output is specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"📝 Recording output to: {args.output}")
    
    # FPS calculation variables
    prev_time = 0
    fps_avg = 0
    frame_count = 0
    
    print(f"\n🚀 Running {args.mode} model detection. Press 'q' to quit.")
    
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        
        if not success:
            print("End of video stream")
            break
        
        # Calculate FPS
        current_time = time.time()
        fps_current = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # Update average FPS
        frame_count += 1
        fps_avg = (fps_avg * (frame_count - 1) + fps_current) / frame_count
        
        # Run YOLOv8 inference on the frame with error handling
        try:
            results = model(frame, conf=args.conf, verbose=False)
            result = results[0]
            
            # Draw detections and get class counts
            annotated_frame, class_counts = draw_detections(
                frame, result, colors, default_color, use_random_colors, model, args.alpha
            )
            
            # Print detection details if verbose mode is enabled
            if args.verbose and class_counts:
                print(f"Frame {frame_count}: Detected {sum(class_counts.values())} objects - {', '.join([f'{v} {k}' for k, v in class_counts.items()])}")
        
        except Exception as e:
            print(f"Error during inference or drawing: {e}")
            import traceback
            traceback.print_exc()
            class_counts = {}
            annotated_frame = frame.copy()
            print("Skipping this frame due to error")
            continue
        
        # Add FPS info to the frame
        cv2.putText(annotated_frame, f"FPS: {fps_current:.1f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add detection counts to the frame if available
        if 'class_counts' in locals() and class_counts:
            y_pos = 80
            for class_name, count in class_counts.items():
                cv2.putText(annotated_frame, f"{class_name}: {count}", (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_pos += 40
        
        # Display the annotated frame
        cv2.imshow(f"YOLO - {os.path.basename(args.mode)}", annotated_frame)
        
        # Write the frame to the output video if specified
        if out is not None:
            out.write(annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n📊 Detection Summary:")
    print(f"  - Processed {frame_count} frames")
    print(f"  - Average FPS: {fps_avg:.2f}")
    if 'class_counts' in locals() and class_counts:
        print("  - Detected objects:")
        for class_name, count in class_counts.items():
            print(f"    {class_name}: {count}")
    
    print("✅ Detection completed")

if __name__ == "__main__":
    main()