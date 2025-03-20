import numpy as np
import cv2
import os
from datetime import datetime

def setup_cameras(camera_ids):
    """Initialize multiple cameras."""
    caps = []
    for cam_id in camera_ids:
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"Failed to open camera {cam_id}")
            return None
        # Minimize frame buffer for better synchronization
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        caps.append(cap)
    return caps

def detect_checkerboard(frame, board_size=(8, 6)):
    """Detect checkerboard in frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    
    if ret:
        # Refine corner detection
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Draw corners for visualization
        cv2.drawChessboardCorners(frame, board_size, corners, ret)
    return ret, frame

def capture_calibration_frames(camera_ids, num_frames=30):
    """Capture synchronized frames from N cameras."""
    # Setup output directories
    base_dir = "calibration_images"
    for cam_id in camera_ids:
        os.makedirs(os.path.join(base_dir, f"cam{cam_id}"), exist_ok=True)

    # Initialize cameras
    caps = setup_cameras(camera_ids)
    if caps is None:
        return False

    frame_count = 0
    frame_idx = 1  # Start from image01.jpg

    print("\nCalibration frame capture:")
    print(f"Capturing {num_frames} synchronized frames")
    print("Press 's' to save frames (will check for checkerboard)")
    print("Press 'q' to quit\n")

    while frame_count < num_frames:
        frames = []
        displays = []

        # Capture frames from all cameras
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                continue
            frames.append(frame)
            displays.append(frame.copy())  # No checkerboard detection here

        # Create visualization of all cameras
        all_displays = np.hstack(displays)
        
        # Add capture counter
        cv2.putText(all_displays, 
                   f"Frames: {frame_count}/{num_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)

        cv2.imshow('All Cameras', all_displays)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Only check for checkerboard when trying to save
            corner_detected = []
            preview_frames = []
            
            # Check all frames for checkerboard
            for frame in frames:
                ret, preview = detect_checkerboard(frame.copy())
                corner_detected.append(ret)
                preview_frames.append(preview)
            
            # Show preview with detected corners
            preview_display = np.hstack(preview_frames)
            cv2.putText(preview_display, 
                       "Checking for checkerboard... Press any key to continue", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            cv2.imshow('Checkerboard Detection', preview_display)
            cv2.waitKey(0)  # Wait for user to see the detection
            cv2.destroyWindow('Checkerboard Detection')
            
            if all(corner_detected):
                # Save frames with format image01.jpg, image02.jpg, etc.
                frame_name = f"image{frame_idx:02d}.jpg"
                
                for i, frame in enumerate(frames):
                    save_path = os.path.join(base_dir, f"cam{camera_ids[i]}", frame_name)
                    cv2.imwrite(save_path, frame)
                
                print(f"Saved frame set {frame_count + 1}/{num_frames} as {frame_name}")
                frame_count += 1
                frame_idx += 1
            else:
                print("Checkerboard not detected in all views! Try again.")
        
        elif key == ord('q'):
            break

    # Cleanup
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Capture calibration images from N cameras')
    parser.add_argument('--camera_ids', type=int, nargs='+', required=True,
                      help='List of camera IDs (e.g., 0 1 2 for three cameras)')
    parser.add_argument('--num_frames', type=int, default=30,
                      help='Number of frame sets to capture')
    
    args = parser.parse_args()
    
    capture_calibration_frames(args.camera_ids, args.num_frames)