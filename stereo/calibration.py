"""
N CAMERA CALIBRATION 

SOURCES: 
https://github.com/TemugeB/python_stereo_camera_calibrate
    - stereo camera calibration example (constrained to 2 cameras)
"""

import numpy as np
import cv2
import glob
import os

# Constants
CALIB_CONFIG = {
    'CHECKERBOARD_SIZE': (8, 6),    # interior points
    'SQUARE_SIZE': 29 * 1e-3,       # meters 
    'NUM_CALIB_FRAMES': 30,         # number of synchronized frames to capture
    'IMAGE_DIR': 'calibration_images',
}

CAMERA_CONFIG = {
    'NUM_CAMERAS': 2,               # adjust based on your setup
    'CAMERA_IDS': [0, 1],           # adjust based on your camera indices
    'REFERENCE_CAM': 0,             # camera to use as world origin
}

def capture_synchronized_frames():
    caps = []
    for cam_id in CAMERA_CONFIG['CAMERA_IDS']:
        cap = cv2.VideoCapture(cam_id)
        # Add these lines to potentially improve capture performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
        caps.append(cap)
    
    os.makedirs(CALIB_CONFIG['IMAGE_DIR'], exist_ok=True)
    frames_captured = 0
    
    print("Position checkerboard visible to all cameras")
    print("Press 's' to save synchronized frames when corners detected in all views")
    print("Press 'q' to quit")
    
    while frames_captured < CALIB_CONFIG['NUM_CALIB_FRAMES']:
        frames = []
        displays = []
        
        # First just capture and display frames without corner detection
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                return False
            
            # Optional: resize for display
            # display = cv2.resize(frame, (640, 480))
            frames.append(frame)
            displays.append(frame)
        
        # Create display without corner detection
        all_displays = np.hstack(displays)
        cv2.putText(all_displays, f"Frames: {frames_captured}/{CALIB_CONFIG['NUM_CALIB_FRAMES']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('All Cameras', all_displays)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Only do corner detection when saving
            corner_detected = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, CALIB_CONFIG['CHECKERBOARD_SIZE'], None)
                corner_detected.append(ret)
            
            if all(corner_detected):
                # Save synchronized frames
                for i, frame in enumerate(frames):
                    filename = os.path.join(CALIB_CONFIG['IMAGE_DIR'], f'frame_{frames_captured}_cam_{i}.png')
                    cv2.imwrite(filename, frame)
                print(f"Saved frame set {frames_captured}")
                frames_captured += 1
            else:
                print("Corners not detected in all views")
        elif key == ord('q'):
            break
    
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    return True

def calibrate_stereo_system():
    """
    Calibrate the multi-camera system with one camera as world reference
    """
    # Prepare object points
    objp = np.zeros((CALIB_CONFIG['CHECKERBOARD_SIZE'][0] * CALIB_CONFIG['CHECKERBOARD_SIZE'][1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CALIB_CONFIG['CHECKERBOARD_SIZE'][0], 
                         0:CALIB_CONFIG['CHECKERBOARD_SIZE'][1]].T.reshape(-1,2)
    objp = objp * CALIB_CONFIG['SQUARE_SIZE']
    
    # Initialize storage for all cameras
    objpoints = [] # 3D points in world space
    imgpoints = [[] for _ in range(CAMERA_CONFIG['NUM_CAMERAS'])] # 2D points in image plane for each camera
    
    # Load all synchronized frames
    for frame_idx in range(CALIB_CONFIG['NUM_CALIB_FRAMES']):
        corner_detected = []
        current_imgpoints = []
        
        for cam_idx in range(CAMERA_CONFIG['NUM_CAMERAS']):
            fname = os.path.join(CALIB_CONFIG['IMAGE_DIR'], f'frame_{frame_idx}_cam_{cam_idx}.png')
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, CALIB_CONFIG['CHECKERBOARD_SIZE'], None)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                current_imgpoints.append(corners)
                corner_detected.append(True)
            else:
                corner_detected.append(False)
        
        if all(corner_detected):
            objpoints.append(objp)
            for cam_idx, corners in enumerate(current_imgpoints):
                imgpoints[cam_idx].append(corners)
    
    # Calibrate each camera individually first
    camera_matrices = []
    dist_coeffs = []
    for cam_idx in range(CAMERA_CONFIG['NUM_CAMERAS']):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints[cam_idx], gray.shape[::-1], None, None)
        camera_matrices.append(mtx)
        dist_coeffs.append(dist)
    
    # Compute transforms relative to reference camera
    transforms = []  # R, t transforms relative to reference camera
    for cam_idx in range(1, CAMERA_CONFIG['NUM_CAMERAS']):
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints[CAMERA_CONFIG['REFERENCE_CAM']], imgpoints[cam_idx],
            camera_matrices[CAMERA_CONFIG['REFERENCE_CAM']], dist_coeffs[CAMERA_CONFIG['REFERENCE_CAM']],
            camera_matrices[cam_idx], dist_coeffs[cam_idx],
            gray.shape[::-1], None, None,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        transforms.append((R, T))
    
    return camera_matrices, dist_coeffs, transforms

def validate_calibration(camera_matrices, dist_coeffs, transforms, image_shape):
    """
    Print and validate all calibration parameters
    """
    print("\n=== CALIBRATION VALIDATION ===")
    
    # Intrinsic Parameters
    for i, K in enumerate(camera_matrices):
        print(f"\nCamera {i} Intrinsics:")
        print("Focal lengths (fx, fy):", K[0,0], K[1,1])
        print("Principal point (cx, cy):", K[0,2], K[1,2])
        print("Skew:", K[0,1])
        print("\nDistortion coefficients:", dist_coeffs[i].ravel())
        
        # Calculate FOV
        fovx = 2 * np.arctan(image_shape[1]/(2*K[0,0])) * 180/np.pi
        fovy = 2 * np.arctan(image_shape[0]/(2*K[1,1])) * 180/np.pi
        print(f"FOV (degrees) - horizontal: {fovx:.1f}, vertical: {fovy:.1f}")
    
    # Extrinsic Parameters (Stereo transform)
    for i, (R, T) in enumerate(transforms):
        print(f"\nStereo Transform - Reference Camera to Camera {i+1}:")
        
        # Convert rotation matrix to euler angles
        euler_angles = cv2.Rodrigues(R)[0].ravel() * 180/np.pi
        print("Rotation (degrees):", euler_angles)
        
        # Translation in mm for easier interpretation
        T_mm = T.ravel() * 1000
        print("Translation (mm):", T_mm)
        
        # Baseline (distance between cameras)
        baseline = np.linalg.norm(T) * 1000
        print(f"Baseline distance: {baseline:.1f}mm")

def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    Compute reprojection error for a camera
    """
    total_error = 0
    for i in range(len(objpoints)):
        projected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                              camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2)/len(projected_points)
        total_error += error
    return total_error/len(objpoints)

def estimate_depth_accuracy(camera_matrices, translations, image_shape):
    """
    Estimate theoretical depth accuracy for stereo setup
    """
    # Use first camera's parameters
    f = camera_matrices[0][0,0]  # focal length in pixels
    B = np.linalg.norm(translations[0]) * 1000  # baseline in mm
    pixel_error = 0.5  # assume 0.5 pixel matching error
    
    # Calculate depth error for different distances
    distances = [500, 1000, 2000, 3000, 10000]  # mm
    print("\nTheoretical Depth Accuracy:")
    print("Distance(mm) | Error(mm) | Error(%)")
    print("-" * 40)
    for Z in distances:
        error = (Z**2 * pixel_error)/(f * B)
        error_percent = (error/Z) * 100
        print(f"{Z:>10.0f} | {error:>9.1f} | {error_percent:>8.1f}")

if __name__ == "__main__":
    # First capture synchronized calibration frames
    if capture_synchronized_frames():
        # Then calibrate the system
        camera_matrices, dist_coeffs, transforms = calibrate_stereo_system()
        
        # Get image shape from a sample image
        sample_img = cv2.imread(os.path.join(CALIB_CONFIG['IMAGE_DIR'], 'frame_0_cam_0.png'))
        image_shape = sample_img.shape[:2]
        
        # Validate calibration
        validate_calibration(camera_matrices, dist_coeffs, transforms, image_shape)
        
        # Estimate depth accuracy
        rotations = [R for R, _ in transforms]
        translations = [T for _, T in transforms]
        estimate_depth_accuracy(camera_matrices, translations, image_shape)
        
        # Save calibration results
        np.savez('stereo_calibration.npz',
                 camera_matrices=camera_matrices,
                 dist_coeffs=dist_coeffs,
                 rotations=rotations,
                 translations=translations,
                 reference_cam=CAMERA_CONFIG['REFERENCE_CAM'],
                 image_shape=image_shape)