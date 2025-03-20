import numpy as np
import cv2
import glob
import os
import json

def print_intrinsic_diagnostics(camera_matrix, dist_coeffs, img_size):
    """Print detailed intrinsic parameter analysis."""
    fx, fy = camera_matrix[0,0], camera_matrix[1,1]
    cx, cy = camera_matrix[0,2], camera_matrix[1,2]
    skew = camera_matrix[0,1]
    
    # Calculate FOV
    fov_x = 2 * np.arctan(img_size[1]/(2*fx)) * 180/np.pi
    fov_y = 2 * np.arctan(img_size[0]/(2*fy)) * 180/np.pi
    
    print("\nIntrinsic Parameters:")
    print("-" * 50)
    print(f"Focal Length (pixels): fx={fx:.1f}, fy={fy:.1f}")
    print(f"Principal Point: cx={cx:.1f}, cy={cy:.1f}")
    print(f"Skew: {skew:.6f}")
    print(f"Field of View: {fov_x:.1f}° x {fov_y:.1f}°")
    
    print("\nDistortion Coefficients:")
    print(f"k1, k2, p1, p2, k3 = {dist_coeffs.ravel()}")

def print_stereo_diagnostics(R, T, cam1_name, cam2_name):
    """Print detailed stereo parameter analysis."""
    # Convert rotation matrix to Euler angles
    euler_angles = cv2.Rodrigues(R)[0].ravel() * 180/np.pi
    
    # Get baseline (translation) in millimeters
    T_mm = T.ravel() * 1000
    baseline = np.linalg.norm(T_mm)
    
    print("\nExtrinsic Parameters (Camera Pair Relationship):")
    print("-" * 50)
    print(f"Cameras: {cam1_name} → {cam2_name}")
    print("\nRotation (degrees):")
    print(f"Roll:  {euler_angles[0]:.2f}°")
    print(f"Pitch: {euler_angles[1]:.2f}°")
    print(f"Yaw:   {euler_angles[2]:.2f}°")
    
    print("\nTranslation (mm):")
    print(f"X: {T_mm[0]:.1f}")
    print(f"Y: {T_mm[1]:.1f}")
    print(f"Z: {T_mm[2]:.1f}")
    print(f"Baseline distance: {baseline:.1f}mm")
    
    # Calculate relative position in human-readable form
    directions = []
    if abs(T_mm[0]) > 5:
        directions.append(f"{'right' if T_mm[0] > 0 else 'left'} ({abs(T_mm[0]):.1f}mm)")
    if abs(T_mm[1]) > 5:
        directions.append(f"{'down' if T_mm[1] > 0 else 'up'} ({abs(T_mm[1]):.1f}mm)")
    if abs(T_mm[2]) > 5:
        directions.append(f"{'forward' if T_mm[2] > 0 else 'backward'} ({abs(T_mm[2]):.1f}mm)")
    
    print(f"\nRelative Position: Camera {cam2_name} is", " and ".join(directions), f"from {cam1_name}")

class MultiCameraCalibrator:
    def __init__(self, checkerboard_size=(8,6), square_size=0.029):
        """
        Initialize calibrator for N cameras
        checkerboard_size: tuple of (width, height) interior points
        square_size: size of square in meters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ..., (7,5,0)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2)
        self.objp *= square_size  # Convert to meters
        
        # Termination criteria for cornerSubPix
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def find_corners(self, img, draw=False):
        """Find checkerboard corners in image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
            if draw:
                cv2.drawChessboardCorners(img, self.checkerboard_size, corners, ret)
            return True, corners
        return False, None

    def calibrate_camera(self, images):
        """Calibrate single camera given list of images."""
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        for img in images:
            ret, corners = self.find_corners(img)
            if ret:
                objpoints.append(self.objp)
                imgpoints.append(corners)
        
        if not objpoints:
            return None, None, None, None, None
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, images[0].shape[:2][::-1], None, None)
        
        return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints

    def compute_reprojection_error(self, objpoints, imgpoints, mtx, dist, rvecs, tvecs):
        """Compute reprojection error."""
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        return mean_error/len(objpoints)

    def calibrate_stereo(self, mtx1, dist1, mtx2, dist2, images1, images2):
        """Calibrate stereo pair."""
        objpoints = []
        imgpoints1 = []
        imgpoints2 = []
        
        for img1, img2 in zip(images1, images2):
            ret1, corners1 = self.find_corners(img1)
            ret2, corners2 = self.find_corners(img2)
            
            if ret1 and ret2:  # If checkerboard found in both images
                objpoints.append(self.objp)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
        
        if not objpoints:
            return None, None, None, None, None
            
        # Calibrate stereo cameras
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            mtx1, dist1, mtx2, dist2,
            images1[0].shape[:2][::-1], None, None,
            flags=cv2.CALIB_FIX_INTRINSIC)
            
        return R, T, E, F, ret

def calibrate_all_cameras(base_dir="calibration_images"):
    """Calibrate all cameras in the directory."""
    # Find all camera directories
    cam_dirs = sorted(glob.glob(os.path.join(base_dir, "cam*")))
    if not cam_dirs:
        print("No camera directories found!")
        return
    
    print(f"Found {len(cam_dirs)} cameras: {[os.path.basename(d) for d in cam_dirs]}")
    print("=" * 80)
    print("CALIBRATION REPORT")
    print("=" * 80)
    
    # Initialize calibrator
    calibrator = MultiCameraCalibrator()
    
    # Store results
    results = {
        "single_camera": {},
        "stereo_pairs": {}
    }
    
    # Single camera calibration
    camera_matrices = {}
    distortion_coeffs = {}
    all_images = {}
    
    print("\nPART 1: INTRINSIC CALIBRATION")
    print("=" * 80)
    
    for cam_dir in cam_dirs:
        cam_name = os.path.basename(cam_dir)
        print(f"\nCalibrating {cam_name}...")
        
        # Load images
        image_files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
        images = [cv2.imread(f) for f in image_files]
        all_images[cam_name] = images
        
        if not images:
            print(f"No images found for {cam_name}")
            continue
            
        # Calibrate
        ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = calibrator.calibrate_camera(images)
        
        if ret is None:
            print(f"Calibration failed for {cam_name}")
            continue
            
        # Compute reprojection error
        mean_error = calibrator.compute_reprojection_error(
            objpoints, imgpoints, mtx, dist, rvecs, tvecs)
            
        # Store results
        camera_matrices[cam_name] = mtx
        distortion_coeffs[cam_name] = dist
        
        results["single_camera"][cam_name] = {
            "camera_matrix": mtx.tolist(),
            "dist_coeffs": dist.tolist(),
            "reprojection_error": float(mean_error)
        }
        
        # Print detailed diagnostics
        print(f"\nResults for {cam_name}:")
        print_intrinsic_diagnostics(mtx, dist, images[0].shape[:2])
        print(f"Reprojection error: {mean_error:.6f} pixels")
    
    print("\nPART 2: EXTRINSIC CALIBRATION")
    print("=" * 80)
    print("\nCalibrating camera pairs and computing relative positions...")
    
    # Stereo calibration for each pair
    cam_names = list(camera_matrices.keys())
    for i in range(len(cam_names)):
        for j in range(i+1, len(cam_names)):
            cam1, cam2 = cam_names[i], cam_names[j]
            print(f"\nCalibrating pair: {cam1} - {cam2}")
            
            R, T, E, F, ret = calibrator.calibrate_stereo(
                camera_matrices[cam1], distortion_coeffs[cam1],
                camera_matrices[cam2], distortion_coeffs[cam2],
                all_images[cam1], all_images[cam2]
            )
            
            if ret is None:
                print(f"Stereo calibration failed for {cam1} - {cam2}")
                continue
                
            results["stereo_pairs"][f"{cam1}_{cam2}"] = {
                "R": R.tolist(),
                "T": T.tolist(),
                "E": E.tolist(),
                "F": F.tolist(),
                "error": float(ret)
            }
            
            # Print detailed stereo diagnostics
            print_stereo_diagnostics(R, T, cam1, cam2)
            print(f"Stereo calibration error: {ret:.6f}")
    
    # Save results
    with open("./output/calibration_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nCalibration complete! Results saved to calibration_results.json")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate N cameras using OpenCV")
    parser.add_argument("--image_dir", default="calibration_images",
                       help="Base directory containing camera folders")
    
    args = parser.parse_args()
    
    results = calibrate_all_cameras(args.image_dir)