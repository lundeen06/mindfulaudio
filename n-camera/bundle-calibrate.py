import numpy as np
import cv2
import glob
import os
import json
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

class BundleAdjustmentCalibrator:
    def __init__(self, checkerboard_size=(8,6), square_size=0.029):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Create 3D object points for the checkerboard
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2)
        self.objp *= square_size
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def find_corners(self, img):
        """Find checkerboard corners in image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
            return True, corners
        return False, None

    def get_initial_params(self, images_dict):
        """Get initial camera parameters using OpenCV's calibration."""
        camera_params = {}
        detection_data = {}
        
        # First do individual calibrations
        for cam_id, images in images_dict.items():
            objpoints = []
            imgpoints = []
            
            for img in images:
                ret, corners = self.find_corners(img)
                if ret:
                    objpoints.append(self.objp)
                    imgpoints.append(corners)
            
            if not objpoints:
                print(f"No corners detected for camera {cam_id}")
                continue
                
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, images[0].shape[:2][::-1], None, None)
            
            # For non-reference cameras, use stereo calibration with reference
            if cam_id == 0:
                R = np.eye(3)
                t = np.zeros((3, 1))
            else:
                # Do stereo calibration with reference camera
                _, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
                    objpoints, 
                    detection_data[0]['imgpoints'], 
                    imgpoints,
                    camera_params[0]['K'], 
                    camera_params[0]['dist'],
                    mtx, 
                    dist,
                    images[0].shape[:2][::-1],
                    None, None,
                    flags=cv2.CALIB_FIX_INTRINSIC
                )
            
            camera_params[cam_id] = {
                'K': mtx,
                'dist': dist,
                'R': R,
                't': t
            }
            
            detection_data[cam_id] = {
                'objpoints': objpoints,
                'imgpoints': imgpoints
            }
            
            print(f"Camera {cam_id} initialized with {len(objpoints)} valid frames")
        
        return camera_params, detection_data

    def params_to_vector(self, camera_params):
        """Convert camera parameters to optimization vector."""
        params = []
        
        # First camera is reference (identity rotation, zero translation)
        params.extend([0, 0, 0, 0, 0, 0])  # R,t for first camera
        
        # Add other cameras' parameters
        for cam_id in sorted(camera_params.keys())[1:]:
            rvec = cv2.Rodrigues(camera_params[cam_id]['R'])[0].ravel()
            tvec = camera_params[cam_id]['t'].ravel()
            params.extend(rvec)
            params.extend(tvec)
        
        return np.array(params, dtype=np.float64)

    def vector_to_params(self, vec, num_cameras):
        """Convert optimization vector back to camera parameters."""
        camera_params = {}
        
        # First camera is reference
        camera_params[0] = {
            'R': np.eye(3),
            't': np.zeros((3, 1))
        }
        
        # Convert parameters for other cameras
        idx = 6  # Skip reference camera parameters
        for i in range(1, num_cameras):
            rvec = vec[idx:idx+3]
            tvec = vec[idx+3:idx+6]
            R = cv2.Rodrigues(rvec)[0]
            t = tvec.reshape(3, 1)
            camera_params[i] = {'R': R, 't': t}
            idx += 6
            
        return camera_params

    def project_points(self, points_3d, R, t, K, dist):
        """Project 3D points using camera parameters."""
        points_3d = np.array(points_3d, dtype=np.float64)
        rvec = cv2.Rodrigues(R)[0]
        points_2d, _ = cv2.projectPoints(points_3d, rvec, t, K, dist)
        return points_2d.reshape(-1, 2)

    def bundle_adjustment_cost(self, params, camera_matrices, dist_coeffs, detection_data, num_cameras):
        """Cost function for bundle adjustment."""
        camera_params = self.vector_to_params(params, num_cameras)
        total_error = []
        
        for cam_id in range(num_cameras):
            if cam_id not in detection_data:
                continue
                
            K = camera_matrices[cam_id]
            dist = dist_coeffs[cam_id]
            R = camera_params[cam_id]['R']
            t = camera_params[cam_id]['t']
            
            for objpoints, imgpoints in zip(detection_data[cam_id]['objpoints'],
                                          detection_data[cam_id]['imgpoints']):
                projected = self.project_points(objpoints, R, t, K, dist)
                error = (projected - imgpoints.reshape(-1, 2)).ravel()
                total_error.extend(error)
        
        return np.array(total_error)

    def optimize_cameras(self, camera_params, detection_data):
        """Perform bundle adjustment optimization."""
        num_cameras = len(camera_params)
        
        # Get initial parameters vector
        initial_params = self.params_to_vector(camera_params)
        
        # Extract fixed parameters
        camera_matrices = {cam_id: params['K'] for cam_id, params in camera_params.items()}
        dist_coeffs = {cam_id: params['dist'] for cam_id, params in camera_params.items()}
        
        # Run optimization
        result = least_squares(
            self.bundle_adjustment_cost,
            initial_params,
            args=(camera_matrices, dist_coeffs, detection_data, num_cameras),
            method='trf',  # Trust Region Reflective algorithm
            loss='huber',  # Robust loss function
            ftol=1e-8,
            xtol=1e-8,
            verbose=2
        )
        
        # Convert optimized parameters back
        optimized_params = self.vector_to_params(result.x, num_cameras)
        
        # Calculate final reprojection error
        final_error = np.sqrt(np.mean(np.square(result.fun)))
        
        return optimized_params, final_error

def calibrate_cameras_bundle(base_dir="calibration_images"):
    """Perform bundle adjustment calibration for all cameras."""
    print("\n=== MULTI-CAMERA BUNDLE ADJUSTMENT CALIBRATION ===")
    
    # Find camera directories
    cam_dirs = sorted(glob.glob(os.path.join(base_dir, "cam*")))
    if len(cam_dirs) < 3:
        print("Need at least 3 cameras for bundle adjustment!")
        return
        
    print(f"Found {len(cam_dirs)} cameras: {[os.path.basename(d) for d in cam_dirs]}")
    
    # Initialize calibrator
    calibrator = BundleAdjustmentCalibrator()
    
    # Load images
    images_dict = {}
    for cam_dir in cam_dirs:
        cam_id = int(os.path.basename(cam_dir)[3:])  # Extract number from "camX"
        image_files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
        images_dict[cam_id] = [cv2.imread(f) for f in image_files]
    
    print("\nPART 1: INITIAL CALIBRATION")
    print("=" * 50)
    
    # Get initial parameters
    camera_params, detection_data = calibrator.get_initial_params(images_dict)
    
    print("\nPART 2: BUNDLE ADJUSTMENT")
    print("=" * 50)
    print("Optimizing all cameras simultaneously...")
    
    # Perform bundle adjustment
    optimized_params, final_cost = calibrator.optimize_cameras(camera_params, detection_data)
    
    # Save original camera parameters for comparison
    results = {
        "initial_params": {},
        "optimized_params": {},
        "optimization_error": float(np.mean(final_cost))
    }
    
    # Convert and save results
    for cam_id in camera_params.keys():
        # Initial parameters
        R_init = camera_params[cam_id]['R']
        t_init = camera_params[cam_id]['t']
        euler_init = Rotation.from_matrix(R_init).as_euler('xyz', degrees=True)
        
        results["initial_params"][f"cam{cam_id}"] = {
            "rotation_matrix": R_init.tolist(),
            "translation_vector": t_init.ravel().tolist(),
            "euler_angles_deg": euler_init.tolist(),
            "translation_mm": (t_init.ravel() * 1000).tolist()
        }
        
        # Optimized parameters
        if cam_id in optimized_params:
            R_opt = optimized_params[cam_id]['R']
            t_opt = optimized_params[cam_id]['t']
            euler_opt = Rotation.from_matrix(R_opt).as_euler('xyz', degrees=True)
            t_mm = t_opt.ravel() * 1000
            
            results["optimized_params"][f"cam{cam_id}"] = {
                "rotation_matrix": R_opt.tolist(),
                "translation_vector": t_opt.ravel().tolist(),
                "euler_angles_deg": euler_opt.tolist(),
                "translation_mm": t_mm.tolist()
            }
            
            # Print human-readable results
            print(f"\nCamera {cam_id} Final Parameters:")
            print("-" * 30)
            print(f"Rotation (degrees):")
            print(f"    Roll:  {euler_opt[0]:.2f}°")
            print(f"    Pitch: {euler_opt[1]:.2f}°")
            print(f"    Yaw:   {euler_opt[2]:.2f}°")
            print(f"Translation (mm):")
            print(f"    X: {t_mm[0]:.1f}")
            print(f"    Y: {t_mm[1]:.1f}")
            print(f"    Z: {t_mm[2]:.1f}")
            
            # Print relative positions for all cameras after camera 0
            if cam_id > 0:
                directions = []
                if abs(t_mm[0]) > 5:
                    directions.append(f"{'right' if t_mm[0] > 0 else 'left'} ({abs(t_mm[0]):.1f}mm)")
                if abs(t_mm[1]) > 5:
                    directions.append(f"{'down' if t_mm[1] > 0 else 'up'} ({abs(t_mm[1]):.1f}mm)")
                if abs(t_mm[2]) > 5:
                    directions.append(f"{'forward' if t_mm[2] > 0 else 'backward'} ({abs(t_mm[2]):.1f}mm)")
                
                print(f"\nRelative to Camera 0: ", " and ".join(directions))
    
    print(f"\nFinal average reprojection error: {np.sqrt(np.mean(final_cost)):.6f} pixels")
    
    # Save results
    with open("bundle_calibration_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nCalibration complete! Results saved to bundle_calibration_results.json")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bundle adjustment calibration for 3+ cameras")
    parser.add_argument("--image_dir", default="calibration_images",
                       help="Base directory containing camera folders")
    
    args = parser.parse_args()
    
    results = calibrate_cameras_bundle(args.image_dir)