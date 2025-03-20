# Stereo Camera System

This directory contains tools for calibrating and working with stereo camera pairs, which are useful for depth estimation and 3D reconstruction of hand movements and gestures.

## Overview

Stereo vision uses two or more cameras to reconstruct 3D information from 2D images. This directory provides tools to:

1. Calibrate a stereo camera system
2. Calculate theoretical depth accuracy
3. Save calibration parameters for later use

## Requirements

The stereo camera system has specific dependencies:

```bash
pip install -r requirements.txt
```

Or install the dependencies manually:

```bash
pip install opencv-python>=4.5.0 numpy>=1.19.0 matplotlib>=3.3.0 ultralytics
```

## Stereo Camera Calibration

The `calibration.py` script provides a comprehensive calibration workflow for stereo camera systems.

### Usage

```bash
python calibration.py
```

The script will guide you through the calibration process:

1. **Camera detection**: The script will detect connected cameras
2. **Capture synchronized frames**: Position a checkerboard pattern in front of both cameras
3. **Calibration**: Find intrinsic and extrinsic camera parameters
4. **Validation**: Calculate reprojection errors and display results
5. **Save calibration**: Store parameters for later use

### Calibration Output

The calibration produces the following information:

- **Intrinsic parameters** for each camera
  - Focal length
  - Principal point
  - Distortion coefficients
- **Extrinsic parameters** between cameras
  - Rotation matrix
  - Translation vector
- **Depth accuracy estimates**
  - Theoretical accuracy at different distances

Example depth accuracy output:

```
Theoretical Depth Accuracy:
Distance(mm) | Error(mm) | Error(%)
----------------------------------------
       500   |      4.2  |      0.8
      1000   |     16.7  |      1.7
      2000   |     66.8  |      3.3
      3000   |    150.3  |      5.0
     10000   |   1672.4  |     16.7
```

### Calibration File

The calibration results are saved to `stereo_calibration.npz` with the following structure:

```
camera_matrices: List of camera matrices
dist_coeffs: List of distortion coefficients
rotations: List of rotation matrices
translations: List of translation vectors
reference_cam: Reference camera index
image_shape: Image dimensions
```

## Theoretical Background

Stereo camera systems work on the principle of triangulation. The accuracy of depth estimation depends on:

1. **Baseline distance**: The distance between cameras
2. **Focal length**: The focal length of the cameras
3. **Disparity precision**: How accurately corresponding points can be matched

The relationship is approximately:

```
depth_error = (Z² * disparity_error) / (f * B)
```

Where:
- Z is the distance to the object
- f is the focal length in pixels
- B is the baseline in the same units as Z
- disparity_error is the error in pixel matching (typically 0.5-1 pixel)

## Tips for Best Results

- **Camera positioning**: Place cameras at 10-20cm apart for hand tracking
- **Lighting**: Ensure uniform, diffuse lighting to avoid glare
- **Calibration pattern**: Use a high-quality printed checkerboard pattern
- **Variety**: Capture calibration images with the pattern at various distances and angles
- **Stability**: Mount cameras firmly to prevent movement after calibration