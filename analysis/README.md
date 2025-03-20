# Analysis Tools

This directory contains scripts for analyzing video and audio data captured as part of the mindful audio research project.

## Directory Structure

- **audio/**: Audio analysis tools
- **custom/**: Custom model training scripts
- **demos/**: Demo scripts for object detection and pose estimation
- **mp-face-tracking.py**: MediaPipe-based face tracking script
- **mp-hand-body-tracking.py**: MediaPipe-based hand and body tracking script

## Audio Analysis

The `audio/` directory contains tools for analyzing audio recordings with a focus on spectral and amplitude features.

### Setup

```bash
pip install librosa numpy plotly pandas matplotlib
```

### Running Audio Analysis

```bash
python audio/audio-analysis.py path/to/your/audio/file.mp3 -o output_directory
```

### Example Output

The audio analysis generates several interactive visualizations:

**Amplitude Analysis**
![Amplitude Analysis](audio/output/pilot1.1_amplitude.html)

**Frequency Analysis**
![Frequency Analysis](audio/output/pilot1.1_frequency.html)

**Average Spectrum**
![Average Spectrum](audio/output/pilot1.1_spectrum.html)

Additionally, a JSON file containing the numerical analysis results is generated.

## MediaPipe Face Tracking

The `mp-face-tracking.py` script provides detailed face tracking functionality using MediaPipe.

### Setup

```bash
pip install mediapipe opencv-python numpy matplotlib pandas
```

### Usage

1. Place video files in a `videos` directory at the root level
2. Run the script:
   ```bash
   python mp-face-tracking.py
   ```

### Features

- Face landmark detection and tracking
- Eye aspect ratio calculation (measure of eye openness)
- Mouth aspect ratio calculation
- Head pose estimation
- Generation of annotated videos and CSV data files
- Visualization of tracking results over time

### Output

- Annotated video with face tracking
- CSV file with landmarks data
- Time series plots for:
  - Face detection status
  - Eye openness
  - Mouth openness
  - Head tilt angle

## MediaPipe Hand and Body Tracking

The `mp-hand-body-tracking.py` script tracks hands and upper body (excluding face and legs) using MediaPipe.

### Setup

Same dependencies as face tracking.

### Usage

1. Place video files in a `videos` directory at the root level
2. Run the script:
   ```bash
   python mp-hand-body-tracking.py
   ```

### Features

- Hand landmark detection and tracking
- Upper body pose tracking (excluding facial landmarks and legs)
- Generation of annotated videos and CSV data files
- Visualization of tracking results over time

### Output

- Annotated video with hand and body tracking
- CSV files with landmarks data
- Time series plots for:
  - Hand movement over time
  - Body movement over time
  - Detection status summary

## YOLO Object Detection Demos

The `demos/` directory contains script for object detection and pose estimation using YOLO models.

### Setup

```bash
pip install ultralytics opencv-python
```

### Usage

```bash
python demos/demo.py --source 0 --mode yolov8n.pt  # Run with webcam
python demos/demo.py --source path/to/video.mp4 --mode yolo11n-pose.pt  # Run with video file
```

### Available Models

- `yolo11n-pose.pt`: Pose estimation model
- `yolo11n-seg.pt`: Segmentation model
- `yolov8n-face.pt`: Face detection model
- `yolov8n.pt`: Object detection model

## Custom Model Training

The `custom/` directory contains scripts for training custom YOLO models on your own data.

### Setup

```bash
pip install ultralytics
```

### Training a Custom Model

1. Prepare your dataset in YOLO format
2. Edit the `custom/training.py` script to point to your dataset
3. Run the training script:
   ```bash
   python custom/training.py
   ```

The script will train a custom model based on the YOLO11n base model.