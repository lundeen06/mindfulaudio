# MindfulAudio - Lundeen

## Project Overview

This repository contains tools for capturing and analyzing human movement and audio data in research settings. The project combines multi-camera computer vision with audio processing to study potential correlations between hand movements, body posture, facial expressions, and sound production.

## Key Features

- **Multi-camera calibration and synchronization**: Set up and calibrate multiple cameras for 3D tracking
- **Real-time body, hand, and face tracking**: Using MediaPipe and YOLO models
- **Audio analysis and visualization**: Spectral and amplitude analysis of audio recordings
- **Data visualization tools**: Interactive 3D visualization of camera setups and movement data

## Repository Structure

- **[analysis/](analysis/)**: Scripts for analyzing video and audio data
  - **[audio/](analysis/audio/)**: Audio analysis tools
  - **demos/**: YOLO model demo scripts
  - **custom/**: Custom model training scripts
- **[n-camera/](n-camera/)**: Multi-camera calibration and capture tools
- **[recording/](recording/)**: Video recording and synchronization tools
- **[stereo/](stereo/)**: Stereo camera calibration tools
- **[utils/](utils/)**: Utility scripts and visualization tools
- **[single-camera-hands.py](single-camera-hands-README.md)**: Single camera hand tracking

## Component-Specific Documentation

Each component has its own detailed README with specific setup instructions:

- [Analysis Tools README](analysis/README.md)
- [Audio Analysis README](analysis/audio/README.md)
- [Multi-Camera System README](n-camera/README.md)
- [Recording System README](recording/README.md)
- [Stereo Camera System README](stereo/README.md)
- [Utilities README](utils/README.md)
- [Single Camera Hand Tracking README](single-camera-hands-README.md)

## Getting Started

Due to the specialized nature of this project, different components have different dependencies. **Please refer to the README files in each subdirectory linked above for specific setup instructions.**

### General Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/mindfulAudio-lundeen.git
   cd mindfulAudio-lundeen
   ```

2. Create a virtual environment (recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies for the component you wish to use (see README files in component directories linked above)

## Quick Start Examples

### List Available Cameras
```
python utils/camera-list.py
```

### Run Audio Analysis
```
python analysis/audio/audio-analysis.py your_audio_file.mp3
```

### Run Hand Tracking on a Single Camera
```
python single-camera-hands.py
```

### Run Multi-Camera Calibration
```
python n-camera/capture.py --camera_ids 0 1 2
python n-camera/calibrate.py
```

## Contact
Lundeen Cahilly - [lcahilly@stanford.edu](lcahilly@stanford.edu)
