# Video Recording System

This directory contains tools for synchronized recording from multiple cameras.

## Overview

The `sync_multicam.py` script provides functionality for recording synchronized video streams from multiple cameras. This ensures that frames from all cameras are captured at approximately the same time, which is essential for accurate multi-view tracking and 3D reconstruction.

## Features

- **Automatic camera detection** - Detects available cameras connected to the system
- **Synchronized recording** - Minimizes timing differences between camera frames
- **Metadata storage** - Saves timestamp information for post-processing synchronization
- **Resource-efficient** - Designed to use minimal RAM compared to alternatives

## Requirements

```bash
pip install opencv-python numpy
```

## Usage

### Basic Recording

To start recording with all detected cameras:

```bash
python sync_multicam.py
```

The script will:
1. Detect available cameras
2. Initialize recording for each camera
3. Begin capturing synchronized frames
4. Save individual video files for each camera
5. Save synchronization metadata

### Stopping Recording

Press `Ctrl+C` to stop recording. The script will properly close all files and save metadata.

## Output Files

Recording output is saved to the `recordings/` directory:

- **Video files**: `camera_{id}_{timestamp}.mp4` for each camera
- **Metadata file**: `sync_metadata_{timestamp}.json`

### Metadata Structure

The metadata JSON file contains:
- Camera information (ID, resolution, FPS)
- Frame timestamps for all cameras
- Frame correspondence information

Example metadata structure:

```json
{
  "camera_info": [
    {
      "id": 0,
      "width": 1280,
      "height": 720,
      "fps": 30.0
    },
    {
      "id": 1,
      "width": 1280,
      "height": 720,
      "fps": 30.0
    }
  ],
  "timestamps": [
    {
      "camera_id": 0,
      "timestamp": 1613419862.123,
      "frame_number": 0
    },
    {
      "camera_id": 1,
      "timestamp": 1613419862.125,
      "frame_number": 0
    },
    ...
  ]
}
```

## Technical Details

### Synchronization Approach

The script uses the following approach to maximize synchronization:

1. **Buffer minimization**: Sets camera buffer size to 1 to reduce latency
2. **Threaded capture**: Each camera runs in its own thread to capture frames as quickly as possible
3. **Timestamp recording**: Captures precise timestamps when each frame is grabbed
4. **Frame queues**: Uses queues to handle different camera frame rates

### Implementation Notes

- The script uses Python's `threading` library for parallel frame grabbing
- OpenCV's `VideoCapture` and `VideoWriter` handle frame capture and writing
- System timestamps are used for synchronization metadata

## Troubleshooting

### Common Issues

- **Camera not detected**: Ensure cameras are properly connected and working. Run `../utils/camera-list.py` to verify camera indices.
- **Out of memory errors**: Reduce the resolution of your cameras or use fewer cameras.
- **Frame dropping**: Ensure your system has adequate USB bandwidth for all cameras. Consider using a powered USB hub.

### Performance Considerations

- Recording multiple high-resolution cameras simultaneously requires substantial processing power and I/O bandwidth
- For best results, use a system with:
  - Multiple USB controllers (to distribute camera bandwidth)
  - SSD storage (for faster write speeds)
  - 16+ GB RAM