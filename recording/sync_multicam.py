# motivation: skellycam uses RAM intensely

import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from threading import Thread, Event, Lock
from queue import Queue
import json

class CameraThread(Thread):
    def __init__(self, camera_id, frame_queue, timestamp_queue, stop_event, frame_ready_event):
        super().__init__()
        self.camera_id = camera_id
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.stop_event = stop_event
        self.frame_ready_event = frame_ready_event
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # Set camera properties for better synchronization
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
        
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                timestamp = time.time()
                self.frame_queue.put((self.camera_id, frame))
                self.timestamp_queue.put((self.camera_id, timestamp))
                self.frame_ready_event.set()
            else:
                break

        self.cap.release()

class SynchronizedRecorder:
    def __init__(self, output_dir="recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.camera_threads = []
        self.frame_queues = []
        self.timestamp_queues = []
        self.video_writers = []
        self.stop_event = Event()
        self.frame_ready_event = Event()
        self.write_lock = Lock()
        
        # Metadata for synchronization
        self.timestamps = []
        self.camera_info = []

    def detect_cameras(self):
        """Detect available cameras and initialize them"""
        camera_ids = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    camera_ids.append(i)
                    # Store camera info
                    self.camera_info.append({
                        'id': i,
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': cap.get(cv2.CAP_PROP_FPS)
                    })
                cap.release()
        return camera_ids

    def initialize_recording(self, camera_ids):
        """Initialize recording for all detected cameras"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for camera_id in camera_ids:
            # Create queues for each camera
            frame_queue = Queue()
            timestamp_queue = Queue()
            self.frame_queues.append(frame_queue)
            self.timestamp_queues.append(timestamp_queue)
            
            # Create video writer
            output_file = self.output_dir / f"camera_{camera_id}_{timestamp}.mp4"
            camera_info = next(info for info in self.camera_info if info['id'] == camera_id)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_file),
                fourcc,
                camera_info['fps'],
                (camera_info['width'], camera_info['height'])
            )
            self.video_writers.append(writer)
            
            # Create and start camera thread
            thread = CameraThread(
                camera_id,
                frame_queue,
                timestamp_queue,
                self.stop_event,
                self.frame_ready_event
            )
            self.camera_threads.append(thread)
            thread.start()

    def record(self, duration=None):
        """Record synchronized video from all cameras"""
        camera_ids = self.detect_cameras()
        if not camera_ids:
            print("No cameras detected!")
            return

        print(f"Detected cameras: {camera_ids}")
        self.initialize_recording(camera_ids)
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Wait for at least one camera to have a frame ready
                self.frame_ready_event.wait(timeout=1.0)
                self.frame_ready_event.clear()
                
                # Process frames from all cameras
                with self.write_lock:
                    for i, queue in enumerate(self.frame_queues):
                        if not queue.empty():
                            camera_id, frame = queue.get()
                            timestamp = self.timestamp_queues[i].get()[1]
                            
                            # Write frame to video file
                            self.video_writers[i].write(frame)
                            
                            # Store timestamp for synchronization metadata
                            self.timestamps.append({
                                'camera_id': camera_id,
                                'timestamp': timestamp,
                                'frame_number': frame_count
                            })
                            
                            # Display progress
                            if i == 0:  # Only count frames from first camera
                                frame_count += 1
                                if frame_count % 30 == 0:  # Update every 30 frames
                                    print(f"Recorded {frame_count} frames...")
                
        except KeyboardInterrupt:
            print("\nRecording stopped by user")
        finally:
            self.stop_recording()

    def stop_recording(self):
        """Stop recording and clean up resources"""
        self.stop_event.set()
        
        # Wait for all threads to finish
        for thread in self.camera_threads:
            thread.join()
        
        # Release video writers
        for writer in self.video_writers:
            writer.release()
        
        # Save synchronization metadata
        metadata = {
            'camera_info': self.camera_info,
            'timestamps': self.timestamps
        }
        
        metadata_file = self.output_dir / f"sync_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nRecording completed. Files saved to {self.output_dir}")
        print(f"Synchronization metadata saved to {metadata_file}")

if __name__ == "__main__":
    # Example usage
    recorder = SynchronizedRecorder()
    
    # Record until ctrl c hit
    recorder.record()