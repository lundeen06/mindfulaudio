"""
FACE TRACKING WITH MEDIAPIPE

TODO: Doesn't work for front view due to distance + lighting
"""

import cv2
import mediapipe as mp
import numpy as np
import datetime
import time
import csv
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import math

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up MediaPipe face mesh with configurations
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Face mesh key points and their meanings
FACE_KEY_POINTS = {
    # Face oval
    10: "chin",
    152: "right_cheek",
    234: "left_cheek",
    
    # Eyes
    33: "right_eye_inner",
    133: "right_eye_outer",
    362: "left_eye_inner",
    263: "left_eye_outer",
    
    # Eyebrows
    105: "right_eyebrow",
    334: "left_eyebrow",
    
    # Nose
    4: "nose_tip",
    197: "nose_bridge",
    
    # Mouth
    0: "mouth_top",
    17: "mouth_bottom",
    61: "mouth_right",
    291: "mouth_left"
}

# Define eye landmarks for aspect ratio calculation
# These are specific points in the MediaPipe face mesh
RIGHT_EYE_LANDMARKS = [
    # Upper right eyelid
    159, 145, 
    # Lower right eyelid
    33, 133
]

LEFT_EYE_LANDMARKS = [
    # Upper left eyelid
    386, 374, 
    # Lower left eyelid
    362, 263
]

# Define mouth landmarks for aspect ratio calculation
MOUTH_LANDMARKS = [
    # Top lip
    13,
    # Bottom lip
    14,
    # Right corner
    78,
    # Left corner
    308
]

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two 3D points"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_eye_aspect_ratio(landmarks, eye_points):
    """Calculate eye aspect ratio: height/width"""
    if not landmarks:
        return 0
        
    # Calculate vertical distance (height)
    height1 = calculate_distance(landmarks[eye_points[0]], landmarks[eye_points[1]])
    
    # Calculate horizontal distance (width)
    width = calculate_distance(landmarks[eye_points[2]], landmarks[eye_points[3]])
    
    # Calculate EAR
    if width > 0:
        return height1 / width
    return 0

def calculate_mouth_aspect_ratio(landmarks, mouth_points):
    """Calculate mouth aspect ratio: height/width"""
    if not landmarks:
        return 0
        
    # Calculate vertical distance (height)
    height = calculate_distance(landmarks[mouth_points[0]], landmarks[mouth_points[1]])
    
    # Calculate horizontal distance (width)
    width = calculate_distance(landmarks[mouth_points[2]], landmarks[mouth_points[3]])
    
    # Calculate MAR
    if width > 0:
        return height / width
    return 0

def analyze_video(video_path):
    """
    Analyze a video file and track facial landmarks.
    Save tracking data and generate plots.
    """
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Analyzing facial features in video: {video_name}")

    # Create output directories if they don't exist
    os.makedirs('./output', exist_ok=True)
    os.makedirs(f'./output/{video_name}', exist_ok=True)
    os.makedirs(f'./output/{video_name}/plots', exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize video writer to save annotated video
    output_video_path = f'./output/{video_name}/{video_name}_face_{timestamp}.mp4'
    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    # Create CSV file for face tracking data
    face_csv_path = f'./output/{video_name}/{video_name}_face_landmarks_{timestamp}.csv'
    face_csv_file = open(face_csv_path, 'w', newline='')
    face_csv_writer = csv.writer(face_csv_file)
    
    # Write CSV header
    face_header = ['timestamp', 'frame', 'face_detected']
    
    # Add columns for all tracked facial landmarks
    for landmark_idx, landmark_name in FACE_KEY_POINTS.items():
        face_header.extend([f'{landmark_name}_x', f'{landmark_name}_y', f'{landmark_name}_z'])
    
    # Add columns for advanced metrics
    face_header.extend([
        'eye_aspect_ratio',      # Measure of eye openness
        'mouth_aspect_ratio',    # Measure of mouth openness
        'face_tilt',             # Head tilt angle
        'face_yaw'               # Face rotation left/right
    ])
    
    face_csv_writer.writerow(face_header)
    
    # Initialize progress tracking
    start_time = time.time()
    frame_count = 0
    
    # Keep track of face detection history for plots
    timestamps = []
    face_detected_history = []
    eye_aspect_ratios = []
    mouth_aspect_ratios = []
    face_tilt_history = []
    
    try:
        while cap.isOpened():
            # Read frame
            success, image = cap.read()
            if not success:
                # End of video
                break
            
            # Convert from BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            image_rgb.flags.writeable = False
            face_results = face_mesh.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Calculate timestamp based on frame count and FPS
            timestamp_seconds = frame_count / fps
            timestamps.append(timestamp_seconds)
            
            # Initialize row for CSV
            face_row = [timestamp_seconds, frame_count]
            
            # Check if face detected
            face_detected = False
            
            # Initialize face metrics
            eye_aspect_ratio = 0
            mouth_aspect_ratio = 0
            face_tilt = 0
            face_yaw = 0
            
            # Process face landmarks if detected
            if face_results.multi_face_landmarks:
                face_detected = True
                face_detected_history.append(1)
                
                face_landmarks = face_results.multi_face_landmarks[0]  # Get first face
                
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Draw face contours
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                
                face_row.append(1)  # Face detected
                
                # Record landmark data for key points
                for landmark_idx, landmark_name in FACE_KEY_POINTS.items():
                    if landmark_idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[landmark_idx]
                        face_row.extend([landmark.x, landmark.y, landmark.z])
                    else:
                        face_row.extend(['', '', ''])
                
                # Calculate eye aspect ratio
                right_ear = calculate_eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_LANDMARKS)
                left_ear = calculate_eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_LANDMARKS)
                eye_aspect_ratio = (right_ear + left_ear) / 2
                eye_aspect_ratios.append(eye_aspect_ratio)
                
                # Calculate mouth aspect ratio
                mouth_aspect_ratio = calculate_mouth_aspect_ratio(face_landmarks.landmark, MOUTH_LANDMARKS)
                mouth_aspect_ratios.append(mouth_aspect_ratio)
                
                # Calculate face tilt (using eyes as reference)
                left_eye = face_landmarks.landmark[LEFT_EYE_LANDMARKS[3]]
                right_eye = face_landmarks.landmark[RIGHT_EYE_LANDMARKS[3]]
                face_tilt = math.degrees(math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x))
                face_tilt_history.append(face_tilt)
                
                # Calculate face yaw (using nose and eyes as reference)
                nose_tip = face_landmarks.landmark[4]
                face_center_x = (left_eye.x + right_eye.x) / 2
                face_yaw = (nose_tip.x - face_center_x) * 100  # Scale for better visualization
                
                # Add metrics to CSV row
                face_row.extend([eye_aspect_ratio, mouth_aspect_ratio, face_tilt, face_yaw])
                
                # Draw key metrics on the image
                metrics_text = [
                    f"Eye AR: {eye_aspect_ratio:.2f}",
                    f"Mouth AR: {mouth_aspect_ratio:.2f}",
                    f"Face Tilt: {face_tilt:.1f}°"
                ]
                
                y_pos = frame_height - 120
                for text in metrics_text:
                    cv2.putText(image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_pos += 30
            else:
                face_detected_history.append(0)
                face_row.append(0)  # Face not detected
                
                # Fill with empty values for landmarks and metrics
                for _ in range(len(FACE_KEY_POINTS) * 3 + 4):
                    face_row.append('')
                
                eye_aspect_ratios.append(np.nan)
                mouth_aspect_ratios.append(np.nan)
                face_tilt_history.append(np.nan)
            
            # Add face tracking info to the image
            tracking_text = []
            tracking_text.append(f"Video: {video_name}")
            tracking_text.append(f"Frame: {frame_count}/{total_frames}")
            tracking_text.append(f"Face detected: {'Yes' if face_detected else 'No'}")
            
            # Display tracking status
            y_pos = 30
            for text in tracking_text:
                cv2.putText(image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
            
            # Add progress indicator
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps_display = frame_count / elapsed_time
                estimated_total_time = total_frames / fps_display if fps_display > 0 else 0
                remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
                
                progress_percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                
                progress_text = f"Progress: {progress_percent:.1f}% | Est. remaining: {remaining_time:.1f}s"
                cv2.putText(image, progress_text, (10, frame_height - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write CSV row
            face_csv_writer.writerow(face_row)
            
            # Add frame to video
            video_writer.write(image)
            
            # Increment frame counter
            frame_count += 1
            
            # Print progress update every 100 frames
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
        
        print(f"Finished processing video: {video_name}")
    
    except Exception as e:
        print(f"An error occurred during face tracking: {e}")
    
    finally:
        # Close resources
        cap.release()
        video_writer.release()
        face_csv_file.close()
        
        # Generate plots
        plot_dir = f'./output/{video_name}/plots/'
        os.makedirs(plot_dir, exist_ok=True)
        
        # Generate plots if we have enough data
        if len(timestamps) > 0:
            try:
                # Plot face detection over time
                plt.figure(figsize=(12, 6))
                plt.title('Face Detection Status')
                plt.plot(timestamps, face_detected_history, 'g-')
                plt.xlabel('Time (s)')
                plt.ylabel('Detected')
                plt.yticks([0, 1], ['No', 'Yes'])
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{plot_dir}/face_detection.png')
                plt.close()
                
                # Plot eye aspect ratio over time
                plt.figure(figsize=(12, 6))
                plt.title('Eye Aspect Ratio (Higher = More Open)')
                plt.plot(timestamps, eye_aspect_ratios, 'b-')
                plt.xlabel('Time (s)')
                plt.ylabel('Eye Aspect Ratio')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{plot_dir}/eye_aspect_ratio.png')
                plt.close()
                
                # Plot mouth aspect ratio over time
                plt.figure(figsize=(12, 6))
                plt.title('Mouth Aspect Ratio (Higher = More Open)')
                plt.plot(timestamps, mouth_aspect_ratios, 'r-')
                plt.xlabel('Time (s)')
                plt.ylabel('Mouth Aspect Ratio')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{plot_dir}/mouth_aspect_ratio.png')
                plt.close()
                
                # Plot face tilt over time
                plt.figure(figsize=(12, 6))
                plt.title('Face Tilt Angle')
                plt.plot(timestamps, face_tilt_history, 'm-')
                plt.xlabel('Time (s)')
                plt.ylabel('Tilt Angle (degrees)')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{plot_dir}/face_tilt.png')
                plt.close()
                
                # Create summary dashboard
                plt.figure(figsize=(15, 10))
                plt.suptitle(f'Face Analysis Dashboard - {video_name}', fontsize=16)
                
                # Face detection
                plt.subplot(2, 2, 1)
                plt.title('Face Detection')
                plt.plot(timestamps, face_detected_history, 'g-')
                plt.xlabel('Time (s)')
                plt.ylabel('Detected')
                plt.yticks([0, 1], ['No', 'Yes'])
                plt.grid(True)
                
                # Eye aspect ratio
                plt.subplot(2, 2, 2)
                plt.title('Eye Openness')
                plt.plot(timestamps, eye_aspect_ratios, 'b-')
                plt.xlabel('Time (s)')
                plt.ylabel('Eye Aspect Ratio')
                plt.grid(True)
                
                # Mouth aspect ratio
                plt.subplot(2, 2, 3)
                plt.title('Mouth Openness')
                plt.plot(timestamps, mouth_aspect_ratios, 'r-')
                plt.xlabel('Time (s)')
                plt.ylabel('Mouth Aspect Ratio')
                plt.grid(True)
                
                # Face tilt
                plt.subplot(2, 2, 4)
                plt.title('Head Tilt')
                plt.plot(timestamps, face_tilt_history, 'm-')
                plt.xlabel('Time (s)')
                plt.ylabel('Tilt Angle (degrees)')
                plt.grid(True)
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(f'{plot_dir}/face_analysis_dashboard.png')
                plt.close()
                
                print(f"Face analysis plots saved to {plot_dir}")
            except Exception as e:
                print(f"Error generating plots: {e}")
        
        print(f"Video saved as: {output_video_path}")
        print(f"Face landmark data saved as: {face_csv_path}")

def main():
    """Main function to process all videos in the videos directory"""
    print("Starting face tracking analysis...")
    
    # Look for video files in the videos directory
    video_folder = "./videos"
    if not os.path.exists(video_folder):
        print(f"Creating videos folder: {video_folder}")
        os.makedirs(video_folder)
        print("Please put video files in the videos folder and run the script again.")
        return
    
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    video_files.extend(glob.glob(os.path.join(video_folder, "*.avi")))
    video_files.extend(glob.glob(os.path.join(video_folder, "*.mov")))
    
    if not video_files:
        print("No video files found in the videos folder.")
        print("Please add video files (MP4, AVI, MOV) to the videos folder.")
        return
    
    print(f"Found {len(video_files)} video files.")
    
    # Process each video
    for i, video_path in enumerate(video_files):
        print(f"Processing video {i+1}/{len(video_files)}: {video_path}")
        analyze_video(video_path)
    
    print("All videos processed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up MediaPipe resources
        face_mesh.close()
        
        print("Face tracking script execution completed.")