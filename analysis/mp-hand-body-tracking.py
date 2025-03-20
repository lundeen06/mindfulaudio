"""
VIDEO ANALYSIS WITH MEDIAPIPE (HANDS AND UPPER BODY ONLY)
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

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up MediaPipe models with configurations
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Define landmarks to exclude (indices 0-10 for face, 23-32 for legs)
# 0-10: Facial landmarks (nose, eyes, ears, mouth)
# 23-32: Leg landmarks (hips, knees, ankles, feet)
FACIAL_LANDMARKS = set(range(11))  # Exclude landmarks 0-10
LEG_LANDMARKS = set(range(23, 33))  # Exclude landmarks 23-32
EXCLUDED_LANDMARKS = FACIAL_LANDMARKS.union(LEG_LANDMARKS)

def analyze_video(video_path):
    """
    Analyze a video file and track hands and upper body (excluding facial landmarks and legs).
    Save tracking data and generate plots.
    """
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Analyzing video: {video_name}")
    
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

    # Create output directories if they don't exist
    os.makedirs('./output', exist_ok=True)
    os.makedirs(f'./output/{video_name}', exist_ok=True)
    os.makedirs(f'./output/{video_name}/plots', exist_ok=True)
    
    # Initialize video writer to save annotated video
    output_video_path = f'./output/{video_name}/{video_name}_analyzed_{timestamp}.mp4'
    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    # Create CSV files for tracking data
    hand_csv_path = f'./output/{video_name}/{video_name}_hands_{timestamp}.csv'
    pose_csv_path = f'./output/{video_name}/{video_name}_pose_{timestamp}.csv'
    
    hand_csv_file = open(hand_csv_path, 'w', newline='')
    pose_csv_file = open(pose_csv_path, 'w', newline='')
    
    hand_csv_writer = csv.writer(hand_csv_file)
    pose_csv_writer = csv.writer(pose_csv_file)
    
    # Write CSV headers
    # Hand header (21 landmarks per hand)
    hand_header = ['timestamp', 'frame']
    for hand_idx in range(2):
        for i in range(21):
            hand_header.extend([f'hand{hand_idx+1}_landmark{i}_x', f'hand{hand_idx+1}_landmark{i}_y', f'hand{hand_idx+1}_landmark{i}_z'])
    hand_csv_writer.writerow(hand_header)
    
    # Pose header (exclude facial and leg landmarks)
    pose_header = ['timestamp', 'frame']
    for i in range(33):
        if i not in EXCLUDED_LANDMARKS:  # Skip facial and leg landmarks
            pose_header.extend([f'pose_landmark{i}_x', f'pose_landmark{i}_y', f'pose_landmark{i}_z', f'pose_landmark{i}_visibility'])
    pose_csv_writer.writerow(pose_header)
    
    # Initialize progress tracking
    start_time = time.time()
    frame_count = 0
    
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
            
            # Process all tracking in sequence
            hand_results = hands.process(image_rgb)
            pose_results = pose.process(image_rgb)
            
            # Make image writeable again for drawing
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Calculate timestamp based on frame count and FPS
            timestamp_seconds = frame_count / fps
            
            # Process hand landmarks
            hand_row = [timestamp_seconds, frame_count]
            hand_data = [''] * (21 * 3 * 2)  # 21 landmarks * 3 coords * 2 hands
            
            if hand_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    if hand_idx >= 2:
                        break  # Only process up to 2 hands
                    
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Record landmark data
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        base_idx = hand_idx * 63 + i * 3
                        hand_data[base_idx] = landmark.x
                        hand_data[base_idx + 1] = landmark.y
                        hand_data[base_idx + 2] = landmark.z
            
            # Add hand data to CSV
            hand_row.extend(hand_data)
            hand_csv_writer.writerow(hand_row)
            
            # Process body pose (excluding facial landmarks)
            pose_row = [timestamp_seconds, frame_count]
            
            # Only track non-facial, non-leg landmarks
            non_facial_leg_landmarks_count = 33 - len(EXCLUDED_LANDMARKS)
            pose_data = [''] * (non_facial_leg_landmarks_count * 4)  # non-excluded landmarks * 4 values (x, y, z, visibility)
            
            if pose_results.pose_landmarks:
                # Create a custom connection list excluding facial and leg landmarks
                custom_connections = []
                for connection in mp_pose.POSE_CONNECTIONS:
                    # Only keep connections where both points are not in excluded landmarks
                    if connection[0] not in EXCLUDED_LANDMARKS and connection[1] not in EXCLUDED_LANDMARKS:
                        custom_connections.append(connection)
                
                # No need to create a modified landmarks object, we'll just filter while drawing
                
                # Draw only the body landmarks (excluding facial landmarks)
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    # Only process non-facial, non-leg landmarks
                    if idx not in EXCLUDED_LANDMARKS:
                        # Add landmark to CSV data
                        data_idx = sum(1 for i in range(idx) if i not in EXCLUDED_LANDMARKS) * 4
                        pose_data[data_idx] = landmark.x
                        pose_data[data_idx + 1] = landmark.y
                        pose_data[data_idx + 2] = landmark.z
                        pose_data[data_idx + 3] = landmark.visibility
                
                # Create a custom drawing function to avoid drawing facial landmarks
                for connection in custom_connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    if start_idx not in EXCLUDED_LANDMARKS and end_idx not in EXCLUDED_LANDMARKS:
                        start_point = pose_results.pose_landmarks.landmark[start_idx]
                        end_point = pose_results.pose_landmarks.landmark[end_idx]
                        
                        start_px = int(start_point.x * frame_width)
                        start_py = int(start_point.y * frame_height)
                        end_px = int(end_point.x * frame_width)
                        end_py = int(end_point.y * frame_height)
                        
                        cv2.line(image, (start_px, start_py), (end_px, end_py), (0, 255, 0), 2)
                
                # Draw body landmarks (excluding facial and leg landmarks)
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    if idx not in EXCLUDED_LANDMARKS:
                        px = int(landmark.x * frame_width)
                        py = int(landmark.y * frame_height)
                        cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
            
            # Add pose data to CSV
            pose_row.extend(pose_data)
            pose_csv_writer.writerow(pose_row)
            
            # Add tracking information to the image
            tracking_text = []
            tracking_text.append(f"Video: {video_name}")
            tracking_text.append(f"Frame: {frame_count}/{total_frames}")
            
            if hand_results.multi_hand_landmarks:
                tracking_text.append(f"Hands: {len(hand_results.multi_hand_landmarks)}")
            if pose_results.pose_landmarks:
                tracking_text.append("Body: Detected (excluding face and legs)")
            
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
        print(f"An error occurred during video processing: {e}")
    
    finally:
        # Close resources
        cap.release()
        video_writer.release()
        
        hand_csv_file.close()
        pose_csv_file.close()
        
        # Generate plots
        generate_plots(hand_csv_path, pose_csv_path, video_name, timestamp)
        
        print(f"Video saved as: {output_video_path}")
        print(f"Hand landmark data saved as: {hand_csv_path}")
        print(f"Body pose data saved as: {pose_csv_path}")

def generate_plots(hand_csv_path, pose_csv_path, video_name, timestamp):
    """Generate plots from tracking data"""
    plot_dir = f'./output/{video_name}/plots/'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot hand movement
    try:
        if os.path.exists(hand_csv_path) and os.path.getsize(hand_csv_path) > 0:
            hand_df = pd.read_csv(hand_csv_path)
            if not hand_df.empty:
                # Get columns containing 'hand1_landmark' and specific landmarks (e.g., index finger tip)
                wrist_cols = [col for col in hand_df.columns if 'hand1_landmark0_' in col]
                index_tip_cols = [col for col in hand_df.columns if 'hand1_landmark8_' in col]
                thumb_tip_cols = [col for col in hand_df.columns if 'hand1_landmark4_' in col]
                
                if wrist_cols and index_tip_cols and thumb_tip_cols:
                    plt.figure(figsize=(12, 8))
                    
                    # Plot X-coordinate movement of key landmarks over time
                    plt.subplot(3, 1, 1)
                    plt.title('Hand Movement Over Time (X-coordinate)')
                    plt.plot(hand_df['timestamp'], hand_df[wrist_cols[0]], label='Wrist')
                    plt.plot(hand_df['timestamp'], hand_df[index_tip_cols[0]], label='Index Tip')
                    plt.plot(hand_df['timestamp'], hand_df[thumb_tip_cols[0]], label='Thumb Tip')
                    plt.xlabel('Time (s)')
                    plt.ylabel('X Position (normalized)')
                    plt.legend()
                    plt.grid(True)
                    
                    # Plot Y-coordinate movement
                    plt.subplot(3, 1, 2)
                    plt.title('Hand Movement Over Time (Y-coordinate)')
                    plt.plot(hand_df['timestamp'], hand_df[wrist_cols[1]], label='Wrist')
                    plt.plot(hand_df['timestamp'], hand_df[index_tip_cols[1]], label='Index Tip')
                    plt.plot(hand_df['timestamp'], hand_df[thumb_tip_cols[1]], label='Thumb Tip')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Y Position (normalized)')
                    plt.legend()
                    plt.grid(True)
                    
                    # Plot Z-coordinate (depth) movement
                    plt.subplot(3, 1, 3)
                    plt.title('Hand Movement Over Time (Z-coordinate)')
                    plt.plot(hand_df['timestamp'], hand_df[wrist_cols[2]], label='Wrist')
                    plt.plot(hand_df['timestamp'], hand_df[index_tip_cols[2]], label='Index Tip')
                    plt.plot(hand_df['timestamp'], hand_df[thumb_tip_cols[2]], label='Thumb Tip')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Z Position (normalized)')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(f'{plot_dir}/hand_movement.png')
                    plt.close()
                    
                    print(f"Hand movement plots saved to {plot_dir}")
    except Exception as e:
        print(f"Error generating hand plots: {e}")

    # Plot body pose (using non-facial landmarks)
    try:
        if os.path.exists(pose_csv_path) and os.path.getsize(pose_csv_path) > 0:
            pose_df = pd.read_csv(pose_csv_path)
            if not pose_df.empty:
                # Select key body landmarks (shoulders, wrists)
                right_shoulder_x = [col for col in pose_df.columns if 'pose_landmark12_x' in col]
                left_shoulder_x = [col for col in pose_df.columns if 'pose_landmark11_x' in col]
                right_wrist_x = [col for col in pose_df.columns if 'pose_landmark16_x' in col]
                left_wrist_x = [col for col in pose_df.columns if 'pose_landmark15_x' in col]
                
                if right_shoulder_x and left_shoulder_x and right_wrist_x and left_wrist_x:
                    plt.figure(figsize=(12, 6))
                    plt.title('Upper Body Movement Over Time (X-coordinate)')
                    plt.plot(pose_df['timestamp'], pose_df[right_shoulder_x[0]], label='Right Shoulder')
                    plt.plot(pose_df['timestamp'], pose_df[left_shoulder_x[0]], label='Left Shoulder')
                    plt.plot(pose_df['timestamp'], pose_df[right_wrist_x[0]], label='Right Wrist')
                    plt.plot(pose_df['timestamp'], pose_df[left_wrist_x[0]], label='Left Wrist')
                    plt.xlabel('Time (s)')
                    plt.ylabel('X Position (normalized)')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f'{plot_dir}/body_movement_x.png')
                    plt.close()
                    
                    print(f"Body movement plot saved to {plot_dir}")
    except Exception as e:
        print(f"Error generating pose plots: {e}")
    
    # Generate summary plot
    try:
        plt.figure(figsize=(12, 10))
        plt.suptitle(f'Motion Analysis Summary - {video_name}', fontsize=16)
        
        # Plot detection status over time
        plot_idx = 1
        
        # Hand detection
        if os.path.exists(hand_csv_path) and os.path.getsize(hand_csv_path) > 0:
            hand_df = pd.read_csv(hand_csv_path)
            if not hand_df.empty:
                wrist_x = [col for col in hand_df.columns if 'hand1_landmark0_x' in col]
                if wrist_x:
                    hand_detected = ~hand_df[wrist_x[0]].isna()
                    
                    plt.subplot(2, 1, plot_idx)
                    plt.title('Hand Detection Status')
                    plt.plot(hand_df['timestamp'], hand_detected.astype(int), 'g-')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Detected')
                    plt.yticks([0, 1], ['No', 'Yes'])
                    plt.grid(True)
                    plot_idx += 1
        
        # Body detection (excluding face)
        if os.path.exists(pose_csv_path) and os.path.getsize(pose_csv_path) > 0:
            pose_df = pd.read_csv(pose_csv_path)
            if not pose_df.empty:
                right_shoulder_x = [col for col in pose_df.columns if 'pose_landmark12_x' in col]
                if right_shoulder_x:
                    pose_detected = ~pose_df[right_shoulder_x[0]].isna()
                    
                    plt.subplot(2, 1, plot_idx)
                    plt.title('Body Detection Status (excluding face and legs)')
                    plt.plot(pose_df['timestamp'], pose_detected.astype(int), 'r-')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Detected')
                    plt.yticks([0, 1], ['No', 'Yes'])
                    plt.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        plt.savefig(f'{plot_dir}/detection_summary.png')
        plt.close()
        
        print(f"Summary plot saved to {plot_dir}")
    except Exception as e:
        print(f"Error generating summary plot: {e}")

def main():
    """Main function to process all videos in the videos directory"""
    print("Starting video analysis (excluding facial and leg landmarks)...")
    
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
        hands.close()
        pose.close()
        
        print("Script execution completed.")