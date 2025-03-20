"""
SINGLE CAMERA HAND TRACKING W/ MEDIAPIPE

SOURCES: https://github.com/TemugeB/handpose3d
    - boilerplate for mediapipe approach to soln, simplified to 1 camera
"""

import cv2
import mediapipe as mp
import numpy as np
import datetime
import time
import csv

# Create mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,   # TODO: play with these parameters
    min_tracking_confidence=0.5)

# Initialize mediapipe joint drawing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open webcam (will change to calibrated cameras)
cap = cv2.VideoCapture(0)

# Get properties of webcam video output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Init video writer to save mp4 files
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f'./output/hand_tracking_{timestamp}.mp4'
video_writer = cv2.VideoWriter(
    video_filename,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

# Create csv file for landmark (joint) data
csv_filename = f'./output/hand_landmarks_{timestamp}.csv'
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)

# Write csv header
header = ['timestamp']
for i in range(21):  # 21 landmarks per hand (see below)
    header.extend([f'hand1_landmark{i}_x', f'hand1_landmark{i}_y', f'hand1_landmark{i}_z'])
    header.extend([f'hand2_landmark{i}_x', f'hand2_landmark{i}_y', f'hand2_landmark{i}_z'])
csv_writer.writerow(header)

start_time = time.time()

try: 
    while cap.isOpened():
        # Read frame
        success, image = cap.read()
        if not success: 
            print("failed to load frame from camera")
            continue

        # Convert from truecolor (bgr) to rgb
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hands (if hands)
        results = hands.process(image_rgb)

        # Prepare csv to add data
        current_time = time.time() - start_time
        row = [current_time]

        # Initialize hand data as a bunch of empty strings (126 of them)
        hand_data = [''] * 126

        # Draw hand joints + landmarks
        if results.multi_hand_landmarks:
            for hand_i, hand_landmarks in enumerate(results.multi_hand_landmarks): 
                if hand_i >= 2:
                    break   # Can only have 2 hands here
            
                # Draw!
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Get hand pose in 3d coords
                for i, landmark in enumerate(hand_landmarks.landmark):
                    # Normalized coordinates (x,y,z); mutliply by image dimensions to get pixel coords
                    j = hand_i * 63 + i * 3  # 63 values per hand (21 landmarks * 3 coordinates)
                    hand_data[j] = landmark.x
                    hand_data[j + 1] = landmark.y
                    hand_data[j + 2] = landmark.z
                    # z would be relative depth but single camera so not fully possible

        # Add hand data to csv
        row.extend(hand_data)
        csv_writer.writerow(row)

        # Add frame to mp4v
        video_writer.write(image)

        # Display image
        cv2.imshow('Hand Tracking', image)
        
        # Exit on q press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally: 
    # Close windows, save files
    hands.close()
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    csv_file.close()

    print(f"Video saved as: {video_filename}")
    print(f"Landmark data saved as: {csv_filename}")