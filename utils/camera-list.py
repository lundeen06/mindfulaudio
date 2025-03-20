import cv2
import subprocess

def list_available_cameras():
    index = 0
    cameras = []
    
    names_dict = {}
    
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            name = names_dict.get(index, f"Camera {index}")
            cameras.append((index, name))
        cap.release()
        index += 1
    
    return cameras

available_cameras = list_available_cameras()
print("Available cameras:")
for index, name in available_cameras:
    print(f"Index {index}: {name}")