#!/usr/bin/env python3
"""
Face Detection Test Script using OpenCV and Camera
Works with USB cameras and Raspberry Pi camera via /dev/video0
"""

import cv2
import time
import numpy as np

# Initialize camera (Try multiple backends for Raspberry Pi camera)
print("Initializing Camera...")
cap = None

# Try V4L2 backend first (best for Raspberry Pi camera)
print("Trying V4L2 backend on /dev/video0...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("V4L2 failed, trying default backend...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    print("Make sure your camera is connected and enabled:")
    print("  sudo raspi-config -> Interface Options -> Camera")
    exit(1)

# Set resolution (MJPEG format works better for Pi camera)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print(f"Camera opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# Load OpenCV cascade classifiers
print("Loading cascade classifiers...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

if face_cascade.empty():
    print("Error: Could not load face cascade classifier")
    exit(1)

print("Starting face detection...")
print("Press 'q' to exit\n")

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Convert to grayscale for cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            maxSize=(500, 500)
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            # Draw face rectangle (green)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Detect eyes within face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # Draw rectangles around detected eyes
            for (ex, ey, ew, eh) in eyes[:2]:  # Limit to 2 eyes
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        
        # Add FPS counter
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Display frame
        cv2.imshow('Face Detection - OpenCV', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")
