#!/usr/bin/env python3
"""
Face Detection Test Script using Picamera2 and OpenCV Cascades
"""

import cv2
import time
from picamera2 import Picamera2
from picamera2.presets import LibcameraPreset
import numpy as np

# Initialize camera
print("Initializing Picamera2...")
picam2 = Picamera2()

# Configure camera with preview
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    presets=LibcameraPreset.HighQuality
)
picam2.configure(config)
picam2.start()

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
print("Press Ctrl+C to exit\n")

frame_count = 0
start_time = time.time()

try:
    while True:
        # Capture frame from camera
        frame = picam2.capture_array()
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for cascade detection
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
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
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Detect eyes within face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame_bgr[y:y+h, x:x+w]
            
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
                frame_bgr,
                f"FPS: {fps:.1f} | Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Display frame
        cv2.imshow('Face Detection - Picamera2', frame_bgr)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    print("Cleaning up...")
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()
    print("Done!")
