#!/usr/bin/env python3
"""
Emotional Face Tracker with Head Servos and Eyes
Combines visual eyes (PIL rendering), face tracking (YuNet), and smooth head servo motion.
"""

import time
import math
import random
import sys
import io
import threading
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
from pathlib import Path

# Hardware / Display Imports
import board
import busio
import digitalio
from PIL import Image, ImageDraw
try:
    from adafruit_rgb_display import st7735
except ImportError:
    print("Error: adafruit-circuitpython-rgb-display not found.")
    print("pip3 install adafruit-circuitpython-rgb-display")
    sys.exit(1)

try:
    from adafruit_servokit import ServoKit
except ImportError:
    print("Error: adafruit-servokit not found.")
    print("pip3 install adafruit-servokit")
    sys.exit(1)

# Camera Import
try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 not found. Please install with: sudo apt install python3-picamera2")
    sys.exit(1)


# --- Configuration ---
SCREEN_WIDTH = 128
SCREEN_HEIGHT = 160
EYE_COLOR = (255, 255, 255)  # White
BG_COLOR = (0, 0, 0)      # Black
EYE_SIZE = 120
FLOOR_Y = SCREEN_HEIGHT - 5

# Camera / Face Tracking Config
FACE_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
CAMERA_MAIN_RES = (1920, 1080)
CAMERA_RES = (1280, 720)
STREAM_RES = (320, 180)
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
CAMERA_ROTATE_180 = True
STREAM_SWAP_RB = True

# Eye Interaction Config
MAX_X_OFFSET = 30
MAX_Y_OFFSET = 22
EYE_BOUND_MARGIN = 8

# Blink Speed
BLINK_SPEED_MIN = 2.0
BLINK_SPEED_MAX = 3.5
LOOK_SIDE_OFFSET = 16.0

# Distance-based behavior
CLOSE_FACE_AREA_RATIO = 0.05
FAR_FACE_AREA_RATIO = 0.018
FAR_SQUINT_CHANCE_PER_SEC = 0.35
FAR_SQUINT_GAP_MIN_SEC = 1.8
FAR_SQUINT_GAP_MAX_SEC = 4.2
FAR_SQUINT_MIN_SEC = 0.22
FAR_SQUINT_MAX_SEC = 0.55

# --- Servo Config ---
PAN_CH = 0
TILT_CH = 1
PAN_MIN = 40
PAN_MAX = 130
TILT_MIN = 80
TILT_MAX = 130
SERVO_SMOOTHING = 0.08  # Easing factor for smooth motion
SERVO_LOOP_DELAY = 0.01  # 50Hz servo update

# --- Emotion Presets ---
EMOTION_PRESETS = {
    "idle": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "happy": {"scale_w": 1.2, "scale_h": 0.65, "top_lid": 0.0, "bottom_lid": 0.55, "lid_angle": -12.0, "mirror_angle": True},
    "excited": {"scale_w": 1.06, "scale_h": 1.09, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "sad": {"scale_w": 1.1, "scale_h": 1.1, "top_lid": 0.35, "bottom_lid": 0.0, "lid_angle": 15.0, "mirror_angle": True},
    "angry": {"scale_w": 1.0, "scale_h": 0.9, "top_lid": 0.35, "bottom_lid": 0.0, "lid_angle": -20.0, "mirror_angle": True},
    "surprised": {"scale_w": 0.9, "scale_h": 1.3, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "suspicious": {"scale_w": 1.1, "scale_h": 0.6, "top_lid": 0.4, "bottom_lid": 0.4, "lid_angle": 0.0, "mirror_angle": True},
    "sleepy": {"scale_w": 1.1, "scale_h": 1.0, "top_lid": 0.6, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "looking_left": {"scale_w": 1.0, "scale_h": 0.9, "top_lid": 0.25, "bottom_lid": 0.0, "lid_angle": -8.0, "mirror_angle": False},
    "looking_right": {"scale_w": 1.0, "scale_h": 0.9, "top_lid": 0.25, "bottom_lid": 0.0, "lid_angle": -8.0, "mirror_angle": False},
    "calm": {"scale_w": 1.0, "scale_h": 0.85, "top_lid": 0.15, "bottom_lid": 0.1, "lid_angle": 0.0, "mirror_angle": True},
    "curious": {"scale_w": 0.95, "scale_h": 1.1, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": -5.0, "mirror_angle": True},
    "afraid": {"scale_w": 1.05, "scale_h": 1.2, "top_lid": 0.0, "bottom_lid": 0.1, "lid_angle": 8.0, "mirror_angle": True},
    "squint": {"scale_w": 1.0, "scale_h": 0.74, "top_lid": 0.24, "bottom_lid": 0.20, "lid_angle": 0.0, "mirror_angle": True},
}

# MJPEG Stream Config
STREAM_ENABLED = True
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8080
STREAM_FPS = 8
STREAM_JPEG_QUALITY = 70
RENDER_FPS = 24
VISION_FPS = 10

# --- Intensity Drift Config ---
INTENSITY_MIN = 0.55
INTENSITY_MAX = 0.95
INTENSITY_DRIFT_INTERVAL_SEC = 0.8
INTENSITY_EASING_SPEED = 0.12
INTENSITY_LOG_INTERVAL_SEC = 3.0


# --- BlockyEye Class ---
class BlockyEye:
    def __init__(self, x, y, scale=1.0, is_left=True):
        self.base_x, self.base_y = x, y
        self.current_pos = [float(x), float(y)]
        self.target_pos = [float(x), float(y)]
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.base_w = EYE_SIZE * scale
        self.base_h = EYE_SIZE * scale
        self.current_w = self.base_w
        self.current_h = self.base_h
        self.target_w = self.base_w
        self.target_h = self.base_h
        self.vel_w = 0.0
        self.vel_h = 0.0
        self.w = self.base_w
        self.h = self.base_h
        self.current_rotation = 0.0
        self.target_rotation = 0.0
        self.is_left = is_left
        self.blink_state = "IDLE"
        self.vy = 0
        self.blink_speed_mult = 1.0
        self.target_scale_w = 1.0
        self.target_scale_h = 1.0
        self.scale_w = 1.0
        self.scale_h = 1.0
        self.scale_w_vel = 0.0
        self.scale_h_vel = 0.0
        self.top_lid = 0.0
        self.bottom_lid = 0.0
        self.lid_angle = 0.0
        self.top_lid_vel = 0.0
        self.bottom_lid_vel = 0.0
        self.lid_angle_vel = 0.0
        self.target_top_lid = 0.0
        self.target_bottom_lid = 0.0
        self.target_lid_angle = 0.0
        self.current_emotion = "idle"
        self.happy_phase = random.uniform(0.0, math.pi * 2)
        self.happy_burst_until = 0.0
        self.noise_t = random.uniform(0, 100)

    def start_blink(self, speed_mult=None):
        if self.blink_state == "IDLE":
            self.blink_state = "DROPPING"
            if speed_mult is not None:
                self.blink_speed_mult = speed_mult
            else:
                self.blink_speed_mult = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
            self.vy = 40 * self.blink_speed_mult

    def set_emotion(self, emotion_name: str, intensity: float = 1.0):
        if emotion_name not in EMOTION_PRESETS:
            return
        if emotion_name == "happy" and self.current_emotion != "happy":
            self.happy_burst_until = time.time() + 0.35
        self.current_emotion = emotion_name
        preset = EMOTION_PRESETS[emotion_name]
        idle = EMOTION_PRESETS["idle"]
        intensity = max(0.0, min(1.0, intensity))
        scale_w = idle["scale_w"] + (preset["scale_w"] - idle["scale_w"]) * intensity
        scale_h = idle["scale_h"] + (preset["scale_h"] - idle["scale_h"]) * intensity
        top_lid = idle["top_lid"] + (preset["top_lid"] - idle["top_lid"]) * intensity
        bottom_lid = idle["bottom_lid"] + (preset["bottom_lid"] - idle["bottom_lid"]) * intensity
        lid_angle = idle["lid_angle"] + (preset["lid_angle"] - idle["lid_angle"]) * intensity
        self.target_scale_w = scale_w
        self.target_scale_h = scale_h
        self.target_top_lid = top_lid
        self.target_bottom_lid = bottom_lid
        if preset.get("mirror_angle", True) and not self.is_left and abs(lid_angle) > 0:
            lid_angle = -lid_angle
        self.target_lid_angle = lid_angle

    def update(self):
        if self.blink_state == "IDLE":
            t = time.time() + self.noise_t
            noise_x = (math.sin(t * 1.3) * 0.2 + math.sin(t * 0.7) * 0.1)
            noise_y = (math.cos(t * 1.1) * 0.2 + math.cos(t * 0.9) * 0.1)
            target_x_phys = self.target_pos[0] + noise_x
            target_y_phys = self.target_pos[1] + noise_y
            burst_active = time.time() < self.happy_burst_until
            if burst_active:
                target_y_phys -= 8.0
            if self.current_emotion == "happy":
                ht = time.time() * 6.0 + self.happy_phase
                target_y_phys -= 2.5 + math.sin(ht) * 2.0
                target_x_phys += math.sin(ht * 1.7) * 1.2
            elif self.current_emotion == "looking_left":
                target_x_phys -= LOOK_SIDE_OFFSET
            elif self.current_emotion == "looking_right":
                target_x_phys += LOOK_SIDE_OFFSET
            dx = target_x_phys - self.current_pos[0]
            dy = target_y_phys - self.current_pos[1]
            speed_x = 0.20
            speed_y = 0.22
            if dy < -1.0:
                speed_y = 0.14
            elif dy > 1.0:
                speed_y = 0.38
            self.current_pos[0] += dx * speed_x
            self.current_pos[1] += dy * speed_y
            self.vel_x = dx * speed_x
            self.vel_y = dy * speed_y
            final_target_rot = 0.0
            self.current_rotation += (final_target_rot - self.current_rotation) * 0.2
            t = time.time()
            breath_w = (math.sin(t * 1.5 + self.base_x) * 1.5 + math.sin(t * 0.5) * 1.0)
            breath_h = (math.cos(t * 1.8 + self.base_y) * 1.5 + math.cos(t * 0.6) * 1.0)
            move_stretch_x = (dx * speed_x) * 2.5
            move_stretch_y = (dy * speed_y) * 2.5
            if self.current_emotion == "surprised":
                move_stretch_x *= 0.25
                move_stretch_y *= 0.25
            k = 0.12
            d = 0.7
            if self.current_emotion == "excited":
                # Make excited expression settle faster with less overshoot.
                k = 0.22
                d = 0.62
            self.scale_w_vel = (self.scale_w_vel + (self.target_scale_w - self.scale_w) * k) * d
            self.scale_h_vel = (self.scale_h_vel + (self.target_scale_h - self.scale_h) * k) * d
            self.scale_w += self.scale_w_vel
            self.scale_h += self.scale_h_vel
            self.top_lid_vel = (self.top_lid_vel + (self.target_top_lid - self.top_lid) * k) * d
            self.bottom_lid_vel = (self.bottom_lid_vel + (self.target_bottom_lid - self.bottom_lid) * k) * d
            self.lid_angle_vel = (self.lid_angle_vel + (self.target_lid_angle - self.lid_angle) * k) * d
            self.top_lid += self.top_lid_vel
            self.bottom_lid += self.bottom_lid_vel
            self.lid_angle += self.lid_angle_vel
            self.target_w = (self.base_w * self.scale_w) + breath_w + (move_stretch_x * 0.5)
            self.target_h = (self.base_h * self.scale_h) + breath_h - (move_stretch_y * 0.2)
            if self.current_pos[1] > SCREEN_HEIGHT * 0.5:
                self.current_pos[1] = SCREEN_HEIGHT * 0.5
            else:
                self.current_pos[1] = max(EYE_BOUND_MARGIN, min(SCREEN_HEIGHT - EYE_BOUND_MARGIN, self.current_pos[1]))
        elif self.blink_state == "DROPPING":
            self.vy += 10 * self.blink_speed_mult
            self.current_pos[1] += self.vy
            self.current_w = self.base_w - 10
            self.current_h = self.base_h + 20
            self.target_w = self.current_w
            self.target_h = self.current_h
            if self.current_pos[1] + self.current_h // 2 >= FLOOR_Y:
                self.current_pos[1] = FLOOR_Y - self.current_h // 2
                self.blink_state = "SQUASHING"
        elif self.blink_state == "SQUASHING":
            squeeze_speed = 65 * self.blink_speed_mult
            spread_speed = 40 * self.blink_speed_mult
            self.current_h -= squeeze_speed
            self.current_w += spread_speed
            self.current_pos[1] = FLOOR_Y - self.current_h // 2
            if self.current_h <= 22:
                self.current_h = 22
                self.blink_state = "JUMPING"
        elif self.blink_state == "JUMPING":
            recovery_speed = max(0.15, min(0.95, 0.85 * self.blink_speed_mult))
            self.current_h += (self.base_h - self.current_h) * recovery_speed
            self.current_w += (self.base_w - self.current_w) * recovery_speed
            self.vel_x = (self.vel_x + (self.target_pos[0] - self.current_pos[0]) * 0.1) * 0.8
            self.current_pos[0] += self.vel_x
            target_y = self.target_pos[1]
            self.current_pos[1] += (target_y - self.current_pos[1]) * 0.8
            if abs(self.current_h - self.base_h) < 5 and abs(self.current_pos[1] - target_y) < 5:
                self.current_h = self.base_h
                self.current_w = self.base_w
                self.blink_state = "IDLE"
                self.vy = 0
                self.vel_x = 0
                self.vel_y = 0
        if self.blink_state == "IDLE":
            k = 0.08
            d = 0.90
            force_w = (self.target_w - self.current_w) * k
            self.vel_w = (self.vel_w + force_w) * d
            self.current_w += self.vel_w
            force_h = (self.target_h - self.current_h) * k
            self.vel_h = (self.vel_h + force_h) * d
            self.current_h += self.vel_h
        else:
            self.vel_w = 0
            self.vel_h = 0
        self.w = self.current_w
        self.h = self.current_h

    def draw_radial_rect(self, draw, x, y, w, h, color, radius, pupil_offset=(0,0)):
        center_x = x + w/2
        center_y = y + h/2
        shift_x = pupil_offset[0] * 15
        shift_y = pupil_offset[1] * 15
        cx = center_x + shift_x
        cy = center_y + shift_y
        x0 = cx - w / 2
        y0 = cy - h / 2
        x1 = cx + w / 2
        y1 = cy + h / 2
        base_radius = int(radius)
        cur_radius = min(base_radius, int(min(w, h) / 2))
        draw.rounded_rectangle([x0, y0, x1, y1], radius=cur_radius, fill=color)

    def draw_eyelids(self, eye_img, rect):
        x0, y0, x1, y1 = rect
        w = int(x1 - x0)
        h = int(y1 - y0)
        lid_color = BG_COLOR
        if self.top_lid > 0.01:
            lid_h = int(h * self.top_lid)
            lid_width = int(w * 2.0)
            lid_height = int(lid_h + max(8, h * 0.22))
            lid_src = Image.new("RGBA", (lid_width, lid_height), (*lid_color, 255))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y0 - max(4, h * 0.10))
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))
        if self.bottom_lid > 0.01:
            lid_h = int(h * self.bottom_lid)
            lid_width = int(w * 2.0)
            lid_height = int(lid_h + max(6, h * 0.16))
            lid_src = Image.new("RGBA", (lid_width, lid_height), (*lid_color, 255))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y1 + max(2, h * 0.05) - lid_src.height)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

    def draw(self, bg_image):
        draw_w = max(4, int(self.w))
        draw_h = max(4, int(self.h))
        eye_img_size = int(max(self.base_w, self.base_h) * 2.5)
        eye_img = Image.new("RGBA", (eye_img_size, eye_img_size), (0, 0, 0, 0))
        eye_draw = ImageDraw.Draw(eye_img)
        base_radius = int(min(self.base_w, self.base_h) * 0.25)
        corner_radius = min(base_radius, int(min(draw_w, draw_h) / 2))
        off_x = max(-1, min(1, (self.current_pos[0] - self.base_x) / 30.0))
        off_y = max(-1, min(1, (self.current_pos[1] - self.base_y) / 20.0))
        cx, cy = eye_img_size / 2, eye_img_size / 2
        x0 = cx - draw_w / 2
        y0 = cy - draw_h / 2
        x1 = cx + draw_w / 2
        y1 = cy + draw_h / 2
        self.draw_radial_rect(eye_draw, x0, y0, draw_w, draw_h, EYE_COLOR, corner_radius, (off_x, off_y))
        self.draw_eyelids(eye_img, (x0, y0, x1, y1))
        rotated = eye_img.rotate(self.current_rotation, resample=Image.BICUBIC, expand=False)
        paste_x = int(self.current_pos[0] - eye_img_size / 2)
        paste_y = int(self.current_pos[1] - eye_img_size / 2)
        
        # Clamp paste position to keep eye within display bounds
        paste_x = max(-(eye_img_size // 2), min(SCREEN_WIDTH - (eye_img_size // 2), paste_x))
        paste_y = max(-(eye_img_size // 2), min(SCREEN_HEIGHT - (eye_img_size // 2), paste_y))
        
        bg_image.alpha_composite(rotated, (paste_x, paste_y))


# --- MJPEG Streaming Server ---
latest_frame = None
frame_lock = threading.Lock()
stream_server = None

class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/stream"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with frame_lock:
                    frame = None if latest_frame is None else latest_frame.copy()
                if frame is None:
                    time.sleep(0.05)
                    continue
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=STREAM_JPEG_QUALITY)
                jpg = buf.getvalue()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / max(1, STREAM_FPS))
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format, *args):
        return

def start_stream_server():
    global stream_server
    stream_server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), MJPEGHandler)
    thread = threading.Thread(target=stream_server.serve_forever, daemon=True)
    thread.start()
    print(f"MJPEG stream started: http://{STREAM_HOST}:{STREAM_PORT}/stream")


# --- Display Setup ---
print("Initializing Displays (Dual SPI)...")
disp_l = None
disp_r = None

try:
    spi0 = board.SPI()
    disp_l = st7735.ST7735R(
        spi0, rotation=0, baudrate=24000000, bgr=True,
        cs=digitalio.DigitalInOut(board.CE1),
        dc=digitalio.DigitalInOut(board.D24),
        rst=digitalio.DigitalInOut(board.D25)
    )
except Exception as e:
    print(f"Error init Left Display (SPI0): {e}")

try:
    spi1 = busio.SPI(clock=board.D21, MOSI=board.D20, MISO=board.D19)
    disp_r = st7735.ST7735R(
        spi1, rotation=0, baudrate=24000000, bgr=True,
        cs=digitalio.DigitalInOut(board.D18),
        dc=digitalio.DigitalInOut(board.D23),
        rst=digitalio.DigitalInOut(board.D27)
    )
except Exception as e:
    print(f"Error init Right Display (SPI1): {e}")


# --- Servo Setup ---
print("Initializing Servos...")
pan_center = (PAN_MIN + PAN_MAX) / 2
tilt_center = (TILT_MIN + TILT_MAX) / 2
kit = None
try:
    kit = ServoKit(channels=16)
    kit.servo[PAN_CH].set_pulse_width_range(450, 2600)
    kit.servo[TILT_CH].set_pulse_width_range(450, 2600)
    kit.servo[PAN_CH].angle = pan_center
    kit.servo[TILT_CH].angle = tilt_center
    print("Servos initialized.")
except Exception as e:
    print(f"Error init servos: {e}")
    kit = None


# --- Camera & Face Detector Setup ---
print("Initializing Picamera2...")
picam2 = None
try:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": 'RGB888', "size": CAMERA_MAIN_RES},
        raw={"size": (3280, 2464)}
    )
    picam2.configure(config)
    picam2.set_controls({"ScalerCrop": (0, 0, 3280, 2464)})
    picam2.start()
    print(f"Camera started: Full sensor -> Main {CAMERA_MAIN_RES}, detect {CAMERA_RES}")
except Exception as e:
    print(f"Error starting Picamera2: {e}")
    sys.exit(1)

print("Initializing YuNet Face Detector...")
try:
    if not Path(FACE_MODEL_PATH).exists():
        print(f"Error: Face model not found at {FACE_MODEL_PATH}")
        sys.exit(1)
    detector = cv2.FaceDetectorYN.create(
        model=FACE_MODEL_PATH, config="", input_size=CAMERA_RES,
        score_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD,
        top_k=5000, backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU
    )
    print("YuNet initialized.")
except Exception as e:
    print(f"Error initializing detector: {e}")
    sys.exit(1)


# --- MJPEG Stream ---
if STREAM_ENABLED:
    try:
        start_stream_server()
    except Exception as e:
        print(f"Error starting MJPEG stream: {e}")


# --- Eye Objects ---
center_x = SCREEN_WIDTH / 2
center_y = SCREEN_HEIGHT / 2
left_eye = BlockyEye(center_x, center_y, scale=1.0, is_left=True)
right_eye = BlockyEye(center_x, center_y, scale=1.0, is_left=False)
right_eye.noise_t = left_eye.noise_t
left_eye.set_emotion("idle", 0.45)
right_eye.set_emotion("idle", 0.45)


# --- Animation Loop Vars ---
running = True
next_blink_time = time.time() + random.uniform(3, 6)
smoothed_x_off = 0.0
smoothed_y_off = 0.0
current_emotion = "idle"

# --- Intensity Drift State ---
emotion_intensity_current = 0.75
emotion_intensity_target = 0.75
next_intensity_shift_time = time.time() + random.uniform(0.5, 1.5)
last_intensity_log_time = 0.0


def log_emotion_state(emotion_name):
    stamp = time.strftime("%H:%M:%S")
    print(f"[EMOTION {stamp}] {emotion_name}")

target_lock = threading.Lock()
target_x_off = 0.0
target_y_off = 0.0
target_squint = 0.0
target_is_close = False
squint_until = 0.0
next_squint_allowed = 0.0

# Servo tracking vars
pan_target = pan_center
tilt_target = tilt_center
pan_current = pan_target
tilt_current = tilt_target


def clamp_eye_target(eye):
    half_w = max(12.0, eye.base_w * 0.42)
    half_h = max(12.0, eye.base_h * 0.42)
    min_x = half_w + EYE_BOUND_MARGIN
    max_x = SCREEN_WIDTH - half_w - EYE_BOUND_MARGIN
    min_y = half_h + EYE_BOUND_MARGIN
    max_y = SCREEN_HEIGHT - half_h - EYE_BOUND_MARGIN
    eye.target_pos[0] = max(min_x, min(max_x, eye.target_pos[0]))
    eye.target_pos[1] = max(min_y, min(max_y, eye.target_pos[1]))
    # Also clamp current position to prevent drift beyond bounds
    eye.current_pos[0] = max(min_x, min(max_x, eye.current_pos[0]))
    eye.current_pos[1] = max(min_y, min(max_y, eye.current_pos[1]))


def trigger_synced_blink(speed_mult):
    avg_y = (left_eye.current_pos[1] + right_eye.current_pos[1]) * 0.5
    avg_w = (left_eye.current_w + right_eye.current_w) * 0.5
    avg_h = (left_eye.current_h + right_eye.current_h) * 0.5
    for eye in (left_eye, right_eye):
        eye.blink_state = "IDLE"
        eye.vy = 0
        eye.current_pos[1] = avg_y
        eye.current_w = avg_w
        eye.current_h = avg_h
        eye.w = avg_w
        eye.h = avg_h
    left_eye.start_blink(speed_mult)
    right_eye.start_blink(speed_mult)


def mirror_full_state(master, slave):
    slave.blink_state = master.blink_state
    slave.vy = master.vy
    slave.current_pos[0] = master.current_pos[0]
    slave.current_pos[1] = master.current_pos[1]
    slave.target_pos[0] = master.target_pos[0]
    slave.target_pos[1] = master.target_pos[1]
    slave.current_w = master.current_w
    slave.current_h = master.current_h
    slave.target_w = master.target_w
    slave.target_h = master.target_h
    slave.current_rotation = master.current_rotation
    slave.target_rotation = master.target_rotation
    slave.scale_w = master.scale_w
    slave.scale_h = master.scale_h
    slave.target_scale_w = master.target_scale_w
    slave.target_scale_h = master.target_scale_h
    slave.top_lid = master.top_lid
    slave.bottom_lid = master.bottom_lid
    slave.lid_angle = master.lid_angle
    slave.target_top_lid = master.target_top_lid
    slave.target_bottom_lid = master.target_bottom_lid
    slave.target_lid_angle = master.target_lid_angle
    slave.w = master.w
    slave.h = master.h


def vision_worker():
    global running, target_x_off, target_y_off, target_squint, target_is_close, squint_until, next_squint_allowed, latest_frame, pan_target, tilt_target
    interval = 1.0 / max(1.0, float(VISION_FPS))
    next_tick = time.perf_counter()
    
    while running:
        try:
            large_frame = picam2.capture_array()
            frame = cv2.resize(large_frame, CAMERA_RES)
            if CAMERA_ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            local_x = 0.0
            local_y = 0.0
            local_squint = 0.0
            is_close = False
            
            if frame is not None and frame.size > 0:
                stream_frame = None
                if STREAM_ENABLED:
                    stream_frame = cv2.resize(frame, STREAM_RES)
                    if STREAM_SWAP_RB:
                        stream_frame = cv2.cvtColor(stream_frame, cv2.COLOR_BGR2RGB)
                    scale_x = STREAM_RES[0] / CAMERA_RES[0]
                    scale_y = STREAM_RES[1] / CAMERA_RES[1]
                
                detector.setInputSize((frame.shape[1], frame.shape[0]))
                faces = detector.detect(frame)
                
                if faces[1] is not None:
                    detected_faces = faces[1]
                    largest_face = max(detected_faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = largest_face[0:4]
                    
                    if STREAM_ENABLED and stream_frame is not None:
                        fx_s, fy_s = int(fx * scale_x), int(fy * scale_y)
                        fw_s, fh_s = int(fw * scale_x), int(fh * scale_y)
                        cv2.rectangle(stream_frame, (fx_s, fy_s), (fx_s + fw_s, fy_s + fh_s), (0, 255, 0), 2)
                    
                    face_cx = (fx + fw / 2) / CAMERA_RES[0]
                    face_cy = (fy + fh / 2) / CAMERA_RES[1]
                    norm_x = -((face_cx - 0.5) * 2.0)
                    norm_y = (face_cy - 0.5) * 2.0
                    
                    local_x = max(-MAX_X_OFFSET, min(MAX_X_OFFSET, norm_x * MAX_X_OFFSET))
                    local_y = max(-MAX_Y_OFFSET, min(MAX_Y_OFFSET, norm_y * MAX_Y_OFFSET))
                    
                    # Calculate servo angles from face position
                    if kit:
                        pan_target = PAN_MIN + (PAN_MAX - PAN_MIN) * (0.5 + norm_x * 0.5)
                        tilt_target = TILT_MIN + (TILT_MAX - TILT_MIN) * (0.5 - norm_y * 0.5)
                        pan_target = max(PAN_MIN, min(PAN_MAX, pan_target))
                        tilt_target = max(TILT_MIN, min(TILT_MAX, tilt_target))
                    
                    face_area_ratio = (fw * fh) / float(CAMERA_RES[0] * CAMERA_RES[1])
                    now = time.time()
                    
                    if face_area_ratio < FAR_FACE_AREA_RATIO:
                        frame_prob = min(0.9, FAR_SQUINT_CHANCE_PER_SEC / max(1.0, float(VISION_FPS)))
                        if now >= next_squint_allowed and now >= squint_until and random.random() < frame_prob:
                            squint_until = now + random.uniform(FAR_SQUINT_MIN_SEC, FAR_SQUINT_MAX_SEC)
                            next_squint_allowed = squint_until + random.uniform(FAR_SQUINT_GAP_MIN_SEC, FAR_SQUINT_GAP_MAX_SEC)
                        if now < squint_until:
                            local_squint = 1.0
                    else:
                        squint_until = 0.0
                        next_squint_allowed = now + random.uniform(0.8, 1.6)
                    
                    is_close = face_area_ratio >= CLOSE_FACE_AREA_RATIO
                else:
                    squint_until = 0.0
                    next_squint_allowed = time.time() + random.uniform(0.8, 1.6)
                
                with target_lock:
                    target_x_off = local_x
                    target_y_off = local_y
                    target_squint = local_squint
                    target_is_close = is_close
                
                if STREAM_ENABLED and stream_frame is not None:
                    with frame_lock:
                        latest_frame = stream_frame
        
        except Exception as e:
            print(f"Capture/Detect Error: {e}")
        
        next_tick += interval
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_tick = time.perf_counter()


print("Starting Vision/Tracking Loop...")
log_emotion_state(current_emotion)
time.sleep(1.0)
vision_thread = threading.Thread(target=vision_worker, daemon=True)
vision_thread.start()

try:
    while running:
        loop_start = time.perf_counter()
        
        with target_lock:
            local_target_x = target_x_off
            local_target_y = target_y_off
            local_target_squint = target_squint
            local_target_is_close = target_is_close
        
        # Smooth eye tracking
        smooth_alpha = 0.15
        smoothed_x_off = smoothed_x_off + (local_target_x - smoothed_x_off) * smooth_alpha
        smoothed_y_off = smoothed_y_off + (local_target_y - smoothed_y_off) * smooth_alpha
        
        # Update Eye Targets
        left_eye.target_pos[0] = left_eye.base_x + smoothed_x_off
        left_eye.target_pos[1] = left_eye.base_y + smoothed_y_off
        clamp_eye_target(left_eye)
        right_eye.target_pos[0] = left_eye.target_pos[0]
        right_eye.target_pos[1] = left_eye.target_pos[1]
        
        # Intensity Drift Engine
        now = time.time()
        if now >= next_intensity_shift_time:
            emotion_intensity_target = random.uniform(INTENSITY_MIN, INTENSITY_MAX)
            next_intensity_shift_time = now + random.uniform(INTENSITY_DRIFT_INTERVAL_SEC * 0.6, INTENSITY_DRIFT_INTERVAL_SEC * 1.4)
        
        drift = emotion_intensity_target - emotion_intensity_current
        emotion_intensity_current += drift * INTENSITY_EASING_SPEED
        emotion_intensity_current = max(INTENSITY_MIN, min(INTENSITY_MAX, emotion_intensity_current))
        
        if now - last_intensity_log_time >= INTENSITY_LOG_INTERVAL_SEC:
            print(f"[INTENSITY {time.strftime('%H:%M:%S')}] Current: {emotion_intensity_current:.2f} Target: {emotion_intensity_target:.2f}")
            last_intensity_log_time = now
        
        # Emotion selection
        should_squint = local_target_squint > 0.5
        if should_squint:
            target_emotion = "squint"
        elif local_target_is_close:
            target_emotion = "excited"
        else:
            target_emotion = "idle"
        
        if target_emotion != current_emotion:
            left_eye.set_emotion(target_emotion, emotion_intensity_current)
            current_emotion = target_emotion
            log_emotion_state(current_emotion)
        else:
            # Reapply emotion with updated intensity to enable drift effect
            left_eye.set_emotion(target_emotion, emotion_intensity_current)
        
        # Blink Logic
        if time.time() > next_blink_time:
            blink_speed = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
            trigger_synced_blink(blink_speed)
            next_blink_time = time.time() + random.uniform(3.5, 7.0)
        
        # Physics Update
        left_eye.update()
        mirror_full_state(left_eye, right_eye)
        
        # Smooth servo motion
        if kit:
            pan_current += (pan_target - pan_current) * SERVO_SMOOTHING
            tilt_current += (tilt_target - tilt_current) * SERVO_SMOOTHING
            try:
                kit.servo[PAN_CH].angle = pan_current
                kit.servo[TILT_CH].angle = tilt_current
            except Exception as e:
                print(f"Servo write error: {e}")
        
        # Draw eyes
        shared_rgb = None
        if disp_l or disp_r:
            img = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), BG_COLOR)
            left_eye.draw(img)
            shared_rgb = img.convert("RGB")
        
        try:
            if disp_l and shared_rgb is not None:
                disp_l.image(shared_rgb)
            if disp_r and shared_rgb is not None:
                disp_r.image(shared_rgb)
        except Exception as e:
            print(f"Display update error: {e}")
        
        frame_budget = (1.0 / max(1.0, float(RENDER_FPS))) - (time.perf_counter() - loop_start)
        if frame_budget > 0:
            time.sleep(frame_budget)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    running = False
    if vision_thread.is_alive():
        vision_thread.join(timeout=1.0)
    
    try:
        if picam2:
            picam2.stop()
            picam2.close()
            print("Camera closed.")
    except Exception as e:
        print(e)
    
    if stream_server:
        try:
            stream_server.shutdown()
            stream_server.server_close()
            print("MJPEG stream stopped.")
        except Exception as e:
            print(e)
    
    if kit:
        try:
            kit.servo[PAN_CH].angle = None
            kit.servo[TILT_CH].angle = None
            print("Servos relaxed.")
        except Exception as e:
            print(e)
    
    black = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
    if disp_l: disp_l.image(black)
    if disp_r: disp_r.image(black)
    print("Displays cleared.")
