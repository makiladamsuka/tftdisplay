#!/usr/bin/env python3
"""
Face Tracking Eyes for Dual SPI Displays (Picamera2)
Combines face tracking (YuNet) with dual SPI display output (ST7735).
"""

import time
import math
import random
import sys
import io
import threading
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
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
# Use a larger 16:9 main stream for wider/detail-rich source frames (wider field of view)
CAMERA_MAIN_RES = (1920, 1080)
# Balanced 16:9 processing for detail + CPU headroom (better for emotion models)
CAMERA_RES = (1280, 720)
STREAM_RES = (320, 180)   # Downscaled for web preview (maintain 16:9, no lag)
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
# Camera adjustments
CAMERA_ROTATE_180 = True
# If stream colors look wrong, swap R/B for MJPEG output
STREAM_SWAP_RB = True

# Eye Interaction Config
MAX_X_OFFSET = 30
MAX_Y_OFFSET = 22
FACE_ROLL_MULT = 0.75
FACE_ROLL_MAX_DEG = 10.0
EYE_BOUND_MARGIN = 8

# Blink Speed (Higher = Faster)
BLINK_SPEED_MIN = 2.0
BLINK_SPEED_MAX = 3.5
LOOK_SIDE_OFFSET = 16.0

# Distance-based behavior
CLOSE_FACE_AREA_RATIO = 0.05  # Trigger joy/excited when user is close (>5% of frame)
FAR_FACE_AREA_RATIO = 0.018   # Trigger squint when user is far (<1.8% of frame)
FAR_SQUINT_CHANCE = 0.08
FAR_SQUINT_MIN_SEC = 0.22
FAR_SQUINT_MAX_SEC = 0.55

# Reactive emotion behavior (surroundings-driven)
NO_FACE_GRACE_SEC = 0.9
NO_PERSON_HOLD_MIN_SEC = 2.8
NO_PERSON_HOLD_MAX_SEC = 5.2
PERSON_HOLD_MIN_SEC = 1.1
PERSON_HOLD_MAX_SEC = 2.8
DIRECTION_TRIGGER_NORM_X = 0.22
DIRECTION_HOLD_MIN_SEC = 0.6
DIRECTION_HOLD_MAX_SEC = 1.2
DIRECTION_COOLDOWN_SEC = 0.9

# Distance hysteresis to avoid rapid near/mid/far bouncing.
NEAR_EXIT_RATIO = CLOSE_FACE_AREA_RATIO * 0.82
FAR_EXIT_RATIO = FAR_FACE_AREA_RATIO * 1.25

# --- Emotion Presets ---
EMOTION_PRESETS = {
    "idle": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    # Reimagined happy: wider, shorter, arched lower lid, subtle upward tilt
    "happy": {"scale_w": 1.15, "scale_h": 0.78, "top_lid": 0.0, "bottom_lid": 0.38, "lid_angle": -7.0, "mirror_angle": True},
    "sad": {"scale_w": 1.1, "scale_h": 1.1, "top_lid": 0.24, "bottom_lid": 0.0, "lid_angle": 11.0, "mirror_angle": True},
    "angry": {"scale_w": 1.0, "scale_h": 0.9, "top_lid": 0.22, "bottom_lid": 0.0, "lid_angle": -15.0, "mirror_angle": True},
    "surprised": {"scale_w": 0.98, "scale_h": 1.08, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "suspicious": {"scale_w": 1.1, "scale_h": 0.6, "top_lid": 0.4, "bottom_lid": 0.4, "lid_angle": 0.0, "mirror_angle": True},
    "sleepy": {"scale_w": 1.1, "scale_h": 1.0, "top_lid": 0.6, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "looking_left": {"scale_w": 1.15, "scale_h": 0.78, "top_lid": 0.0, "bottom_lid": 0.38, "lid_angle": -7.0, "mirror_angle": False},
    "looking_right": {"scale_w": 1.15, "scale_h": 0.78, "top_lid": 0.0, "bottom_lid": 0.38, "lid_angle": 7.0, "mirror_angle": False},
    "excited": {"scale_w": 1.2, "scale_h": 0.72, "top_lid": 0.0, "bottom_lid": 0.28, "lid_angle": 0.0, "mirror_angle": True},
    "calm": {"scale_w": 1.05, "scale_h": 0.95, "top_lid": 0.12, "bottom_lid": 0.08, "lid_angle": 0.0, "mirror_angle": True},
    "curious": {"scale_w": 1.03, "scale_h": 1.02, "top_lid": 0.0, "bottom_lid": 0.45, "lid_angle": 11.0, "mirror_angle": False},
    "afraid": {"scale_w": 0.9, "scale_h": 1.22, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "squint": {"scale_w": 1.0, "scale_h": 0.62, "top_lid": 0.42, "bottom_lid": 0.35, "lid_angle": 0.0, "mirror_angle": True},
}

SPECIAL_EMOTIONS = ["happy", "suspicious", "sleepy"]

KEY_TO_EMOTION = {
    "0": "idle",
    "1": "happy",
    "2": "sad",
    "3": "angry",
    "4": "surprised",
    "5": "suspicious",
    "6": "sleepy",
    "7": "looking_left",
    "8": "looking_right",
    "9": "excited",
    "a": "calm",
    "s": "curious",
    "d": "afraid",
}

# MJPEG Stream Config (for headless SSH viewing)
STREAM_ENABLED = True
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8080
STREAM_FPS = 8
STREAM_JPEG_QUALITY = 70
RENDER_FPS = 24
VISION_FPS = 10
EMOTION_INTENSITY = 1.0
EMOTION_LOG_TO_TERMINAL = True
EMOTION_CHANGE_COOLDOWN = 0.75
TERMINAL_CONTROL_ENABLED = True


# --- BlockyEye Class (PIL Version with emotion controls) ---
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
        self.rot_sensitivity = random.uniform(0.3, 0.5)
        self.rot_speed = random.uniform(0.15, 0.25)

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
        self.last_emotion_change_time = 0.0
        self.pending_emotion = None
        self.pending_intensity = 1.0
        self.pending_apply_time = 0.0
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

    def set_emotion(self, emotion_name: str, intensity: float = 1.0, force: bool = False):
        if emotion_name not in EMOTION_PRESETS:
            return

        now = time.time()
        if (
            emotion_name != self.current_emotion
            and not force
            and (now - self.last_emotion_change_time) < EMOTION_CHANGE_COOLDOWN
        ):
            # Keep only the latest requested emotion to avoid stale transitions.
            self.pending_emotion = emotion_name
            self.pending_intensity = intensity
            self.pending_apply_time = self.last_emotion_change_time + EMOTION_CHANGE_COOLDOWN
            return

        if emotion_name == "happy" and self.current_emotion != "happy":
            self.happy_burst_until = time.time() + 0.35

        if emotion_name != self.current_emotion:
            self.last_emotion_change_time = now
        self.pending_emotion = None
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
        if self.pending_emotion is not None and time.time() >= self.pending_apply_time:
            queued_emotion = self.pending_emotion
            queued_intensity = self.pending_intensity
            self.pending_emotion = None
            self.set_emotion(queued_emotion, queued_intensity, force=True)

        if self.blink_state == "IDLE":
            t = time.time() + self.noise_t
            noise_x = (math.sin(t * 1.3) * 0.2 + math.sin(t * 0.7) * 0.1)
            noise_y = (math.cos(t * 1.1) * 0.2 + math.cos(t * 0.9) * 0.1)

            target_x_phys = self.target_pos[0] + noise_x
            target_y_phys = self.target_pos[1] + noise_y

            # Use per-frame temporary eyelid targets so one-shot effects
            # (like happy burst) do not permanently override emotion targets.
            top_lid_target = self.target_top_lid
            bottom_lid_target = self.target_bottom_lid
            lid_angle_target = self.target_lid_angle

            burst_active = time.time() < self.happy_burst_until
            if burst_active:
                target_y_phys -= 8.0
                # Keep the happy burst as a quick upward motion only.
                # Avoid temporary lid squeeze to prevent an initial "squint" look.

            # Happy: small jump + wiggle
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

            rel_x = self.current_pos[0] - self.base_x
            rel_y = self.current_pos[1] - self.base_y
            look_rot = (rel_x * 0.5 + rel_y * 0.8) * self.rot_sensitivity
            if self.current_emotion == "happy":
                look_rot += math.sin(time.time() * 8.0 + self.happy_phase) * 1.2
            final_target_rot = look_rot + self.target_rotation
            self.current_rotation += (final_target_rot - self.current_rotation) * self.rot_speed

            t = time.time()
            breath_w = (math.sin(t * 1.5 + self.base_x) * 1.5 + math.sin(t * 0.5) * 1.0)
            breath_h = (math.cos(t * 1.8 + self.base_y) * 1.5 + math.cos(t * 0.6) * 1.0)

            move_stretch_x = (dx * speed_x) * 2.5
            move_stretch_y = (dy * speed_y) * 2.5
            if self.current_emotion == "surprised":
                # Keep surprised wide-open but avoid rubbery stretching.
                move_stretch_x *= 0.25
                move_stretch_y *= 0.25

            k = 0.12
            d = 0.7
            if self.current_emotion == "surprised":
                # Snap into surprised quickly.
                k = 0.30
                d = 0.52
            self.scale_w_vel = (self.scale_w_vel + (self.target_scale_w - self.scale_w) * k) * d
            self.scale_h_vel = (self.scale_h_vel + (self.target_scale_h - self.scale_h) * k) * d
            self.scale_w += self.scale_w_vel
            self.scale_h += self.scale_h_vel

            self.top_lid_vel = (self.top_lid_vel + (top_lid_target - self.top_lid) * k) * d
            self.bottom_lid_vel = (self.bottom_lid_vel + (bottom_lid_target - self.bottom_lid) * k) * d
            self.lid_angle_vel = (self.lid_angle_vel + (lid_angle_target - self.lid_angle) * k) * d

            self.top_lid += self.top_lid_vel
            self.bottom_lid += self.bottom_lid_vel
            self.lid_angle += self.lid_angle_vel

            self.target_w = (self.base_w * self.scale_w) + breath_w + (move_stretch_x * 0.5)
            self.target_h = (self.base_h * self.scale_h) + breath_h - (move_stretch_y * 0.2)

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
                self.velocity = [0.0, 0.0]

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

        # Keep the eye fully inside the display area.
        half_w = max(2.0, self.w * 0.5)
        half_h = max(2.0, self.h * 0.5)
        min_x = half_w
        max_x = SCREEN_WIDTH - half_w
        min_y = half_h
        max_y = SCREEN_HEIGHT - half_h

        if min_x > max_x:
            self.current_pos[0] = SCREEN_WIDTH * 0.5
        else:
            self.current_pos[0] = max(min_x, min(max_x, self.current_pos[0]))

        if min_y > max_y:
            self.current_pos[1] = SCREEN_HEIGHT * 0.5
        else:
            self.current_pos[1] = max(min_y, min(max_y, self.current_pos[1]))

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

        # Angle-aware padding keeps diagonal lids from exposing tiny bright slivers.
        angle_abs = abs(self.lid_angle)
        angle_pad = int(min(8.0, 2.0 + angle_abs * 0.18))
        top_bleed = 6 + angle_pad
        bottom_bleed = 6 + angle_pad

        def _crop_rotated_fringes(img, px):
            if px <= 0:
                return img
            if img.width <= px * 2 or img.height <= px * 2:
                return img
            return img.crop((px, px, img.width - px, img.height - px))

        if self.top_lid > 0.01:
            lid_h = max(1, int(h * self.top_lid))
            lid_width = int(w + (top_bleed * 2) + (angle_pad * 2))
            lid_height = int(lid_h + 14 + top_bleed + angle_pad)
            lid_src = Image.new("RGBA", (lid_width, lid_height), (*lid_color, 255))
            if angle_abs > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
                fringe_px = 1 + (1 if angle_abs > 10.0 else 0)
                lid_src = _crop_rotated_fringes(lid_src, fringe_px)
            lid_x = int(x0 + (w / 2) - (lid_src.width / 2))
            lid_y = int(y0 - top_bleed - (angle_pad // 2))
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

        if self.bottom_lid > 0.01:
            lid_h = max(1, int(h * self.bottom_lid))
            lid_width = int(w + (bottom_bleed * 2) + (angle_pad * 2))
            lid_height = int(lid_h + 12 + bottom_bleed + angle_pad)
            lid_src = Image.new("RGBA", (lid_width, lid_height), (*lid_color, 255))
            if angle_abs > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
                fringe_px = 1 + (1 if angle_abs > 10.0 else 0)
                lid_src = _crop_rotated_fringes(lid_src, fringe_px)
            lid_x = int(x0 + (w / 2) - (lid_src.width / 2))
            lid_y = int(y1 + bottom_bleed + (angle_pad // 2) - lid_src.height)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

        # Safety seal strips close residual anti-aliased seams without bulky lids.
        seal_draw = ImageDraw.Draw(eye_img)
        if self.top_lid > 0.01:
            top_seal_h = max(1, min(4, int(1 + (h * self.top_lid * 0.06) + (angle_pad * 0.25))))
            seal_draw.rectangle(
                [int(x0) - 1, int(y0), int(x1) + 1, int(y0) + top_seal_h],
                fill=(*lid_color, 255),
            )
        if self.bottom_lid > 0.01:
            bot_seal_h = max(1, min(4, int(1 + (h * self.bottom_lid * 0.06) + (angle_pad * 0.25))))
            seal_draw.rectangle(
                [int(x0) - 1, int(y1) - bot_seal_h, int(x1) + 1, int(y1)],
                fill=(*lid_color, 255),
            )

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

                # Encode to JPEG (frame is RGB)
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

# SPI 0 (Left Screen)
try:
    spi0 = board.SPI()
    disp_l = st7735.ST7735R(
        spi0, 
        rotation=0, 
        baudrate=24000000, 
        bgr=True,
        cs=digitalio.DigitalInOut(board.CE1),   
        dc=digitalio.DigitalInOut(board.D24),   
        rst=digitalio.DigitalInOut(board.D25)
    )
except Exception as e:
    print(f"Error init Left Display (SPI0): {e}")

# SPI 1 (Right Screen)
try:
    spi1 = busio.SPI(clock=board.D21, MOSI=board.D20, MISO=board.D19)
    disp_r = st7735.ST7735R(
        spi1, 
        rotation=0, 
        baudrate=24000000, 
        bgr=True,
        cs=digitalio.DigitalInOut(board.D18),   
        dc=digitalio.DigitalInOut(board.D23),   
        rst=digitalio.DigitalInOut(board.D27)
    )
except Exception as e:
    print(f"Error init Right Display (SPI1): {e}")


# --- Camera & Face Detector Setup ---
print("Initializing Picamera2...")
picam2 = None
try:
    picam2 = Picamera2()
    
    # Pi Camera v2: Full sensor resolution (3280x2464) for widest FOV
    # Using full sensor ensures no digital zoom, maximizing field of view
    config = picam2.create_video_configuration(
        main={"format": 'RGB888', "size": CAMERA_MAIN_RES},
        raw={"size": (3280, 2464)}
    )
    picam2.configure(config)
    # Force full sensor area (no cropping)
    picam2.set_controls({"ScalerCrop": (0, 0, 3280, 2464)})
    picam2.start()
    print(f"Camera started: Full sensor (3280x2464) -> Main ({CAMERA_MAIN_RES[0]}x{CAMERA_MAIN_RES[1]}), detect ({CAMERA_RES[0]}x{CAMERA_RES[1]})")
except Exception as e:
    print(f"Error starting Picamera2: {e}")
    sys.exit(1)

print("Initializing YuNet Face Detector...")
try:
    if not Path(FACE_MODEL_PATH).exists():
        print(f"Error: Face model not found at {FACE_MODEL_PATH}")
        sys.exit(1)
        
    detector = cv2.FaceDetectorYN.create(
        model=FACE_MODEL_PATH,
        config="",
        input_size=CAMERA_RES,
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
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
# Keep both eyes using identical dynamics to avoid drift during blink phases.
right_eye.noise_t = left_eye.noise_t
right_eye.rot_sensitivity = left_eye.rot_sensitivity
right_eye.rot_speed = left_eye.rot_speed
right_eye.happy_phase = left_eye.happy_phase
left_eye.set_emotion("idle", EMOTION_INTENSITY)
right_eye.set_emotion("idle", EMOTION_INTENSITY)

# Animation Loop Vars
running = True
next_blink_time = time.time() + random.uniform(3, 6)
last_blink_time = time.time()
smoothed_x_off = 0.0
smoothed_y_off = 0.0
smoothed_rotation = 0.0
current_emotion = "idle"  # Track current emotion to avoid redundant updates

target_lock = threading.Lock()
target_x_off = 0.0
target_y_off = 0.0
target_rotation = 0.0
target_squint = 0.0
target_is_close = False  # Track if user is close for emotion switching
target_face_detected = False
target_face_area_ratio = 0.0
target_face_norm_x = 0.0
squint_until = 0.0

last_seen_face_time = 0.0
distance_zone = "mid"
no_person_next_emotion = "sleepy"
next_emotion_change_time = time.time() + random.uniform(1.6, 3.0)
direction_cooldown_until = 0.0
emotion_history = []
EMOTION_HISTORY_LEN = 3
prev_target_x = 0.0
prev_target_y = 0.0
prev_target_rot = 0.0
manual_emotion_override = None
command_lock = threading.Lock()

def clamp_eye_target(eye):
    # Keep eye center inside the panel bounds even during shape changes.
    half_w = max(12.0, eye.base_w * 0.42)
    half_h = max(12.0, eye.base_h * 0.42)
    min_x = half_w + EYE_BOUND_MARGIN
    max_x = SCREEN_WIDTH - half_w - EYE_BOUND_MARGIN
    min_y = half_h + EYE_BOUND_MARGIN
    max_y = SCREEN_HEIGHT - half_h - EYE_BOUND_MARGIN
    eye.target_pos[0] = max(min_x, min(max_x, eye.target_pos[0]))
    eye.target_pos[1] = max(min_y, min(max_y, eye.target_pos[1]))


def trigger_synced_blink(speed_mult):
    # Align blink start conditions so both displays animate the same phase.
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


def mirror_blink_state(master, slave):
    # Force exact blink phase matching once a blink is active.
    slave.blink_state = master.blink_state
    slave.vy = master.vy
    slave.current_pos[1] = master.current_pos[1]
    slave.current_w = master.current_w
    slave.current_h = master.current_h
    slave.target_w = master.target_w
    slave.target_h = master.target_h
    slave.w = master.w
    slave.h = master.h


def mirror_full_state(master, slave):
    # Keep both eyes identical by driving one master state.
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


def push_emotion_history(emotion_name):
    emotion_history.append(emotion_name)
    if len(emotion_history) > EMOTION_HISTORY_LEN:
        emotion_history.pop(0)


def apply_emotion(emotion_name):
    global current_emotion
    if emotion_name != current_emotion:
        left_eye.set_emotion(emotion_name, EMOTION_INTENSITY)
        right_eye.set_emotion(emotion_name, EMOTION_INTENSITY)
        current_emotion = emotion_name
        push_emotion_history(emotion_name)
        if EMOTION_LOG_TO_TERMINAL:
            ts = time.strftime("%H:%M:%S")
            print(f"[emotion {ts}] {emotion_name}")


def terminal_command_worker():
    global running, manual_emotion_override, EMOTION_INTENSITY, next_emotion_change_time

    print("Terminal control ready. Commands: emotion <name>, <shortcut>, auto, list, intensity <0..1>, blink, status, help")
    print("Shortcuts: 0-9, a, s, d")
    while running:
        try:
            raw = input().strip()
        except EOFError:
            time.sleep(0.1)
            continue
        except Exception:
            time.sleep(0.1)
            continue

        if not raw:
            continue

        cmd = raw.lower()
        parts = cmd.split()

        if cmd in ("help", "h", "?"):
            print("Commands: emotion <name> | <name> | <shortcut> | auto | list | intensity <0..1> | blink | status")
            print("Shortcuts: 0-9, a, s, d")
            continue

        if cmd in ("list", "emotions"):
            print("Available emotions:", ", ".join(sorted(EMOTION_PRESETS.keys())))
            continue

        if cmd in ("auto", "clear", "reactive"):
            with command_lock:
                manual_emotion_override = None
            next_emotion_change_time = time.time() + random.uniform(0.2, 0.6)
            print("[mode] reactive auto mode enabled")
            continue

        if cmd == "blink":
            trigger_synced_blink(random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX))
            print("[action] blink")
            continue

        if parts and parts[0] == "intensity" and len(parts) >= 2:
            try:
                value = float(parts[1])
                EMOTION_INTENSITY = max(0.0, min(1.0, value))
                print(f"[emotion] intensity={EMOTION_INTENSITY:.2f}")
            except ValueError:
                print("[error] intensity must be a number between 0 and 1")
            continue

        if cmd == "status":
            with command_lock:
                manual = manual_emotion_override
            mode = "manual" if manual else "auto"
            print(f"[status] mode={mode}, current={current_emotion}, manual={manual}, intensity={EMOTION_INTENSITY:.2f}")
            continue

        selected = None
        if parts and parts[0] == "emotion" and len(parts) >= 2:
            selected = parts[1]
        elif cmd in KEY_TO_EMOTION:
            selected = KEY_TO_EMOTION[cmd]
        elif cmd in EMOTION_PRESETS:
            selected = cmd

        if selected is not None:
            if selected not in EMOTION_PRESETS:
                print(f"[error] unknown emotion: {selected}")
                continue
            with command_lock:
                manual_emotion_override = selected
            print(f"[mode] manual emotion={selected}")
            continue

        print(f"[error] unknown command: {raw}")


def classify_distance_zone(face_area_ratio, prev_zone):
    if prev_zone == "near" and face_area_ratio >= NEAR_EXIT_RATIO:
        return "near"
    if prev_zone == "far" and face_area_ratio <= FAR_EXIT_RATIO:
        return "far"
    if face_area_ratio >= CLOSE_FACE_AREA_RATIO:
        return "near"
    if face_area_ratio < FAR_FACE_AREA_RATIO:
        return "far"
    return "mid"


def weighted_pick(weights, fallback="idle"):
    total = 0.0
    cleaned = {}
    for name, w in weights.items():
        if w > 0.0:
            cleaned[name] = float(w)
            total += float(w)
    if total <= 0.0:
        return fallback
    r = random.uniform(0.0, total)
    acc = 0.0
    for name, w in cleaned.items():
        acc += w
        if r <= acc:
            return name
    return fallback


def choose_no_person_emotion():
    global no_person_next_emotion
    base = no_person_next_emotion
    no_person_next_emotion = "idle" if no_person_next_emotion == "sleepy" else "sleepy"

    # Keep no-person mode mostly boring, with rare subtle accents.
    r = random.random()
    if r < 0.08:
        return "sad"
    if r < 0.14:
        return "calm"
    return base


def choose_person_emotion(zone, activity, squint_hint):
    if zone == "near":
        weights = {
            "excited": 1.0,
            "happy": 0.55,
            "curious": 0.12,
            "calm": 0.12,
            "surprised": 0.09,
            "afraid": 0.06,
            "angry": 0.04,
        }
        if activity > 0.75:
            weights["surprised"] += 0.20
            weights["afraid"] += 0.08
        if activity < 0.25:
            weights["calm"] += 0.10
    elif zone == "far":
        weights = {
            "curious": 1.0,
            "happy": 0.25,
            "calm": 0.30,
            "squint": 0.22,
            "sad": 0.10,
            "sleepy": 0.08,
            "idle": 0.10,
        }
        if squint_hint > 0.5:
            weights["squint"] += 0.35
    else:
        weights = {
            "happy": 1.0,
            "excited": 0.22,
            "curious": 0.30,
            "calm": 0.28,
            "suspicious": 0.15,
            "surprised": 0.08,
            "angry": 0.04,
            "afraid": 0.03,
        }
        if activity > 0.8:
            weights["surprised"] += 0.20
        if activity < 0.2:
            weights["calm"] += 0.12

    # Anti-repeat so expressions feel less robotic.
    for recent in emotion_history[-EMOTION_HISTORY_LEN:]:
        if recent in weights:
            weights[recent] *= 0.35
    if current_emotion in weights:
        weights[current_emotion] *= 0.45

    return weighted_pick(weights, fallback="happy")


def vision_worker():
    global running, target_x_off, target_y_off, target_rotation, target_squint, target_is_close
    global target_face_detected, target_face_area_ratio, target_face_norm_x, squint_until, latest_frame

    interval = 1.0 / max(1.0, float(VISION_FPS))
    next_tick = time.perf_counter()

    while running:
        try:
            # Capture full frame and resize once for detector input
            large_frame = picam2.capture_array()
            frame = cv2.resize(large_frame, CAMERA_RES)

            if CAMERA_ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            local_x = 0.0
            local_y = 0.0
            local_rot = 0.0
            local_squint = 0.0
            local_face_detected = False
            local_face_area_ratio = 0.0
            local_face_norm_x = 0.0
            is_close = False  # Initialize before any conditional use

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
                    re_x, re_y = largest_face[4], largest_face[5]
                    le_x, le_y = largest_face[6], largest_face[7]

                    if STREAM_ENABLED and stream_frame is not None:
                        fx_s, fy_s = int(fx * scale_x), int(fy * scale_y)
                        fw_s, fh_s = int(fw * scale_x), int(fh * scale_y)
                        re_x_s, re_y_s = int(re_x * scale_x), int(re_y * scale_y)
                        le_x_s, le_y_s = int(le_x * scale_x), int(le_y * scale_y)
                        cv2.rectangle(stream_frame, (fx_s, fy_s), (fx_s + fw_s, fy_s + fh_s), (0, 255, 0), 2)
                        cv2.circle(stream_frame, (re_x_s, re_y_s), 5, (255, 0, 0), -1)
                        cv2.circle(stream_frame, (le_x_s, le_y_s), 5, (255, 0, 0), -1)

                    face_cx = (fx + fw / 2) / CAMERA_RES[0]
                    face_cy = (fy + fh / 2) / CAMERA_RES[1]
                    norm_x = -((face_cx - 0.5) * 2.0)
                    norm_y = (face_cy - 0.5) * 2.0
                    local_face_detected = True
                    local_face_norm_x = norm_x

                    local_x = max(-MAX_X_OFFSET, min(MAX_X_OFFSET, norm_x * MAX_X_OFFSET))
                    local_y = max(-MAX_Y_OFFSET, min(MAX_Y_OFFSET, norm_y * MAX_Y_OFFSET))

                    # Distance-based emotion: squint when far, excited when close
                    face_area_ratio = (fw * fh) / float(CAMERA_RES[0] * CAMERA_RES[1])
                    local_face_area_ratio = face_area_ratio
                    now = time.time()
                    
                    # Check for far-distance squinting
                    if face_area_ratio < FAR_FACE_AREA_RATIO:
                        if now > squint_until and random.random() < FAR_SQUINT_CHANCE:
                            squint_until = now + random.uniform(FAR_SQUINT_MIN_SEC, FAR_SQUINT_MAX_SEC)
                        if now < squint_until:
                            local_squint = 1.0
                    else:
                        squint_until = 0.0
                    
                    # Track if user is close for emotion switching
                    is_close = face_area_ratio >= CLOSE_FACE_AREA_RATIO

                    dx = re_x - le_x
                    dy = re_y - le_y
                    if dx != 0:
                        angle_rad = math.atan2(dy, dx)
                        angle_deg = math.degrees(angle_rad)
                        local_rot = max(-FACE_ROLL_MAX_DEG, min(FACE_ROLL_MAX_DEG, -angle_deg * FACE_ROLL_MULT))
                else:
                    squint_until = 0.0

                with target_lock:
                    target_x_off = local_x
                    target_y_off = local_y
                    target_rotation = local_rot
                    target_squint = local_squint
                    target_is_close = is_close
                    target_face_detected = local_face_detected
                    target_face_area_ratio = local_face_area_ratio
                    target_face_norm_x = local_face_norm_x

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

print("Starting Tracking Loop...")
time.sleep(1.0) # Warmup

vision_thread = threading.Thread(target=vision_worker, daemon=True)
vision_thread.start()

cmd_thread = None
if TERMINAL_CONTROL_ENABLED:
    cmd_thread = threading.Thread(target=terminal_command_worker, daemon=True)
    cmd_thread.start()

try:
    while running:
        loop_start = time.perf_counter()

        with target_lock:
            local_target_x = target_x_off
            local_target_y = target_y_off
            local_target_rot = target_rotation
            local_target_squint = target_squint
            local_target_is_close = target_is_close
            local_face_detected = target_face_detected
            local_face_area_ratio = target_face_area_ratio
            local_face_norm_x = target_face_norm_x

        # Smooth tracking to reduce jitter
        smooth_alpha = 0.15
        smoothed_x_off = smoothed_x_off + (local_target_x - smoothed_x_off) * smooth_alpha
        smoothed_y_off = smoothed_y_off + (local_target_y - smoothed_y_off) * smooth_alpha
        smoothed_rotation = smoothed_rotation + (local_target_rot - smoothed_rotation) * smooth_alpha
        
        # 2. Update Eye Targets
        left_eye.target_pos[0] = left_eye.base_x + smoothed_x_off
        left_eye.target_pos[1] = left_eye.base_y + smoothed_y_off
        clamp_eye_target(left_eye)

        right_eye.target_pos[0] = left_eye.target_pos[0]
        right_eye.target_pos[1] = left_eye.target_pos[1]
        left_eye.target_rotation = smoothed_rotation
        right_eye.target_rotation = smoothed_rotation

        now = time.time()
        if local_face_detected:
            last_seen_face_time = now
        person_present = (now - last_seen_face_time) <= NO_FACE_GRACE_SEC

        dx_activity = abs(local_target_x - prev_target_x) / max(1.0, float(MAX_X_OFFSET))
        dy_activity = abs(local_target_y - prev_target_y) / max(1.0, float(MAX_Y_OFFSET))
        dr_activity = abs(local_target_rot - prev_target_rot) / max(1.0, float(FACE_ROLL_MAX_DEG))
        activity = min(1.0, (dx_activity + dy_activity + dr_activity) / 3.0)
        prev_target_x = local_target_x
        prev_target_y = local_target_y
        prev_target_rot = local_target_rot

        with command_lock:
            manual_emotion = manual_emotion_override

        if manual_emotion is not None:
            apply_emotion(manual_emotion)
        elif now >= next_emotion_change_time:
            if not person_present:
                next_emotion = choose_no_person_emotion()
                hold_sec = random.uniform(NO_PERSON_HOLD_MIN_SEC, NO_PERSON_HOLD_MAX_SEC)
            else:
                distance_zone = classify_distance_zone(local_face_area_ratio, distance_zone)
                can_directional = now >= direction_cooldown_until and abs(local_face_norm_x) >= DIRECTION_TRIGGER_NORM_X
                if can_directional and random.random() < (0.5 if distance_zone == "mid" else 0.35):
                    next_emotion = "looking_right" if local_face_norm_x > 0 else "looking_left"
                    hold_sec = random.uniform(DIRECTION_HOLD_MIN_SEC, DIRECTION_HOLD_MAX_SEC)
                    direction_cooldown_until = now + hold_sec + DIRECTION_COOLDOWN_SEC
                else:
                    next_emotion = choose_person_emotion(distance_zone, activity, local_target_squint)
                    hold_base = random.uniform(PERSON_HOLD_MIN_SEC, PERSON_HOLD_MAX_SEC)
                    hold_sec = max(PERSON_HOLD_MIN_SEC, hold_base * (1.12 - 0.42 * activity))

            apply_emotion(next_emotion)
            next_emotion_change_time = now + hold_sec
        
        # 3. Blink Logic
        if time.time() > next_blink_time:
            blink_speed = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
            trigger_synced_blink(blink_speed)
            last_blink_time = time.time()
            next_blink_time = time.time() + random.uniform(3.5, 7.0)

        # Keep idle motion deterministic to avoid perceived micro-jitter.
        
        # 5. Physics Update
        left_eye.update()
        right_eye.update()
        
        # 6. Draw
        rgb_l = None
        rgb_r = None
        if disp_l:
            img_l = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), BG_COLOR)
            left_eye.draw(img_l)
            rgb_l = img_l.convert("RGB")
        if disp_r:
            img_r = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), BG_COLOR)
            right_eye.draw(img_r)
            rgb_r = img_r.convert("RGB")

        try:
            if disp_l and rgb_l is not None:
                disp_l.image(rgb_l)
            if disp_r and rgb_r is not None:
                disp_r.image(rgb_r)
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

    # Cleanup attributes
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
        
    # Clear screens
    black = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
    if disp_l: disp_l.image(black)
    if disp_r: disp_r.image(black)
    print("Displays cleared.")