#!/usr/bin/env python3
"""
Emotion Tuner for SPI Eyes (round-eye version, no camera)
Use keyboard to switch emotions and tweak intensity.

Keys:
  0 = idle
  1 = happy
  2 = sad
  3 = angry
  4 = surprised
  5 = suspicious
  6 = sleepy
    7 = looking left (natural)
    8 = looking right (natural)
    f = looking left (happy)
    g = looking right (happy)
  9 = excited
  a = calm
  s = curious
  d = afraid
  [ / ] = decrease/increase intensity
  b = trigger blink
  q = quit
"""

import sys
import time
import math
import random
import select
import termios
import tty

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


# --- Configuration ---
SCREEN_WIDTH = 128
SCREEN_HEIGHT = 160
EYE_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)
EYE_SIZE = 126
FLOOR_Y = SCREEN_HEIGHT - 6

BLINK_SPEED_MIN = 2.0
BLINK_SPEED_MAX = 3.5
LOOK_SIDE_OFFSET = 16.0
EMOTION_CHANGE_COOLDOWN = 0.75

EMOTION_PRESETS = {
    "idle": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "happy": {"scale_w": 1.10, "scale_h": 0.84, "top_lid": 0.0, "bottom_lid": 0.30, "lid_angle": -6.0, "mirror_angle": True},
    "sad": {"scale_w": 0.98, "scale_h": 1.08, "top_lid": 0.20, "bottom_lid": 0.0, "lid_angle": 10.0, "mirror_angle": True},
    "angry": {"scale_w": 1.02, "scale_h": 0.90, "top_lid": 0.24, "bottom_lid": 0.0, "lid_angle": -14.0, "mirror_angle": True},
    "surprised": {"scale_w": 0.98, "scale_h": 1.12, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "suspicious": {"scale_w": 1.06, "scale_h": 0.74, "top_lid": 0.38, "bottom_lid": 0.35, "lid_angle": 0.0, "mirror_angle": True},
    "sleepy": {"scale_w": 1.04, "scale_h": 0.88, "top_lid": 0.56, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "looking_left_natural": {"scale_w": 1.02, "scale_h": 0.98, "top_lid": 0.0, "bottom_lid": 0.05, "lid_angle": -3.0, "mirror_angle": False},
    "looking_right_natural": {"scale_w": 1.02, "scale_h": 0.98, "top_lid": 0.0, "bottom_lid": 0.05, "lid_angle": 3.0, "mirror_angle": False},
    "looking_left_happy": {"scale_w": 1.10, "scale_h": 0.84, "top_lid": 0.0, "bottom_lid": 0.30, "lid_angle": -6.0, "mirror_angle": False},
    "looking_right_happy": {"scale_w": 1.10, "scale_h": 0.84, "top_lid": 0.0, "bottom_lid": 0.30, "lid_angle": 6.0, "mirror_angle": False},
    "excited": {"scale_w": 1.14, "scale_h": 0.80, "top_lid": 0.0, "bottom_lid": 0.24, "lid_angle": 0.0, "mirror_angle": True},
    "calm": {"scale_w": 1.02, "scale_h": 0.95, "top_lid": 0.10, "bottom_lid": 0.08, "lid_angle": 0.0, "mirror_angle": True},
    "curious": {"scale_w": 1.00, "scale_h": 1.04, "top_lid": 0.0, "bottom_lid": 0.38, "lid_angle": 10.0, "mirror_angle": False},
    "afraid": {"scale_w": 0.86, "scale_h": 1.24, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
}

KEY_TO_EMOTION = {
    "0": "idle",
    "1": "happy",
    "2": "sad",
    "3": "angry",
    "4": "surprised",
    "5": "suspicious",
    "6": "sleepy",
    "7": "looking_left_natural",
    "8": "looking_right_natural",
    "f": "looking_left_happy",
    "g": "looking_right_happy",
    "9": "excited",
    "a": "calm",
    "s": "curious",
    "d": "afraid",
}


class RoundEye:
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
        self.rot_sensitivity = random.uniform(0.24, 0.40)
        self.rot_speed = random.uniform(0.13, 0.22)

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
        self.surprise_shock_until = 0.0
        self.look_entry_until = 0.0
        self.noise_t = random.uniform(0.0, 100.0)

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
            self.pending_emotion = emotion_name
            self.pending_intensity = intensity
            self.pending_apply_time = self.last_emotion_change_time + EMOTION_CHANGE_COOLDOWN
            return

        if emotion_name == "happy" and self.current_emotion != "happy":
            self.happy_burst_until = time.time() + 0.35
        if emotion_name == "surprised" and self.current_emotion != "surprised":
            self.surprise_shock_until = time.time() + 0.18
        if emotion_name.startswith("looking_") and self.current_emotion != emotion_name:
            self.look_entry_until = time.time() + 0.16

        if emotion_name != self.current_emotion:
            self.last_emotion_change_time = now
        self.pending_emotion = None
        self.current_emotion = emotion_name
        preset = EMOTION_PRESETS[emotion_name]
        idle = EMOTION_PRESETS["idle"]

        intensity = max(0.0, min(1.0, intensity))
        self.target_scale_w = idle["scale_w"] + (preset["scale_w"] - idle["scale_w"]) * intensity
        self.target_scale_h = idle["scale_h"] + (preset["scale_h"] - idle["scale_h"]) * intensity
        self.target_top_lid = idle["top_lid"] + (preset["top_lid"] - idle["top_lid"]) * intensity
        self.target_bottom_lid = idle["bottom_lid"] + (preset["bottom_lid"] - idle["bottom_lid"]) * intensity

        lid_angle = idle["lid_angle"] + (preset["lid_angle"] - idle["lid_angle"]) * intensity
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
            now = time.time()
            t = now + self.noise_t
            noise_x = (math.sin(t * 1.3) * 0.2 + math.sin(t * 0.7) * 0.1)
            noise_y = (math.cos(t * 1.1) * 0.2 + math.cos(t * 0.9) * 0.1)

            target_x_phys = self.target_pos[0] + noise_x
            target_y_phys = self.target_pos[1] + noise_y

            top_lid_target = self.target_top_lid
            bottom_lid_target = self.target_bottom_lid
            lid_angle_target = self.target_lid_angle

            burst_active = time.time() < self.happy_burst_until
            if burst_active:
                target_y_phys -= 8.0

            if self.current_emotion == "happy":
                ht = time.time() * 6.0 + self.happy_phase
                target_y_phys -= 2.2 + math.sin(ht) * 1.8
                target_x_phys += math.sin(ht * 1.7) * 1.2
            elif self.current_emotion.startswith("looking_") and "left" in self.current_emotion:
                target_x_phys -= LOOK_SIDE_OFFSET
            elif self.current_emotion.startswith("looking_") and "right" in self.current_emotion:
                target_x_phys += LOOK_SIDE_OFFSET

            look_entry_active = self.current_emotion.startswith("looking_") and now < self.look_entry_until
            if look_entry_active:
                # Keep the side-look entry clean: fast horizontal move, minimal vertical wobble.
                side_sign = -1.0 if "left" in self.current_emotion else 1.0
                target_x_phys = self.base_x + side_sign * (LOOK_SIDE_OFFSET * 0.9)
                target_y_phys = self.base_y

            dx = target_x_phys - self.current_pos[0]
            dy = target_y_phys - self.current_pos[1]

            speed_x = 0.20
            speed_y = 0.22
            if dy < -1.0:
                speed_y = 0.14
            elif dy > 1.0:
                speed_y = 0.38
            if look_entry_active:
                speed_x = 0.42
                speed_y = 0.18

            self.current_pos[0] += dx * speed_x
            self.current_pos[1] += dy * speed_y

            self.vel_x = dx * speed_x
            self.vel_y = dy * speed_y

            rel_x = self.current_pos[0] - self.base_x
            rel_y = self.current_pos[1] - self.base_y
            look_rot = (rel_x * 0.45 + rel_y * 0.65) * self.rot_sensitivity
            if self.current_emotion == "happy":
                look_rot += math.sin(time.time() * 8.0 + self.happy_phase) * 1.1
            final_target_rot = look_rot + self.target_rotation
            self.current_rotation += (final_target_rot - self.current_rotation) * self.rot_speed

            t = time.time()
            breath_w = (math.sin(t * 1.5 + self.base_x) * 1.3 + math.sin(t * 0.5) * 0.9)
            breath_h = (math.cos(t * 1.8 + self.base_y) * 1.3 + math.cos(t * 0.6) * 0.9)

            move_stretch_x = (dx * speed_x) * 2.0
            move_stretch_y = (dy * speed_y) * 2.0
            if self.current_emotion == "surprised":
                # Keep surprise clean and snappy, without rubber-band deformation.
                move_stretch_x = 0.0
                move_stretch_y = 0.0
            elif self.current_emotion.startswith("looking_"):
                if look_entry_active:
                    move_stretch_x = 0.0
                    move_stretch_y = 0.0
                else:
                    move_stretch_x *= 0.45
                    move_stretch_y *= 0.45

            k = 0.12
            d = 0.70
            if self.current_emotion == "surprised":
                if time.time() < self.surprise_shock_until:
                    k = 0.46
                    d = 0.44
                else:
                    k = 0.20
                    d = 0.72

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

        elif self.blink_state == "SQUASHING":
            squeeze_speed = 45 * self.blink_speed_mult
            spread_speed = 30 * self.blink_speed_mult
            self.current_h -= squeeze_speed
            self.current_w += spread_speed
            self.current_pos[1] = FLOOR_Y - self.current_h // 2

            if self.current_h <= 25:
                self.current_h = 25
                self.blink_state = "JUMPING"

        elif self.blink_state == "JUMPING":
            recovery_speed = max(0.1, min(0.9, 0.7 * self.blink_speed_mult))
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

    def draw_radial_eye(self, draw, x, y, w, h, color, pupil_offset=(0.0, 0.0)):
        # Solid eye fill (no radial gradient)
        draw.ellipse([x, y, x + w, y + h], fill=color)

    def draw_eyelids(self, eye_img, rect):
        x0, y0, x1, y1 = rect
        w = int(x1 - x0)
        h = int(y1 - y0)
        lid_color = BG_COLOR

        if self.top_lid > 0.01:
            lid_h = int(h * self.top_lid)
            lid_width = int(w * 2.1)
            lid_height = int(lid_h + 64)
            lid_src = Image.new("RGBA", (lid_width, lid_height), (*lid_color, 255))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y0 - 32)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

        if self.bottom_lid > 0.01:
            lid_h = int(h * self.bottom_lid)
            lid_width = int(w * 2.1)
            lid_height = int(lid_h + 28)
            lid_src = Image.new("RGBA", (lid_width, lid_height), (*lid_color, 255))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y1 + 13 - lid_src.height)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

    def draw(self, bg_image):
        draw_w = max(4, int(self.w))
        draw_h = max(4, int(self.h))

        eye_img_size = int(max(self.base_w, self.base_h) * 2.5)
        eye_img = Image.new("RGBA", (eye_img_size, eye_img_size), (0, 0, 0, 0))
        eye_draw = ImageDraw.Draw(eye_img)

        off_x = max(-1, min(1, (self.current_pos[0] - self.base_x) / 30.0))
        off_y = max(-1, min(1, (self.current_pos[1] - self.base_y) / 20.0))

        cx = eye_img_size / 2
        cy = eye_img_size / 2
        x0 = cx - draw_w / 2
        y0 = cy - draw_h / 2
        x1 = cx + draw_w / 2
        y1 = cy + draw_h / 2

        self.draw_radial_eye(eye_draw, x0, y0, draw_w, draw_h, EYE_COLOR, (off_x, off_y))
        self.draw_eyelids(eye_img, (x0, y0, x1, y1))

        rotated = eye_img.rotate(self.current_rotation, resample=Image.BICUBIC, expand=False)
        paste_x = int(self.current_pos[0] - eye_img_size / 2)
        paste_y = int(self.current_pos[1] - eye_img_size / 2)
        bg_image.alpha_composite(rotated, (paste_x, paste_y))


def get_key_nonblocking():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def main():
    print("Initializing Displays (Dual SPI) - round eyes...")
    disp_l = None
    disp_r = None

    try:
        spi0 = board.SPI()
        disp_l = st7735.ST7735R(
            spi0,
            rotation=0,
            baudrate=24000000,
            bgr=True,
            cs=digitalio.DigitalInOut(board.CE1),
            dc=digitalio.DigitalInOut(board.D24),
            rst=digitalio.DigitalInOut(board.D25),
        )
    except Exception as e:
        print(f"Error init Left Display (SPI0): {e}")

    try:
        spi1 = busio.SPI(clock=board.D21, MOSI=board.D20, MISO=board.D19)
        disp_r = st7735.ST7735R(
            spi1,
            rotation=0,
            baudrate=24000000,
            bgr=True,
            cs=digitalio.DigitalInOut(board.D18),
            dc=digitalio.DigitalInOut(board.D23),
            rst=digitalio.DigitalInOut(board.D27),
        )
    except Exception as e:
        print(f"Error init Right Display (SPI1): {e}")

    center_x = SCREEN_WIDTH / 2
    center_y = SCREEN_HEIGHT / 2

    left_eye = RoundEye(center_x, center_y, scale=1.0, is_left=True)
    right_eye = RoundEye(center_x, center_y, scale=1.0, is_left=False)

    current_emotion = "idle"
    intensity = 0.45
    left_eye.set_emotion(current_emotion, intensity)
    right_eye.set_emotion(current_emotion, intensity)

    print("\nRound Emotion Tuner Ready")
    print("Keys: 0-9 + a/s/d/f/g emotions, [ ] intensity, b blink, q quit")

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            key = get_key_nonblocking()
            if key:
                if key == "q":
                    break
                if key in KEY_TO_EMOTION:
                    new_emotion = KEY_TO_EMOTION[key]
                    if new_emotion == current_emotion:
                        left_eye.set_emotion("idle", intensity)
                        right_eye.set_emotion("idle", intensity)
                    current_emotion = new_emotion
                    left_eye.set_emotion(current_emotion, intensity)
                    right_eye.set_emotion(current_emotion, intensity)
                    print(f"Emotion: {current_emotion} (intensity {intensity:.2f})")
                elif key == "[":
                    intensity = max(0.0, intensity - 0.05)
                    left_eye.set_emotion(current_emotion, intensity)
                    right_eye.set_emotion(current_emotion, intensity)
                    print(f"Intensity: {intensity:.2f}")
                elif key == "]":
                    intensity = min(1.0, intensity + 0.05)
                    left_eye.set_emotion(current_emotion, intensity)
                    right_eye.set_emotion(current_emotion, intensity)
                    print(f"Intensity: {intensity:.2f}")
                elif key == "b":
                    blink_speed = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
                    left_eye.start_blink(blink_speed)
                    right_eye.start_blink(blink_speed)

            left_eye.update()
            right_eye.update()

            if disp_l:
                img_l = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), BG_COLOR)
                left_eye.draw(img_l)
                disp_l.image(img_l.convert("RGB"))

            if disp_r:
                img_r = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), BG_COLOR)
                right_eye.draw(img_r)
                disp_r.image(img_r.convert("RGB"))

            time.sleep(0.02)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        black = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
        if disp_l:
            disp_l.image(black)
        if disp_r:
            disp_r.image(black)
        print("Displays cleared.")


if __name__ == "__main__":
    main()