import time
import board
import busio
import digitalio
from PIL import Image, ImageDraw
from adafruit_rgb_display import st7735
import math
import random

# --- Configuration ---
# User specified 160x128 (Rotation 90)
SCREEN_WIDTH = 128
SCREEN_HEIGHT = 160

EYE_COLOR = (255, 255, 255)  # White
BG_COLOR = (0, 0, 0) # Black background matches user test
EYE_SIZE = 120 # Base size
FLOOR_Y = SCREEN_HEIGHT - 5

# --- BlockyEye Class (Ported from test.py to PIL) ---
class BlockyEye:
    def __init__(self, x, y, scale=1.0, is_left=True):
        self.base_x, self.base_y = x, y
        self.current_pos = [float(x), float(y)]
        self.target_pos = [float(x), float(y)]
        
        # Physics
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
        self.rot_sensitivity = random.uniform(0.7, 0.9) 
        self.rot_speed = random.uniform(0.08, 0.15)
        
        self.is_left = is_left 
        self.blink_state = "IDLE" 
        self.vy = 0 
        self.jump_velocity = 0.0  # For elastic bounce effect
        self.blink_speed_mult = 1.0 
        
        self.noise_t = random.uniform(0, 100)

    def start_blink(self):
        if self.blink_state == "IDLE":
            self.blink_state = "DROPPING"
            self.vy = 40 * self.blink_speed_mult

    def start_pop(self):
        if self.blink_state == "IDLE":
            self.blink_state = "POPPING"
            self.current_w -= 8 
            self.current_h -= 8
            self.target_w = self.base_w
            self.target_h = self.base_h

    def update(self):
        # --- Physics Update (Same as test.py) ---
        if self.blink_state == "IDLE":
            # Noise
            t = time.time() + self.noise_t
            noise_x = (math.sin(t * 1.3) * 0.5 + math.sin(t * 0.7) * 0.3)
            noise_y = (math.cos(t * 1.1) * 0.5 + math.cos(t * 0.9) * 0.3)
            
            target_x_phys = self.target_pos[0] + noise_x
            target_y_phys = self.target_pos[1] + noise_y
            
            # Spring Physics (Position)
            stiffness = 0.08
            damping = 0.5
            
            force_x = (target_x_phys - self.current_pos[0]) * stiffness
            self.vel_x = (self.vel_x + force_x) * damping
            self.current_pos[0] += self.vel_x
            
            force_y = (target_y_phys - self.current_pos[1]) * stiffness
            self.vel_y = (self.vel_y + force_y) * damping
            self.current_pos[1] += self.vel_y
            
            # Constrain to screen bounds
            half_w = self.base_w / 2
            half_h = self.base_h / 2
            self.current_pos[0] = max(half_w, min(SCREEN_WIDTH - half_w, self.current_pos[0]))
            self.current_pos[1] = max(half_h, min(SCREEN_HEIGHT - half_h, self.current_pos[1]))

            # Rotation
            rel_x = self.current_pos[0] - self.base_x
            rel_y = self.current_pos[1] - self.base_y
            target_rot = (rel_x * 0.5 + rel_y * 0.8) * self.rot_sensitivity
            self.current_rotation += (target_rot - self.current_rotation) * self.rot_speed
            
            # Breathing / Dimensions
            t = time.time()
            breath_w = (math.sin(t * 1.5 + self.base_x) * 1.5 + math.sin(t * 0.5) * 1.0) 
            breath_h = (math.cos(t * 1.8 + self.base_y) * 1.5 + math.cos(t * 0.6) * 1.0)
            
            move_stretch_x = self.vel_x * 8.0 # Tuned down slightly for PIL visual
            move_stretch_y = self.vel_y * 8.0
            
            self.target_w = self.base_w + breath_w + (move_stretch_x * 0.5)
            self.target_h = self.base_h + breath_h - (move_stretch_y * 0.2)
            
        elif self.blink_state == "DROPPING":
            self.vy += 15 * self.blink_speed_mult
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
            squeeze_speed = 60 * self.blink_speed_mult
            spread_speed = 45 * self.blink_speed_mult
            self.current_h -= squeeze_speed
            self.current_w += spread_speed
            self.current_pos[1] = FLOOR_Y - self.current_h // 2
            
            if self.current_h <= 25: 
                self.current_h = 25
                self.jump_velocity = -7 * self.blink_speed_mult  # Smaller overshoot
                self.blink_state = "JUMPING"
                
        elif self.blink_state == "JUMPING":
            recovery_speed = max(0.1, min(0.95, 0.95 * self.blink_speed_mult))
            self.current_h += (self.base_h - self.current_h) * recovery_speed
            self.current_w += (self.base_w - self.current_w) * recovery_speed
            
            # Subtle overshoot upward then recover to original position
            target_y = self.target_pos[1]
            # Add small overshoot above target, then pull down
            overshoot = -3.0 * (1.0 - recovery_speed)  # Subtle upward overshoot
            self.current_pos[1] += (target_y + overshoot - self.current_pos[1]) * recovery_speed
            
            self.vel_x = (self.vel_x + (self.target_pos[0] - self.current_pos[0]) * 0.1) * 0.8
            self.current_pos[0] += self.vel_x
            
            # Constrain to screen bounds during jump recovery
            half_w = self.base_w / 2
            half_h = self.base_h / 2
            self.current_pos[0] = max(half_w, min(SCREEN_WIDTH - half_w, self.current_pos[0]))
            self.current_pos[1] = max(half_h, min(SCREEN_HEIGHT - half_h, self.current_pos[1]))
            
            if abs(self.current_h - self.base_h) < 5 and abs(self.current_pos[1] - target_y) < 5:
                self.current_h = self.base_h
                self.current_w = self.base_w
                self.current_pos[1] = target_y
                self.blink_state = "IDLE"
                self.vy = 0
                self.vel_x = 0
                self.vel_y = 0

        elif self.blink_state == "POPPING":
            self.current_w += (self.base_w - self.current_w) * 0.25
            self.current_h += (self.base_h - self.current_h) * 0.25
            if abs(self.current_w - self.base_w) < 1:
                self.current_w = self.base_w
                self.current_h = self.base_h
                self.blink_state = "IDLE"

        # Size Physics
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
        # Simulate radial gradient with concentric rectangles using PIL
        center_x = x + w/2
        center_y = y + h/2
        
        steps = 10
        for i in range(steps):
             size_factor = 1.0 - (i / steps) 
             current_w = w * size_factor
             current_h = h * size_factor
             
             if current_w <= 0 or current_h <= 0: continue
             
             b_factor = 0.85 + 0.15 * (i / steps)  # Even less fading 
             cur_color = (int(color[0] * b_factor), int(color[1] * b_factor), int(color[2] * b_factor))
             
             # Calculate offset shift
             shift_x = pupil_offset[0] * (1.0 - size_factor) * 15
             shift_y = pupil_offset[1] * (1.0 - size_factor) * 15
             
             cx = center_x + shift_x
             cy = center_y + shift_y
             
             x0 = cx - current_w / 2
             y0 = cy - current_h / 2
             x1 = cx + current_w / 2
             y1 = cy + current_h / 2
             
             cur_radius = max(2, int(radius * size_factor))
             
             # PIL rounded_rectangle
             draw.rounded_rectangle([x0, y0, x1, y1], radius=cur_radius, fill=cur_color)

    def draw(self, bg_image):
        draw_w = max(4, int(self.w))
        draw_h = max(4, int(self.h))
        
        needs_rotation = abs(self.current_rotation) > 1.0
        
        if needs_rotation:
             # Create temp image large enough to hold rotated eye
             eye_img_size = int(max(self.base_w, self.base_h) * 2.5) 
             eye_img = Image.new("RGBA", (eye_img_size, eye_img_size), (0,0,0,0))
             eye_draw = ImageDraw.Draw(eye_img)
             
             corner_radius = int(self.base_w * 0.25)
             off_x = max(-1, min(1, (self.current_pos[0] - self.base_x) / 30.0))
             off_y = max(-1, min(1, (self.current_pos[1] - self.base_y) / 20.0))
             
             cx, cy = eye_img_size/2, eye_img_size/2
             
             self.draw_radial_rect(eye_draw, cx - draw_w/2, cy - draw_h/2, 
                                   draw_w, draw_h, EYE_COLOR, corner_radius, (off_x, off_y))
             
             # Rotate
             rotated = eye_img.rotate(self.current_rotation, resample=Image.BICUBIC)
             
             # Paste onto background
             paste_x = int(self.current_pos[0] - eye_img_size/2)
             paste_y = int(self.current_pos[1] - eye_img_size/2)
             
             # Composite
             bg_image.alpha_composite(rotated, (paste_x, paste_y))
             
        else:
             draw = ImageDraw.Draw(bg_image)
             corner_radius = int(self.base_w * 0.25)
             off_x = max(-1, min(1, (self.current_pos[0] - self.base_x) / 30.0))
             off_y = max(-1, min(1, (self.current_pos[1] - self.base_y) / 20.0))
             
             x = self.current_pos[0] - draw_w/2
             y = self.current_pos[1] - draw_h/2
             
             self.draw_radial_rect(draw, x, y, draw_w, draw_h, EYE_COLOR, corner_radius, (off_x, off_y))


# --- HARDWARE SETUP (From User Snippet) ---
print("Initializing Displays (Dual SPI)...")

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


# --- Init Agents ---
center_x = SCREEN_WIDTH / 2
center_y = SCREEN_HEIGHT / 2

# Scale 1.0 fits loosely in 160x128
left_eye = BlockyEye(center_x, center_y, scale=1.0, is_left=True)
right_eye = BlockyEye(center_x, center_y, scale=1.0, is_left=False)

left_eye.base_x = center_x
left_eye.base_y = center_y
right_eye.base_x = center_x
right_eye.base_y = center_y

# Animation Vars
running = True
next_blink_time = time.time() + random.uniform(1, 4)
last_blink_time = time.time()
right_eye_target_queue = []
DELAY_SECONDS = 0.016
SACCADE_CHANCE = 0.05
POP_CHANCE = 0.005

print("Starting Animation Loop...")

try:
    while running:
        start_time = time.time()
        
        # --- Logic (Copied from test.py) ---
        
        # Blink
        if time.time() > next_blink_time:
            # Use same blink speed for both eyes to keep them in sync
            blink_speed = random.uniform(1.5, 2.5)
            left_eye.blink_speed_mult = blink_speed
            right_eye.blink_speed_mult = blink_speed
            left_eye.start_blink()
            right_eye.start_blink()
            last_blink_time = time.time()
            next_blink_time = time.time() + random.uniform(4, 8)

        # Pop
        if (random.random() < POP_CHANCE and 
            left_eye.blink_state == "IDLE" and
            time.time() > last_blink_time + 1.0 and 
            time.time() < next_blink_time - 1.0):
             left_eye.start_pop()
             right_eye.start_pop()

        # Saccade (Movement)
        if left_eye.blink_state == "IDLE":
            if random.random() < SACCADE_CHANCE:
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(3, 10) 
                
                dx = math.cos(angle) * distance
                dy = math.sin(angle) * distance
                
                new_target_x = left_eye.base_x + dx
                new_target_y = left_eye.base_y + dy
                
                # Clamp
                new_target_x = max(left_eye.base_x - 20, min(left_eye.base_x + 20, new_target_x))
                new_target_y = max(left_eye.base_y - 15, min(left_eye.base_y + 15, new_target_y))
                
                left_eye.target_pos = [new_target_x, new_target_y]
                
                offset_x = new_target_x - left_eye.base_x
                offset_y = new_target_y - left_eye.base_y
                right_target = [right_eye.base_x + offset_x, right_eye.base_y + offset_y]
                
                right_eye_target_queue.append((time.time() + DELAY_SECONDS, right_target))

        # Process Queue
        if right_eye_target_queue:
            if time.time() >= right_eye_target_queue[0][0]:
                _, target = right_eye_target_queue.pop(0)
                right_eye.target_pos = target

        # Update
        left_eye.update()
        right_eye.update()

        # --- Draw ---
        # Create RGBA images for composition (transparent helper) then paste on RGB black
        # Actually easier to just draw on fresh RGB black images
        
        canvas_l = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), (0,0,0,255))
        canvas_r = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), (0,0,0,255))
        
        left_eye.draw(canvas_l)
        right_eye.draw(canvas_r)
        
        # Convert to RGB for Display
        img_l_final = canvas_l.convert("RGB")
        img_r_final = canvas_r.convert("RGB")
        
        disp_l.image(img_l_final)
        disp_r.image(img_r_final)
        
        # FPS Control (Roughly)
        # diff = time.time() - start_time
        # if diff < 0.03: time.sleep(0.03 - diff)

except KeyboardInterrupt:
    print("Stopping...")
    # Clear screens
    black = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
    disp_l.image(black)
    disp_r.image(black)
