import time
import board
import busio
import digitalio
from PIL import Image, ImageDraw
from adafruit_rgb_display import st7735

# --- Hardware Setup (Same as yours) ---
spi0 = board.SPI() 
disp_l = st7735.ST7735R(spi0, rotation=90, baudrate=32000000, bgr=True, # Bumped to 16MHz
    cs=digitalio.DigitalInOut(board.CE1), dc=digitalio.DigitalInOut(board.D24), rst=digitalio.DigitalInOut(board.D25))

spi1 = busio.SPI(clock=board.D21, MOSI=board.D20, MISO=board.D19) 
disp_r = st7735.ST7735R(spi1, rotation=90, baudrate=32000000, bgr=True, # Bumped to 16MHz
    cs=digitalio.DigitalInOut(board.D18), dc=digitalio.DigitalInOut(board.D23), rst=digitalio.DigitalInOut(board.D27))

width, height = 160, 128
total_width = width * 2
rect_size = 20

# Initial position and velocity
x, y = 0, height // 2 - (rect_size // 2)
vx, vy = 8, 4  # Pixels per frame

print("Testing Refresh Rate. Watch for tearing at the seam.")

last_time = time.time()
frames = 0

try:
    while True:
        # 1. Update Position
        x += vx
        y += vy

        # Bounce logic
        if x <= 0 or x >= total_width - rect_size: vx *= -1
        if y <= 0 or y >= height - rect_size: vy *= -1

        # 2. Create Blank Frames
        img_l = Image.new("RGB", (width, height), (0, 0, 0))
        img_r = Image.new("RGB", (width, height), (0, 0, 0))
        draw_l = ImageDraw.Draw(img_l)
        draw_r = ImageDraw.Draw(img_r)

        # 3. Draw the block (Split logic)
        if x < width: # Block is at least partially on Left screen
            draw_l.rectangle([x, y, x + rect_size, y + rect_size], fill=(255, 255, 0))
        if x + rect_size > width: # Block is at least partially on Right screen
            draw_r.rectangle([x - width, y, x - width + rect_size, y + rect_size], fill=(255, 255, 0))

        # 4. Push to Hardware
        disp_l.image(img_l)
        disp_r.image(img_r)

        # 5. Calculate FPS
        frames += 1
        if time.time() - last_time >= 1.0:
            print(f"FPS: {frames}")
            frames = 0
            last_time = time.time()

except KeyboardInterrupt:
    print("\nTest stopped.")