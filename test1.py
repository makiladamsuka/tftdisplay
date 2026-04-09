import time
import board
import busio
import digitalio
from PIL import Image, ImageDraw
from adafruit_rgb_display import st7735

# --- SPI 0 (Left Screen) ---
# Left eye uses SPI 0 (default hardware SPI bus)
spi0 = board.SPI() 
disp_l = st7735.ST7735R(spi0, rotation=90, baudrate=8000000, bgr=True,
    cs=digitalio.DigitalInOut(board.CE1),   
    dc=digitalio.DigitalInOut(board.D24),   
    rst=digitalio.DigitalInOut(board.D25))  

# --- SPI 1 (Right Screen) ---
# Right eye uses SPI 1 (custom SPI bus with clock D21, MOSI D20, MISO D19)
spi1 = busio.SPI(clock=board.D21, MOSI=board.D20, MISO=board.D19) 
disp_r = st7735.ST7735R(spi1, rotation=90, baudrate=8000000, bgr=True,
    cs=digitalio.DigitalInOut(board.D18),   
    dc=digitalio.DigitalInOut(board.D23),   
    rst=digitalio.DigitalInOut(board.D27))  

width, height = 160, 128

# Colors to test: Red, Green, Blue, White
test_colors = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 255) # White
]

print("Starting Dual-Display Color Test...")

try:
    while True:
        for color in test_colors:
            # Create a solid color image for both
            img_l = Image.new("RGB", (width, height), color)
            img_r = Image.new("RGB", (width, height), color)
            
            # Add a small label so you know which is which
            draw_l = ImageDraw.Draw(img_l)
            draw_r = ImageDraw.Draw(img_r)
            draw_l.text((10, 10), "LEFT BUS 0", fill=(0,0,0) if color == (255,255,255) else (255,255,255))
            draw_r.text((10, 10), "RIGHT BUS 1", fill=(0,0,0) if color == (255,255,255) else (255,255,255))

            # Push to screens
            disp_l.image(img_l)
            disp_r.image(img_r)
            
            print(f"Displaying color: {color}")
            time.sleep(2) # Hold each color for 2 seconds

except KeyboardInterrupt:
    # Clear to black on exit
    disp_l.image(Image.new("RGB", (width, height), (0, 0, 0)))
    disp_r.image(Image.new("RGB", (width, height), (0, 0, 0)))
    print("\nTest complete.")