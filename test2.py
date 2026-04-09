import time
import board
import busio
import digitalio
from PIL import Image, ImageDraw
from adafruit_rgb_display import st7735

# --- SPI 0 (Left Screen) ---
spi0 = board.SPI() 
disp_l = st7735.ST7735R(spi0, rotation=90, baudrate=8000000, bgr=True,
    cs=digitalio.DigitalInOut(board.CE1),   # Pin 26
    dc=digitalio.DigitalInOut(board.D24),   # Pin 18
    rst=digitalio.DigitalInOut(board.D25))  # Pin 22

# --- SPI 1 (Right Screen) ---
# Clock=Pin 40 (D21), MOSI=Pin 38 (D20), MISO=Pin 35 (D19)
spi1 = busio.SPI(clock=board.D21, MOSI=board.D20, MISO=board.D19) 
disp_r = st7735.ST7735R(spi1, rotation=90, baudrate=8000000, bgr=True,
    cs=digitalio.DigitalInOut(board.D18),   # Pin 12
    dc=digitalio.DigitalInOut(board.D23),   # Pin 16
    rst=digitalio.DigitalInOut(board.D27))  # Pin 13

width, height = 160, 128
mid_y = height // 2

# EKG Pulse Shape
pulse = [0, 0, -2, -8, 40, -20, 10, 0, 0]
wave_data = [0] * 320 # Combined width of both screens

print("Dual-Bus Heartbeat Running. No ghosts allowed.")

try:
    while True:
        # Shift wave and add pulses
        wave_data = wave_data[2:] + [0, 0]
        if time.time() % 1.5 < 0.1 and sum(wave_data[-20:]) == 0:
            wave_data[-len(pulse):] = pulse

        # Create frames
        img_l = Image.new("RGB", (width, height), (0, 0, 0))
        img_r = Image.new("RGB", (width, height), (0, 0, 0))
        draw_l = ImageDraw.Draw(img_l)
        draw_r = ImageDraw.Draw(img_r)

        # Plot across both screens
        for x in range(len(wave_data) - 1):
            y1, y2 = mid_y - wave_data[x], mid_y - wave_data[x+1]
            if x < 159:
                draw_l.line((x, y1, x+1, y2), fill=(0, 255, 0), width=2)
            elif x >= 160 and x < 319:
                draw_r.line((x-160, y1, x-159, y2), fill=(0, 255, 0), width=2)

        # Push to hardware
        disp_l.image(img_l)
        disp_r.image(img_r)
        
        # We can run faster now because the buses don't interfere!
        time.sleep(0.001) 

except KeyboardInterrupt:
    print("\nMonitor stopped.")