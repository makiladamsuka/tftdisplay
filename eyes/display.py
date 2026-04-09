"""
PyGame & Hardware Display — Supports desktop preview and ST7735 TFTs.

Renders each eye at native TFT resolution (128×160).
- Desktop: Scales up 3× for PyGame window.
- Hardware: Sends PIL images to ST7735 displays via SPI.
"""

import os
import time
import random
import threading
from abc import ABC, abstractmethod

from eye_renderer import EyeRenderer
from life_engine import LifeEngine
from eye_state import EyeStateMachine, EyeState
from voice_sync import VoiceSync


# ─── TFT display constants ──────────────────────────────────────────────
TFT_WIDTH = 128       # Width of one ST7735 TFT
TFT_HEIGHT = 160      # Height of one ST7735 TFT
TFT_GAP = 12          # Gap between the two TFT panels in preview
SCALE = 3             # How much to scale up for desktop preview


class BaseEyeDisplay(ABC):
    """
    Base class for eye display logic.
    Handles the simulation loop (FSM, LifeEngine, VoiceSync, Renderer updates).
    Subclasses handle the actual output (PyGame window vs Hardware SPI).
    """

    # Total canvas = two TFTs side by side with a gap
    CANVAS_WIDTH = TFT_WIDTH * 2 + TFT_GAP   # 268 native
    CANVAS_HEIGHT = TFT_HEIGHT                 # 160 native

    # Startle vibration amplitude (pixels at TFT scale)
    STARTLE_VIBRATE_AMP = 2.0

    def __init__(self):
        # Renderer works at the native TFT canvas resolution
        self.renderer = EyeRenderer(
            screen_width=self.CANVAS_WIDTH,
            screen_height=self.CANVAS_HEIGHT,
        )
        self.life = LifeEngine()
        self.voice = VoiceSync()
        self.fsm = EyeStateMachine(on_state_change=self._on_state_change)

        self.running = False
        self._canvas = None  # Set by PygameEyeDisplay when using desktop
        self._simulate_voice = False

    def _on_state_change(self, emotion_name: str, bypass_inertia: bool = False):
        """Called by FSM when state changes — triggers renderer transition."""
        self.renderer.set_emotion(emotion_name, force=bypass_inertia)

        # Force a blink on startle
        if emotion_name == "startled":
            self.life.force_blink()

        # Reset voice sync when not talking
        if emotion_name != "talking":
            self.voice.reset()

    def update_logic(self, dt: float):
        """Update all simulation subsystems."""
        # ── Update subsystems ──
        self.fsm.update(dt)
        life_offsets = self.life.update(dt)
        self.voice.update(dt)

        # Simulate voice if toggled on
        if self._simulate_voice:
            self.voice._current_amplitude = 0.3 + random.random() * 0.5
            self.voice._target_amplitude = self.voice._current_amplitude

        # Apply voice amplitude to renderer
        self.renderer.set_voice_amplitude(self.voice.amplitude)

        # Update renderer transition interpolation
        self.renderer.update(dt)

        # Apply life behavior offsets AFTER renderer update
        self.renderer.apply_life_offsets(life_offsets)

        # Apply blink simultaneously to both eyes (exact same values)
        if life_offsets.blink_amount > 0.01:
            amount = life_offsets.blink_amount
            bottom = amount * 0.3
            for params in (self.renderer.left_params, self.renderer.right_params):
                params.top_lid = amount
                params.bottom_lid = bottom

        # Startle vibration
        if self.fsm.is_startled:
            for params in (self.renderer.left_params, self.renderer.right_params):
                params.x_offset += random.uniform(
                    -self.STARTLE_VIBRATE_AMP, self.STARTLE_VIBRATE_AMP
                )
                params.y_offset += random.uniform(
                    -self.STARTLE_VIBRATE_AMP, self.STARTLE_VIBRATE_AMP
                )

        # ── Render at native TFT resolution (pygame canvas or PIL in hardware) ──
        if self._canvas is not None:
            self.renderer.render(self._canvas)

    def stop(self):
        """Signal the display loop to stop."""
        self.running = False


class PygameEyeDisplay(BaseEyeDisplay):
    """
    Renders to a desktop PyGame window for testing/preview.
    """
    SCREEN_WIDTH = BaseEyeDisplay.CANVAS_WIDTH * SCALE
    SCREEN_HEIGHT = BaseEyeDisplay.CANVAS_HEIGHT * SCALE
    FPS = 60
    WINDOW_TITLE = "👁 Procedural Eyes (128×160 TFT Preview)"

    def __init__(self):
        super().__init__()
        self.clock = None
        self.screen = None
        self._show_debug = False

    def init_pygame(self):
        """Initialize PyGame window."""
        import pygame
        self._canvas = pygame.Surface((self.CANVAS_WIDTH, self.CANVAS_HEIGHT))
        pygame.init()
        pygame.display.set_caption(self.WINDOW_TITLE)
        self.screen = pygame.display.set_mode(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        )
        self.clock = pygame.time.Clock()
        self.running = True

    def run_loop(self):
        """Main PyGame loop."""
        import pygame
        if not self.screen:
            self.init_pygame()

        font = pygame.font.SysFont("monospace", 14)

        while self.running:
            dt = self.clock.tick(self.FPS) / 1000.0
            dt = min(dt, 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            if not self.running:
                break

            self.update_logic(dt)
            self._draw_preview()
            
            if self._show_debug:
                self._draw_debug(font)

            pygame.display.flip()

        pygame.quit()

    def _draw_preview(self):
        """Draw borders and scale up for preview."""
        import pygame
        # ── Draw TFT panel borders on canvas ──
        border_color = (50, 50, 70)
        pygame.draw.rect(
            self._canvas, border_color,
            pygame.Rect(0, 0, TFT_WIDTH, TFT_HEIGHT), 1
        )
        pygame.draw.rect(
            self._canvas, border_color,
            pygame.Rect(TFT_WIDTH + TFT_GAP, 0, TFT_WIDTH, TFT_HEIGHT), 1
        )

        # ── Scale up to preview window ──
        scaled = pygame.transform.scale(self._canvas, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.screen.blit(scaled, (0, 0))

    def _handle_key(self, key: int):
        """Handle keyboard input for testing."""
        import pygame
        key_map = {
            pygame.K_1: EyeState.IDLE,
            pygame.K_2: EyeState.LISTENING,
            pygame.K_3: EyeState.HAPPY,
            pygame.K_4: EyeState.TALKING,
            pygame.K_5: EyeState.PROCESSING,
            pygame.K_6: EyeState.CONFUSED,
            pygame.K_7: EyeState.STARTLED,
        }

        if key in key_map:
            self.fsm.transition_to(key_map[key])
        elif key == pygame.K_b:
            self.life.force_blink()
        elif key == pygame.K_SPACE:
            self.life.enabled = not self.life.enabled
        elif key == pygame.K_d:
            self._show_debug = not self._show_debug
        elif key == pygame.K_v:
            self._simulate_voice = not self._simulate_voice
            if not self._simulate_voice:
                self.voice.reset()
        elif key == pygame.K_ESCAPE:
            self.running = False

    def _draw_debug(self, font):
        import pygame
        lines = [
            f"State: {self.fsm.current_state.name}",
            f"Emotion: {self.renderer.current_emotion}",
            f"Voice Amp: {self.voice.amplitude:.3f}",
            f"Life: {'ON' if self.life.enabled else 'OFF'}",
            f"FPS: {self.clock.get_fps():.0f}",
        ]
        y = 10
        for line in lines:
            text_surf = font.render(line, True, (100, 200, 100))
            self.screen.blit(text_surf, (10, y))
            y += 18


# Match test_two_st7735.py: set True if Right DC is on GPIO 23 (Pin 16) to reduce flicker
USE_SEPARATE_DC = False


class HardwareEyeDisplay(BaseEyeDisplay):
    """
    Renders to two ST7735 displays attached to a Raspberry Pi.
    Pin layout matches test_two_st7735.py:
      Left:  CS=CE1 (Pin 26), RST=GPIO 25 (Pin 22), DC=GPIO 24 (Pin 18)
      Right: CS=CE0 (Pin 24), RST=GPIO 27 (Pin 13), DC=GPIO 24 or GPIO 23 if USE_SEPARATE_DC
    """

    SETTLE_MS = 50  # ms between display updates to reduce SPI crosstalk

    def __init__(self):
        super().__init__()
        # Import hardware libraries here so they don't break desktop dev
        import digitalio
        import board
        from PIL import Image
        from adafruit_rgb_display import st7735

        self.Image = Image  # Keep reference
        spi = board.SPI()

        # Left: CE1, RST=GPIO 25, DC=GPIO 24
        cs_left = digitalio.DigitalInOut(board.CE1)
        rst_left = digitalio.DigitalInOut(board.D25)
        dc_left = digitalio.DigitalInOut(board.D24)

        # Right: CE0, RST=GPIO 27, DC=shared or GPIO 23
        cs_right = digitalio.DigitalInOut(board.CE0)
        rst_right = digitalio.DigitalInOut(board.D27)
        dc_right = (
            digitalio.DigitalInOut(board.D23) if USE_SEPARATE_DC else dc_left
        )

        # Hardware reset both (order doesn't matter)
        for pin in (rst_left, rst_right):
            pin.direction = digitalio.Direction.OUTPUT
            pin.value = False
        time.sleep(0.1)
        for pin in (rst_left, rst_right):
            pin.value = True
        time.sleep(0.1)

        baud = 6_000_000
        self.disp_left = st7735.ST7735R(
            spi, rotation=90, cs=cs_left, dc=dc_left, rst=rst_left, baudrate=baud
        )
        self.disp_right = st7735.ST7735R(
            spi, rotation=90, cs=cs_right, dc=dc_right, rst=rst_right, baudrate=baud
        )
        self.running = True

    def run_loop(self):
        """Main hardware loop. No pygame: uses render_to_pil() and sends to ST7735."""
        target_fps = 30
        last = time.perf_counter()

        while self.running:
            now = time.perf_counter()
            dt = min(now - last, 0.05)
            last = now
            time.sleep(max(0, 1.0 / target_fps - dt))

            self.update_logic(dt)
            full_image = self.renderer.render_to_pil()

            # Crop out the two eyes (each 128×160)
            img_left = full_image.crop((0, 0, TFT_WIDTH, TFT_HEIGHT))
            right_x = TFT_WIDTH + TFT_GAP
            img_right = full_image.crop((right_x, 0, right_x + TFT_WIDTH, TFT_HEIGHT))

            # adafruit_rgb_display rotates the image by 90° before sending, so it expects
            # 160×128 (after our rotation it fits 128×160 logical). Rotate -90 to get 160×128.
            img_left = img_left.rotate(-90, expand=True)
            img_right = img_right.rotate(-90, expand=True)

            # Send to displays (right first, then left + settle — reduces flicker)
            self.disp_right.image(img_right)
            time.sleep(self.SETTLE_MS / 1000.0)
            self.disp_left.image(img_left)


# ─── Legacy Wrapper/Main ───
def run_demo():
    """Run standalone demo with Pygame."""
    print("🎮 Procedural Eyes Demo (Desktop Mode)")
    display = PygameEyeDisplay()
    display.fsm.transition_to(EyeState.IDLE)
    display.run_loop()


if __name__ == "__main__":
    run_demo()
