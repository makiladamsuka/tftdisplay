"""
Procedural Eye Renderer — Vector-Style Living Eyes

Draws two plain rounded-rectangle eyes (no pupils, no iris, no highlights).
Uses a bouncy spring easing for lively transitions between emotions.
Every parameter is animatable for smooth interpolation.
"""

import math
import time
import random
from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, Tuple

from PIL import Image, ImageDraw

try:
    import pygame
except ImportError:
    pygame = None  # Optional: only needed for render(surface) / desktop


# ─── Eye Parameters (all animatable) ────────────────────────────────────

@dataclass
class EyeParams:
    """Complete state of one eye — every field can be smoothly interpolated."""
    # Shape
    eye_width: float = 80.0       # Width of eye opening
    eye_height: float = 100.0     # Height of eye opening
    corner_radius: float = 20

    # Eyelids (0.0 = fully open, 1.0 = fully closed)
    top_lid: float = 0.0
    bottom_lid: float = 0.0
    top_lid_angle: float = 0.0    # degrees: negative = angry, positive = sad

    # Gaze direction (-1.0 to +1.0) — moves the whole eye rectangle
    gaze_x: float = 0.0           # -1 = left, +1 = right
    gaze_y: float = 0.0           # -1 = up, +1 = down

    # Position offsets (for breathing, saccades etc.)
    x_offset: float = 0.0
    y_offset: float = 0.0

    # Eye color
    eye_color: Tuple[int, int, int] = (255, 255, 255)   # White


@dataclass
class PendingEmotion:
    """Queued emotion that will be applied once inertia cooldown allows."""
    name: str
    duration: Optional[float] = None


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def lerp_color(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Interpolate between two RGB colors."""
    return (
        int(lerp(a[0], b[0], t)),
        int(lerp(a[1], b[1], t)),
        int(lerp(a[2], b[2], t)),
    )


def lerp_params(a: 'EyeParams', b: 'EyeParams', t: float) -> 'EyeParams':
    """Interpolate all parameters between two EyeParams states."""
    result = EyeParams()
    result.eye_width = lerp(a.eye_width, b.eye_width, t)
    result.eye_height = lerp(a.eye_height, b.eye_height, t)
    result.corner_radius = lerp(a.corner_radius, b.corner_radius, t)
    result.top_lid = lerp(a.top_lid, b.top_lid, t)
    result.bottom_lid = lerp(a.bottom_lid, b.bottom_lid, t)
    result.top_lid_angle = lerp(a.top_lid_angle, b.top_lid_angle, t)
    result.gaze_x = lerp(a.gaze_x, b.gaze_x, t)
    result.gaze_y = lerp(a.gaze_y, b.gaze_y, t)
    result.x_offset = lerp(a.x_offset, b.x_offset, t)
    result.y_offset = lerp(a.y_offset, b.y_offset, t)
    result.eye_color = lerp_color(a.eye_color, b.eye_color, t)
    return result


# ─── Easing Functions ───────────────────────────────────────────────────

def ease_in_out_cubic(t: float) -> float:
    """Smooth start and end — natural for emotions."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_bounce(t: float) -> float:
    """Bouncy overshoot — the eye overshoots then springs back."""
    n1 = 7.5625
    d1 = 2.75
    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


def ease_out_elastic(t: float) -> float:
    """Elastic spring — overshoots and wobbles back. Very lively."""
    if t == 0 or t == 1:
        return t
    c4 = (2 * math.pi) / 3
    return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def ease_out_back(t: float) -> float:
    """Overshoot then settle — medium elastic with subtle overshoot."""
    c1 = 1.2
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


# ─── Emotion Presets ────────────────────────────────────────────────────

WHITE = (255, 255, 255)

# Human-like / Vector-style emotion presets: subtle lids, natural gaze
EMOTION_PRESETS = {
    "idle": EyeParams(
        eye_width=110, eye_height=120, corner_radius=20,
        top_lid=0.04, bottom_lid=0.0, top_lid_angle=0,
        gaze_x=0, gaze_y=0,
        eye_color=WHITE,
    ),
    "listening": EyeParams(
        eye_width=114, eye_height=128, corner_radius=20,
        top_lid=0.0, bottom_lid=0.0, top_lid_angle=0,
        gaze_x=0, gaze_y=0,
        eye_color=WHITE,
    ),
    # Happy: wider, more open eyes with slight upward curve and cheerful gaze
    "happy": EyeParams(
        eye_width=118, eye_height=110, corner_radius=20,
        top_lid=0.0, bottom_lid=0.08, top_lid_angle=-5,
        gaze_x=0, gaze_y=-0.10,
        eye_color=WHITE,
    ),
    "talking": EyeParams(
        eye_width=108, eye_height=112, corner_radius=20,
        top_lid=0.03, bottom_lid=0.0, top_lid_angle=0,
        gaze_x=0, gaze_y=0,
        eye_color=WHITE,
    ),
    "processing": EyeParams(
        eye_width=102, eye_height=88, corner_radius=20,
        top_lid=0.18, bottom_lid=0.08, top_lid_angle=1,
        gaze_x=0.25, gaze_y=-0.15,
        eye_color=WHITE,
    ),
    "confused": EyeParams(
        eye_width=108, eye_height=118, corner_radius=20,
        top_lid=0.12, bottom_lid=0.0, top_lid_angle=8,
        gaze_x=0.08, gaze_y=0.06,
        eye_color=WHITE,
    ),
    "startled": EyeParams(
        eye_width=118, eye_height=140, corner_radius=20,
        top_lid=0.0, bottom_lid=0.0, top_lid_angle=0,
        gaze_x=0, gaze_y=0,
        eye_color=WHITE,
    ),
    "sad": EyeParams(
        eye_width=130, eye_height=145, corner_radius=20,
        top_lid=0.75, bottom_lid=0.0, top_lid_angle=15,
        gaze_x=0, gaze_y=0.08,
        eye_color=WHITE,
    ),
    "angry": EyeParams(
        eye_width=128, eye_height=140, corner_radius=20,
        top_lid=0.75, bottom_lid=0.0, top_lid_angle=-20,
        gaze_x=0, gaze_y=0,
        eye_color=WHITE,
    ),
}

# Confused: right eye slightly different (one brow up = human confused)
CONFUSED_RIGHT_EYE_OVERRIDE = EyeParams(
    eye_width=112, eye_height=128, corner_radius=20,
    top_lid=0.0, bottom_lid=0.0, top_lid_angle=-6,
    gaze_x=0.12, gaze_y=-0.04,
    eye_color=WHITE,
)

# Transition durations — snappy for responsiveness (Vector-like)
TRANSITION_SPEEDS = {
    "idle": 0.35,
    "listening": 0.12,
    "happy": 0.25,
    "talking": 0.18,
    "processing": 0.35,
    "confused": 0.25,
    "startled": 0.06,
    "sad": 0.40,
    "angry": 0.20,
}

# Which easing to use per emotion (bouncy for lively ones, smooth for calm)
TRANSITION_EASING = {
    "idle": ease_out_back,
    "listening": ease_out_back,
    "happy": ease_out_back,
    "talking": ease_out_back,
    "processing": ease_out_back,
    "confused": ease_out_back,
    "startled": ease_out_back,
    "sad": ease_out_back,
    "angry": ease_out_back,
}


# ─── Eye Renderer ───────────────────────────────────────────────────────

class EyeRenderer:
    """
    Renders a pair of plain rounded-rectangle eyes onto a PyGame surface.
    No pupils, no iris — just clean solid shapes with glow and eyelids.
    Uses bouncy easing for lively transitions.
    """

    BG_COLOR = (26, 26, 46)       # Dark charcoal-blue
    GLOW_LAYERS = 3               # Number of glow layers behind eye
    GAZE_RANGE = 30               # Max pixels the eye moves from gaze (Vector-style look-around)

    def __init__(
        self,
        screen_width: int = 800,
        screen_height: int = 400,
        min_emotion_interval: float = 0.75,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Center positions of left and right eyes
        self.left_center = (screen_width // 4, screen_height // 2)
        self.right_center = (3 * screen_width // 4, screen_height // 2)

        # Current interpolated params (what's actually drawn)
        self.left_params = deepcopy(EMOTION_PRESETS["idle"])
        self.right_params = deepcopy(EMOTION_PRESETS["idle"])

        # Target params (what we're transitioning toward)
        self.target_left = deepcopy(EMOTION_PRESETS["idle"])
        self.target_right = deepcopy(EMOTION_PRESETS["idle"])

        # Transition tracking
        self.transition_start_left = deepcopy(EMOTION_PRESETS["idle"])
        self.transition_start_right = deepcopy(EMOTION_PRESETS["idle"])
        self.transition_progress = 1.0  # 1.0 = done
        self.transition_duration = 0.4
        self.transition_start_time = 0.0
        self.transition_easing = ease_in_out_cubic

        # Voice sync overlay
        self.voice_amplitude = 0.0      # 0.0 to 1.0
        self.voice_ema = 0.0            # Exponential moving average for voice pulse
        self.voice_ema_decay = 0.4      # Decay rate

        # Current emotion name
        self.current_emotion = "idle"
        self.min_emotion_interval = max(0.0, float(min_emotion_interval))
        self.last_emotion_change_time = 0.0
        self.pending_emotion: Optional[PendingEmotion] = None
        self.pending_apply_time = 0.0

    def set_emotion(
        self,
        emotion_name: str,
        duration: Optional[float] = None,
        force: bool = False,
    ):
        """Trigger smooth transition to a new emotion with optional inertia bypass."""
        if emotion_name not in EMOTION_PRESETS:
            return

        now = time.time()
        is_new_emotion = emotion_name != self.current_emotion

        if (
            is_new_emotion
            and not force
            and self.min_emotion_interval > 0.0
            and (now - self.last_emotion_change_time) < self.min_emotion_interval
        ):
            # Keep only the last queued emotion request to avoid stale transitions.
            self.pending_emotion = PendingEmotion(name=emotion_name, duration=duration)
            self.pending_apply_time = self.last_emotion_change_time + self.min_emotion_interval
            return

        self.current_emotion = emotion_name
        if is_new_emotion:
            self.last_emotion_change_time = now
        self.pending_emotion = None
        self.pending_apply_time = 0.0

        # Snapshot current state as transition start
        self.transition_start_left = deepcopy(self.left_params)
        self.transition_start_right = deepcopy(self.right_params)

        # Set target
        self.target_left = deepcopy(EMOTION_PRESETS[emotion_name])
        if emotion_name == "confused":
            self.target_right = deepcopy(CONFUSED_RIGHT_EYE_OVERRIDE)
        else:
            self.target_right = deepcopy(EMOTION_PRESETS[emotion_name])

        # Duration & easing
        self.transition_duration = duration or TRANSITION_SPEEDS.get(emotion_name, 0.3)
        self.transition_easing = TRANSITION_EASING.get(emotion_name, ease_in_out_cubic)
        self.transition_start_time = time.time()
        self.transition_progress = 0.0

    def set_voice_amplitude(self, amplitude: float):
        """Set current voice amplitude for voice sync (0.0 – 1.0) with EMA."""
        self.voice_amplitude = max(0.0, min(1.0, amplitude))
        # Smooth the pulse
        self.voice_ema += (self.voice_amplitude - self.voice_ema) * self.voice_ema_decay

    def apply_life_offsets(self, offsets: 'LifeOffsets'):
        """Apply additive offsets from LifeEngine (saccades, breathing, etc.)."""
        # Apply Left Eye Offsets
        self.left_params.gaze_x += offsets.left.gaze_dx
        self.left_params.gaze_y += offsets.left.gaze_dy
        self.left_params.y_offset += offsets.left.y_offset
        self.left_params.x_offset += offsets.left.x_offset
        self.left_params.top_lid = max(0.0, min(1.0, self.left_params.top_lid + offsets.left.lid_offset))

        # Apply Right Eye Offsets
        self.right_params.gaze_x += offsets.right.gaze_dx
        self.right_params.gaze_y += offsets.right.gaze_dy
        self.right_params.y_offset += offsets.right.y_offset
        self.right_params.x_offset += offsets.right.x_offset
        self.right_params.top_lid = max(0.0, min(1.0, self.right_params.top_lid + offsets.right.lid_offset))

    def update(self, dt: float):
        """Advance the interpolation by dt seconds."""
        if self.pending_emotion is not None and time.time() >= self.pending_apply_time:
            queued = self.pending_emotion
            self.pending_emotion = None
            self.pending_apply_time = 0.0
            self.set_emotion(queued.name, duration=queued.duration, force=True)

        if self.transition_progress < 1.0:
            elapsed = time.time() - self.transition_start_time
            raw_t = min(1.0, elapsed / max(self.transition_duration, 0.001))
            self.transition_progress = raw_t

            # Apply the emotion-specific easing (bouncy / elastic / smooth)
            eased_t = self.transition_easing(raw_t)

            # Interpolate both eyes
            self.left_params = lerp_params(
                self.transition_start_left, self.target_left, eased_t
            )
            self.right_params = lerp_params(
                self.transition_start_right, self.target_right, eased_t
            )
        else:
            # Snap to clean target each frame so life offsets don't accumulate
            self.left_params = deepcopy(self.target_left)
            self.right_params = deepcopy(self.target_right)

        # Apply voice sync (squash and stretch), same scale both eyes
        if self.voice_ema > 0.01:
            amp = self.voice_ema
            h_add = amp * 10.0
            w_sub = amp * 5.0
            self.left_params.eye_height += h_add
            self.left_params.eye_width -= w_sub
            self.right_params.eye_height += h_add
            self.right_params.eye_width -= w_sub

        # ── Subtle shared micro-variation (Vector-style: both eyes in sync) ──
        t = time.time()
        micro = math.sin(t * 1.2) * 0.5
        self.left_params.eye_width += micro
        self.left_params.eye_height += math.cos(t * 1.0) * 0.5
        self.right_params.eye_width += micro
        self.right_params.eye_height += math.cos(t * 1.0) * 0.5

        # ── Processing Jitter ──
        # Adds high-frequency vibration during thinking (reduced for stability)
        if self.current_emotion == "processing":
            jitter_amp = 0.8  # Reduced from 1.5
            self.left_params.x_offset += random.uniform(-jitter_amp, jitter_amp)
            self.left_params.y_offset += random.uniform(-jitter_amp, jitter_amp)
            self.right_params.x_offset += random.uniform(-jitter_amp, jitter_amp)
            self.right_params.y_offset += random.uniform(-jitter_amp, jitter_amp)

    def render(self, surface: "pygame.Surface"):
        """Draw both eyes onto the given surface (requires pygame). Use render_to_pil() for hardware."""
        if pygame is None:
            raise RuntimeError("Pygame is required for render(surface). Use render_to_pil() for hardware.")
        surface.fill(self.BG_COLOR)

        self._draw_eye(surface, self.left_center, self.left_params, mirror=False)
        self._draw_eye(surface, self.right_center, self.right_params, mirror=True)

    def _draw_eye(self, surface: pygame.Surface,
                  center: Tuple[int, int], params: EyeParams, mirror: bool):
        """Draw a single plain rounded rectangle eye."""
        # Gaze offsets move the whole eye
        gaze_offset_x = params.gaze_x * self.GAZE_RANGE
        gaze_offset_y = params.gaze_y * self.GAZE_RANGE

        cx_raw = center[0] + params.x_offset + gaze_offset_x
        cy_raw = center[1] + params.y_offset + gaze_offset_y

        w = params.eye_width
        h = params.eye_height
        r = min(params.corner_radius, w / 2, h / 2)

        # ── STRICT CLAMPING ──
        # Ensure eye shape stays within its 128x160 half-panel
        panel_min_x = 0 if not mirror else (self.screen_width // 2)
        panel_max_x = (self.screen_width // 2) if not mirror else self.screen_width
        
        # Clamp center so the eye rectangle stays fully inside the panel
        cx = max(panel_min_x + w/2, min(panel_max_x - w/2, cx_raw))
        cy = max(h/2, min(self.screen_height - h/2, cy_raw))

        eye_rect = pygame.Rect(cx - w / 2, cy - h / 2, w, h)

        # ── Glow effect (subtle halo behind eye) ──
        for i in range(self.GLOW_LAYERS, 0, -1):
            glow_alpha = 15 + i * 10
            glow_expand = i * 8
            glow_rect = eye_rect.inflate(glow_expand, glow_expand)
            glow_r = r + glow_expand / 2
            glow_color = (
                params.eye_color[0] // 4,
                params.eye_color[1] // 4,
                params.eye_color[2] // 4,
            )
            glow_surf = pygame.Surface(
                (glow_rect.width, glow_rect.height), pygame.SRCALPHA
            )
            pygame.draw.rect(
                glow_surf, (*glow_color, glow_alpha),
                pygame.Rect(0, 0, glow_rect.width, glow_rect.height),
                border_radius=int(glow_r)
            )
            surface.blit(glow_surf, glow_rect.topleft)

        # ── Main eye shape (plain filled rounded rect — nothing inside) ──
        pygame.draw.rect(surface, params.eye_color, eye_rect, border_radius=int(r))

        # ── Eyelids ──
        self._draw_eyelids(surface, eye_rect, params, cx, cy, w, h, r, mirror)

    def _draw_eyelids(self, surface: pygame.Surface, eye_rect: pygame.Rect,
                      params: EyeParams, cx: float, cy: float,
                      w: float, h: float, r: float, mirror: bool):
        """Draw top and bottom eyelids as background-colored overlays."""
        lid_color = self.BG_COLOR  # Lids match background to "close" the eye

        # Top eyelid
        if params.top_lid > 0.01:
            lid_height = h * params.top_lid

            # Extend lid size to ensure full coverage including glow
            lid_surf = pygame.Surface((int(w) + 20, int(h) + 20), pygame.SRCALPHA)
            lid_rect = pygame.Rect(0, 0, int(w) + 20, int(lid_height) + 20)
            pygame.draw.rect(lid_surf, (*lid_color, 255), lid_rect)

            # Rotate if there's an angle
            if abs(params.top_lid_angle) > 0.5:
                angle = -params.top_lid_angle if not mirror else params.top_lid_angle
                lid_surf = pygame.transform.rotate(lid_surf, angle)

            lid_pos = (eye_rect.left - 10, eye_rect.top - 10)
            surface.blit(lid_surf, lid_pos)

        # Bottom eyelid
        if params.bottom_lid > 0.01:
            lid_height = h * params.bottom_lid

            lid_surf = pygame.Surface((int(w) + 4, int(lid_height) + 4), pygame.SRCALPHA)
            lid_rect = pygame.Rect(0, 0, int(w) + 4, int(lid_height) + 4)
            pygame.draw.rect(lid_surf, (*lid_color, 255), lid_rect)

            lid_y = eye_rect.bottom - lid_height
            lid_pos = (eye_rect.left - 2, int(lid_y))
            surface.blit(lid_surf, lid_pos)

    # ─── PIL-only path (no pygame) for hardware displays ─────────────────

    def render_to_pil(self) -> Image.Image:
        """Render the same frame to a PIL Image (RGB). Use for hardware without pygame."""
        img = Image.new("RGB", (self.screen_width, self.screen_height), self.BG_COLOR)
        draw = ImageDraw.Draw(img)
        self._draw_eye_pil(draw, self.left_center, self.left_params, mirror=False)
        self._draw_eye_pil(draw, self.right_center, self.right_params, mirror=True)
        return img

    def _draw_eye_pil(self, draw: ImageDraw.ImageDraw,
                      center: Tuple[int, int], params: EyeParams, mirror: bool):
        """Draw one eye with PIL (same layout as _draw_eye)."""
        gaze_offset_x = params.gaze_x * self.GAZE_RANGE
        gaze_offset_y = params.gaze_y * self.GAZE_RANGE
        cx_raw = center[0] + params.x_offset + gaze_offset_x
        cy_raw = center[1] + params.y_offset + gaze_offset_y
        w, h = params.eye_width, params.eye_height
        r = min(params.corner_radius, w / 2, h / 2)

        panel_min_x = 0 if not mirror else (self.screen_width // 2)
        panel_max_x = (self.screen_width // 2) if not mirror else self.screen_width
        cx = max(panel_min_x + w / 2, min(panel_max_x - w / 2, cx_raw))
        cy = max(h / 2, min(self.screen_height - h / 2, cy_raw))

        left = int(cx - w / 2)
        top = int(cy - h / 2)
        right = int(left + w)
        bottom = int(top + h)
        eye_rect = (left, top, right, bottom)

        # Glow (solid rounded rects, no alpha)
        for i in range(self.GLOW_LAYERS, 0, -1):
            glow_expand = i * 8
            gl = max(0, left - glow_expand // 2)
            gt = max(0, top - glow_expand // 2)
            gr = min(self.screen_width, right + glow_expand // 2)
            gb = min(self.screen_height, bottom + glow_expand // 2)
            glow_color = (
                params.eye_color[0] // 4,
                params.eye_color[1] // 4,
                params.eye_color[2] // 4,
            )
            draw.rounded_rectangle([gl, gt, gr, gb], radius=int(r + glow_expand / 2), fill=glow_color)

        # Main eye
        draw.rounded_rectangle(eye_rect, radius=int(r), fill=params.eye_color)

        # Eyelids (background-colored rects that cover glow too)
        lid_color = self.BG_COLOR
        if params.top_lid > 0.01:
            lid_h = int(h * params.top_lid) + 30
            # Extend to cover glow layers (3 layers * 8px each = 24px extra)
            draw.rectangle(
                [left - 30, top - 30, right + 30, top + lid_h],
                fill=lid_color
            )
        if params.bottom_lid > 0.01:
            lid_h = int(h * params.bottom_lid) + 30
            draw.rectangle(
                [left - 30, bottom - lid_h, right + 30, bottom + 30],
                fill=lid_color
            )
