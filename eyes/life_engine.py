"""
Life Engine — Autonomous micro-behaviors that make eyes feel alive.

Runs independently of the FSM state, adding saccades, micro-drift,
breathing rhythm, natural blinks, and squint variation on every frame.
"""

import math
import time
import random
from dataclasses import dataclass, field


@dataclass
class EyeOffsets:
    """Offsets for a single eye."""
    gaze_dx: float = 0.0
    gaze_dy: float = 0.0
    y_offset: float = 0.0
    x_offset: float = 0.0
    lid_offset: float = 0.0

@dataclass
class LifeOffsets:
    """Additive offsets for both eyes, decoupled."""
    left: EyeOffsets = field(default_factory=EyeOffsets)
    right: EyeOffsets = field(default_factory=EyeOffsets)
    blink_amount: float = 0.0   # 0.0 = open, 1.0 = fully closed


class LifeEngine:
    """
    Life Engine — Autonomous micro-behaviors that make eyes feel alive.
    Generates independent offsets for each eye to prevent simple mirroring.
    """

    def __init__(self):
        self._enabled = True
        self._start_time = time.time()

        # ── Saccades: Vector-style conjugate gaze (both eyes look same direction) ──
        self._saccade_target_x = 0.0
        self._saccade_target_y = 0.0
        self._saccade_current_x = 0.0
        self._saccade_current_y = 0.0
        self._saccade_velocity_x = 0.0
        self._saccade_velocity_y = 0.0
        self._next_saccade_time = time.time() + random.uniform(0.5, 2.0)
        self._saccade_spring = 0.28
        self._saccade_damping = 0.72

        # ── Micro-drift: shared so both eyes drift together (Vector “looking around”) ──
        self._drift_phase_x = random.uniform(0, math.tau)
        self._drift_phase_y = random.uniform(0, math.tau)
        self._drift_speed_x = 0.9
        self._drift_speed_y = 0.7

        # ── Breathing (Shared rhythm) ──
        self._breath_phase = 0.0
        self._breath_speed = 0.8  # Brpm ~1.0
        
        # ── Blink state: physics-based (DROPPING/SQUASHING/JUMPING like emotion_tuner) ──
        self._blink_state = "IDLE"
        self._blink_velocity = 0.0
        self._blink_amount = 0.0
        self._blink_progress = 0.0
        self._blink_speed_mult = 1.0
        self._next_blink_time = time.time() + random.uniform(1.5, 4.0)
        self._blink_drop_time = 0.0

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def update(self, dt: float) -> LifeOffsets:
        """Advance all life behaviors by dt seconds, return combined offsets."""
        offsets = LifeOffsets()
        if not self._enabled:
            return offsets

        self._update_saccades(dt, offsets)
        self._update_drift(dt, offsets)
        self._update_breathing(dt, offsets)
        self._update_blink(dt, offsets)
        self._update_squint(dt, offsets)

        return offsets

    # ── Saccades: Vector-style conjugate gaze (both eyes move together) ─────

    def _update_saccades(self, dt: float, offsets: LifeOffsets):
        now = time.time()

        if now >= self._next_saccade_time:
            self._saccade_target_x = random.uniform(-0.22, 0.22)
            self._saccade_target_y = random.uniform(-0.14, 0.14)
            self._next_saccade_time = now + random.uniform(0.6, 2.8)

        self._saccade_velocity_x += (self._saccade_target_x - self._saccade_current_x) * self._saccade_spring
        self._saccade_velocity_x *= self._saccade_damping
        self._saccade_current_x += self._saccade_velocity_x

        self._saccade_velocity_y += (self._saccade_target_y - self._saccade_current_y) * self._saccade_spring
        self._saccade_velocity_y *= self._saccade_damping
        self._saccade_current_y += self._saccade_velocity_y

        # Conjugate gaze: both eyes look the same way (Vector-style)
        offsets.left.gaze_dx = self._saccade_current_x
        offsets.left.gaze_dy = self._saccade_current_y
        offsets.right.gaze_dx = self._saccade_current_x
        offsets.right.gaze_dy = self._saccade_current_y

    # ── Micro-drift: shared drift so both eyes wander together ───────────────

    def _update_drift(self, dt: float, offsets: LifeOffsets):
        t = time.time() - self._start_time
        dx = (math.sin(t * self._drift_speed_x + self._drift_phase_x) * 0.018 +
              math.sin(t * 2.2 + 1.5) * 0.008)
        dy = (math.sin(t * self._drift_speed_y + self._drift_phase_y) * 0.012 +
              math.sin(t * 1.8 + 2.1) * 0.006)
        offsets.left.gaze_dx += dx
        offsets.left.gaze_dy += dy
        offsets.right.gaze_dx += dx
        offsets.right.gaze_dy += dy

    # ── Breathing: subtle vertical oscillation ──────────────────────────

    def _update_breathing(self, dt: float, offsets: LifeOffsets):
        self._breath_phase += self._breath_speed * dt
        # Calmer breathing (1.0px bob instead of 1.5px)
        breath_y = math.sin(self._breath_phase) * 1.0
        offsets.left.y_offset += breath_y
        offsets.right.y_offset += breath_y

    # ── Blinks: Physics-based bouncy motion (DROPPING/SQUASHING/JUMPING) ─────

    def _update_blink(self, dt: float, offsets: LifeOffsets):
        """Physics-based blink: clean drop, squash, and bouncy recovery."""
        now = time.time()

        if self._blink_state == "IDLE":
            if now >= self._next_blink_time:
                self._blink_state = "DROPPING"
                self._blink_velocity = 0.0
                self._blink_progress = 0.0
                self._blink_speed_mult = random.uniform(1.5, 2.5)
                self._blink_drop_time = now

        elif self._blink_state == "DROPPING":
            # Eye accelerates downward (closed)
            self._blink_velocity += 10 * self._blink_speed_mult
            self._blink_progress += self._blink_velocity

            # Reach maximum closure (1.0)
            if self._blink_progress >= 100:
                self._blink_progress = 100
                self._blink_state = "SQUASHING"

        elif self._blink_state == "SQUASHING":
            # Hold closed for a moment
            if now - self._blink_drop_time > 0.02:  # ~20ms squashed
                self._blink_state = "JUMPING"
                self._blink_velocity = 0.0

        elif self._blink_state == "JUMPING":
            # Bouncy recovery back to open
            recovery_speed = max(0.1, min(0.9, 0.7 * self._blink_speed_mult))
            self._blink_progress += (0 - self._blink_progress) * recovery_speed

            if abs(self._blink_progress) < 2:
                self._blink_progress = 0.0
                self._blink_state = "IDLE"
                self._next_blink_time = now + random.uniform(1.2, 3.5)

        # Normalize blink_amount to 0-1 range (0=open, 1=closed)
        amount = max(0.0, min(1.0, self._blink_progress / 100.0))
        offsets.blink_amount = amount
        offsets.left.lid_offset = amount
        offsets.right.lid_offset = amount

    def force_blink(self):
        """Force an immediate full blink (e.g., on startle)."""
        self._blink_state = "DROPPING"
        self._blink_velocity = 0.0
        self._blink_progress = 0.0
        self._blink_speed_mult = 2.0
        self._blink_drop_time = time.time()

    # ── Squint variation: tiny eyelid adjustments ───────────────────────

    def _update_squint(self, dt: float, offsets: LifeOffsets):
        """Same squint for both eyes so lid movement is always simultaneous."""
        t = time.time() - self._start_time
        squint = math.sin(t * 0.5) * 0.015 + math.sin(t * 2.1) * 0.005
        offsets.left.lid_offset += squint
        offsets.right.lid_offset += squint
