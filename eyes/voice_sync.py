"""
Voice Sync — Maps audio amplitude to eye shape for speaking animation.

Level 1 "Subwoofer" method: eye height bounces with voice loudness.
Includes squash-and-stretch to preserve visual "mass".
"""

import math
import struct
from collections import deque
from typing import Optional


class VoiceSync:
    """
    Processes audio frames and outputs an amplitude value (0.0–1.0)
    that drives eye height during speech.
    """

    def __init__(self, sensitivity: float = 1.2, smoothing_frames: int = 4):
        """
        Args:
            sensitivity: Multiplier for amplitude → eye effect mapping.
            smoothing_frames: Number of frames to average for smooth output.
        """
        self.sensitivity = sensitivity
        self.smoothing_frames = smoothing_frames

        self._amplitude_buffer = deque(maxlen=smoothing_frames)
        self._current_amplitude = 0.0
        self._target_amplitude = 0.0

        # Decay: amplitude fades toward zero when no audio arrives
        self._decay_rate = 8.0   # Per second
        self._last_audio_time = 0.0

    @property
    def amplitude(self) -> float:
        """Current smoothed amplitude (0.0–1.0)."""
        return self._current_amplitude

    def process_audio_frame(self, audio_data: bytes, sample_rate: int = 48000,
                            sample_width: int = 2, channels: int = 1):
        """
        Process a raw PCM audio chunk and update amplitude.

        Args:
            audio_data: Raw PCM bytes (signed 16-bit little-endian by default)
            sample_rate: Audio sample rate
            sample_width: Bytes per sample (2 = 16-bit)
            channels: Number of channels (1 = mono)
        """
        if not audio_data:
            return

        rms = self._calculate_rms(audio_data, sample_width)

        # Normalize to 0.0–1.0 range (16-bit audio max is 32767)
        normalized = min(1.0, (rms / 8000.0) * self.sensitivity)

        self._amplitude_buffer.append(normalized)
        self._target_amplitude = sum(self._amplitude_buffer) / len(self._amplitude_buffer)

        import time
        self._last_audio_time = time.time()

    def _calculate_rms(self, audio_data: bytes, sample_width: int = 2) -> float:
        """Calculate Root Mean Square of audio samples."""
        if sample_width == 2:
            fmt = f"<{len(audio_data) // 2}h"  # Little-endian signed 16-bit
            try:
                samples = struct.unpack(fmt, audio_data)
            except struct.error:
                return 0.0
        else:
            return 0.0

        if not samples:
            return 0.0

        sum_squares = sum(s * s for s in samples)
        rms = math.sqrt(sum_squares / len(samples))
        return rms

    def update(self, dt: float):
        """
        Advance the voice sync state. Call every frame.
        Smoothly interpolates toward target and decays when silent.
        """
        import time
        now = time.time()

        # If no audio received recently, decay toward zero
        if now - self._last_audio_time > 0.1:
            self._target_amplitude *= max(0, 1.0 - self._decay_rate * dt)

        # Smooth interpolation toward target
        speed = 15.0 if self._target_amplitude > self._current_amplitude else 10.0
        self._current_amplitude += (self._target_amplitude - self._current_amplitude) * min(1.0, speed * dt)

        # Clamp
        self._current_amplitude = max(0.0, min(1.0, self._current_amplitude))

    def reset(self):
        """Reset amplitude to zero (e.g., when agent stops speaking)."""
        self._amplitude_buffer.clear()
        self._target_amplitude = 0.0
        self._current_amplitude = 0.0
