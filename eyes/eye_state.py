"""
Eye State Machine — FSM that maps agent events to eye emotions.

States: IDLE, LISTENING, PROCESSING, TALKING, CONFUSED, STARTLED, HAPPY
Transitions are triggered by LiveKit agent events.
"""

import time
from enum import Enum, auto
from typing import Optional, Callable


class EyeState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    TALKING = auto()
    HAPPY = auto()
    CONFUSED = auto()
    STARTLED = auto()


# Map states to emotion preset names in eye_renderer
STATE_TO_EMOTION = {
    EyeState.IDLE: "idle",
    EyeState.LISTENING: "listening",
    EyeState.PROCESSING: "processing",
    EyeState.TALKING: "talking",
    EyeState.HAPPY: "happy",
    EyeState.CONFUSED: "confused",
    EyeState.STARTLED: "startled",
}

# Reaction latency: delay before eyes react (simulates cognition)
REACTION_DELAY = {
    EyeState.IDLE: 0.0,
    EyeState.LISTENING: 0.05,        # Very fast — we noticed you
    EyeState.PROCESSING: 0.2,        # Natural ~200ms human reaction
    EyeState.TALKING: 0.1,
    EyeState.HAPPY: 0.15,
    EyeState.CONFUSED: 0.25,
    EyeState.STARTLED: 0.0,          # Instant — reflex
}

# Auto-return states: some states automatically return to another
AUTO_RETURN = {
    EyeState.STARTLED: (EyeState.IDLE, 0.6),    # Return to idle after 0.6s
    EyeState.HAPPY: (EyeState.IDLE, 3.0),        # Return to idle after 3s
}


class EyeStateMachine:
    """
    Controls which emotion the eyes are displaying based on agent events.
    Supports reaction latency and auto-return timers.
    """

    def __init__(self, on_state_change: Optional[Callable[..., None]] = None):
        self.current_state = EyeState.IDLE
        self.on_state_change = on_state_change

        # Pending transition (for reaction latency)
        self._pending_state: Optional[EyeState] = None
        self._pending_time: float = 0.0

        # Auto-return timer
        self._auto_return_state: Optional[EyeState] = None
        self._auto_return_time: float = 0.0

        # Startle vibrate timer
        self._startle_end_time: float = 0.0

    @property
    def is_startled(self) -> bool:
        return self.current_state == EyeState.STARTLED and time.time() < self._startle_end_time

    def transition_to(self, new_state: EyeState):
        """Request a state transition. May be delayed by reaction latency."""
        if new_state == self.current_state:
            return

        delay = REACTION_DELAY.get(new_state, 0.0)
        if delay > 0:
            self._pending_state = new_state
            self._pending_time = time.time() + delay
        else:
            self._apply_transition(new_state)

    def _apply_transition(self, new_state: EyeState):
        """Immediately switch to new state."""
        self.current_state = new_state
        self._pending_state = None

        # Set auto-return if applicable
        if new_state in AUTO_RETURN:
            return_state, duration = AUTO_RETURN[new_state]
            self._auto_return_state = return_state
            self._auto_return_time = time.time() + duration
        else:
            self._auto_return_state = None

        # Special handling
        if new_state == EyeState.STARTLED:
            self._startle_end_time = time.time() + 0.4  # Vibrate for 400ms

        # Notify renderer
        emotion_name = STATE_TO_EMOTION.get(new_state, "idle")
        bypass_inertia = new_state == EyeState.STARTLED
        if self.on_state_change:
            try:
                self.on_state_change(emotion_name, bypass_inertia)
            except TypeError:
                self.on_state_change(emotion_name)

    def update(self, dt: float):
        """Check pending transitions and auto-returns."""
        now = time.time()

        # Check pending reaction-delayed transition
        if self._pending_state is not None and now >= self._pending_time:
            self._apply_transition(self._pending_state)

        # Check auto-return
        if self._auto_return_state is not None and now >= self._auto_return_time:
            self._apply_transition(self._auto_return_state)

    # ── Convenience methods for agent events ────────────────────────────

    def on_user_started_speaking(self):
        """VAD detected speech start."""
        self.transition_to(EyeState.LISTENING)

    def on_user_stopped_speaking(self):
        """VAD detected speech end — agent is now processing."""
        self.transition_to(EyeState.PROCESSING)

    def on_agent_started_speaking(self):
        """TTS started playing."""
        self.transition_to(EyeState.TALKING)

    def on_agent_stopped_speaking(self):
        """TTS finished playing."""
        self.transition_to(EyeState.IDLE)

    def on_agent_thinking(self):
        """LLM is generating a response."""
        self.transition_to(EyeState.PROCESSING)

    def on_happy(self):
        """Recognition / greeting event."""
        self.transition_to(EyeState.HAPPY)

    def on_confused(self):
        """Unrecognized command or error."""
        self.transition_to(EyeState.CONFUSED)

    def on_startle(self):
        """Sudden loud noise or unexpected event."""
        self.transition_to(EyeState.STARTLED)
