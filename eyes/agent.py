"""
LiveKit Agent — Groq-powered voice assistant with procedural eye integration.

Hooks into LiveKit agent events (VAD, TTS, LLM) and controls the
eye state machine + voice sync in real-time.
"""

import os
import asyncio
import threading
import struct
from pathlib import Path
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, AgentSession, Agent
from livekit.agents.llm import function_tool
from livekit.plugins import openai, deepgram, silero

# Load environment
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

# Debug: verify credentials
print(f"🔑 LiveKit URL: {os.getenv('LIVEKIT_URL')}")
print(f"🔑 Groq API Key: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing'}")


# ─── Tools ──────────────────────────────────────────────────────────────

@function_tool
async def calculate(operation: str, a: float, b: float) -> str:
    """
    Perform a mathematical calculation.

    Args:
        operation: The operation to perform: 'add', 'subtract', 'multiply', or 'divide'
        a: First number
        b: Second number
    """
    if operation == "add":
        result = a + b
        return f"{a} + {b} = {result}"
    elif operation == "subtract":
        result = a - b
        return f"{a} - {b} = {result}"
    elif operation == "multiply":
        result = a * b
        return f"{a} × {b} = {result}"
    elif operation == "divide":
        if b == 0:
            return "Error: Cannot divide by zero"
        result = a / b
        return f"{a} ÷ {b} = {result}"
    else:
        return f"Unknown operation: {operation}"


@function_tool
async def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert temperature between Celsius, Fahrenheit, and Kelvin.

    Args:
        value: The temperature value to convert
        from_unit: The source unit: 'celsius', 'fahrenheit', or 'kelvin'
        to_unit: The target unit: 'celsius', 'fahrenheit', or 'kelvin'
    """
    if from_unit.lower() == "celsius":
        celsius = value
    elif from_unit.lower() == "fahrenheit":
        celsius = (value - 32) * 5 / 9
    elif from_unit.lower() == "kelvin":
        celsius = value - 273.15
    else:
        return f"Unknown unit: {from_unit}"

    if to_unit.lower() == "celsius":
        result = celsius
    elif to_unit.lower() == "fahrenheit":
        result = celsius * 9 / 5 + 32
    elif to_unit.lower() == "kelvin":
        result = celsius + 273.15
    else:
        return f"Unknown unit: {to_unit}"

    return f"{value}° {from_unit} = {result:.2f}° {to_unit}"


@function_tool
async def convert_text_case(text: str, case_type: str) -> str:
    """
    Convert text to different cases.

    Args:
        text: The text to convert
        case_type: Type of case: 'upper', 'lower', 'title', or 'reverse'
    """
    if case_type.lower() == "upper":
        return f"Uppercase: {text.upper()}"
    elif case_type.lower() == "lower":
        return f"Lowercase: {text.lower()}"
    elif case_type.lower() == "title":
        return f"Title case: {text.title()}"
    elif case_type.lower() == "reverse":
        return f"Reversed: {text[::-1]}"
    else:
        return f"Unknown case type: {case_type}"


@function_tool
async def get_time_info(format_type: str = "full") -> str:
    """
    Get current time and date information.

    Args:
        format_type: Type of time info: 'time', 'date', 'full', or 'short'
    """
    from datetime import datetime
    now = datetime.now()

    if format_type.lower() == "time":
        return f"The time is {now.strftime('%I:%M %p')}"
    elif format_type.lower() == "date":
        return f"Today is {now.strftime('%A, %B %d, %Y')}"
    elif format_type.lower() == "full":
        return now.strftime("It's %I:%M %p on %A, %B %d, %Y")
    elif format_type.lower() == "short":
        return now.strftime("%I:%M %p, %m/%d/%Y")
    else:
        return now.strftime("It's %I:%M %p on %A, %B %d, %Y")


# ─── Global eye display reference ──────────────────────────────────────
# Set by run.py before starting the agent
_eye_display = None


def set_eye_display(display):
    """Register the EyeDisplay instance for agent event hooks."""
    global _eye_display
    _eye_display = display


# ─── Agent Entrypoint ───────────────────────────────────────────────────

async def entrypoint(ctx: JobContext):
    print(f"🔌 Connecting to room: {ctx.room.name}")

    groq_llm = openai.LLM(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
    )

    session = AgentSession(
        stt=deepgram.STT(),
        tts=deepgram.TTS(model="aura-luna-en"),
        vad=silero.VAD.load(),
        llm=groq_llm,
    )

    agent = Agent(
        instructions="""You are a helpful, friendly voice assistant with calculator, temperature, text, and time tools.
        Keep your responses short and conversational.
        Use the appropriate tool based on what users ask for:
        - Calculator for math
        - Temperature converter for temperature conversions
        - Text converter for text manipulation
        - Time info for current time/date
        Be warm and engaging.""",
        tools=[calculate, convert_temperature, convert_text_case, get_time_info],
    )

    # ── Wire up eye events (use LiveKit session state events) ─────────

    if _eye_display:
        fsm = _eye_display.fsm

        @session.on("user_state_changed")
        def on_user_state(event):
            if event.new_state == "speaking":
                fsm.on_user_started_speaking()   # Eyes → LISTENING
            elif event.new_state == "listening":
                fsm.on_user_stopped_speaking()   # User done → PROCESSING next

        @session.on("agent_state_changed")
        def on_agent_state(event):
            if event.new_state == "thinking":
                fsm.on_agent_thinking()          # Eyes → PROCESSING
            elif event.new_state == "speaking":
                fsm.on_agent_started_speaking()  # Eyes → TALKING
            elif event.new_state in ("idle", "listening"):
                fsm.on_agent_stopped_speaking() # Eyes → IDLE

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant):
        print(f"👤 Participant connected: {participant.identity}")
        if _eye_display:
            _eye_display.fsm.on_happy()

    await session.start(room=ctx.room, agent=agent)

    print("✅ Agent with procedural eyes is live!")
    print("🧮 Tools: calculate, convert_temperature, convert_text_case, get_time_info")

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)


def run_agent():
    """Start the LiveKit agent (called from a background thread)."""
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="campus-greeting-agent",
        )
    )
