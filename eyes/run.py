#!/usr/bin/env python3
"""
Entry Point — Starts PyGame eyes display + LiveKit agent.

Usage:
    python run.py --demo [--hardware]   # Standalone eyes demo
    python run.py start [--hardware]    # Eyes + LiveKit agent
"""

import sys
import threading
import argparse

from display import EyeState  # Needed for demo state transition


def main():
    parser = argparse.ArgumentParser(
        description="Procedural Eyes + LiveKit Agent",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run standalone eyes demo without LiveKit",
    )
    parser.add_argument(
        "--hardware", action="store_true",
        help="Use two ST7735 TFT panels instead of PyGame window (required for TFT)",
    )

    args, remaining = parser.parse_known_args()

    if args.demo:
        run_demo(use_hardware=args.hardware)
    else:
        run_with_agent(remaining, use_hardware=args.hardware)


def get_display_class(use_hardware: bool):
    """Factory to get the correct display class."""
    if use_hardware:
        from display import HardwareEyeDisplay
        print("📺 Using Hardware Display (ST7735)")
        return HardwareEyeDisplay
    else:
        from display import PygameEyeDisplay
        print("🖥️  Using Desktop PyGame Display")
        print("   💡 To use your two TFT panels, run with:  python run.py start --hardware")
        return PygameEyeDisplay


def run_demo(use_hardware: bool = False):
    """Standalone eyes demo — no LiveKit needed."""
    DisplayClass = get_display_class(use_hardware)
    display = DisplayClass()
    
    print("🎮 Procedural Eyes Demo")
    print("━━━━━━━━━━━━━━━━━━━━━━━━")
    
    display.fsm.transition_to(EyeState.IDLE)
    display.run_loop()


def run_with_agent(livekit_args: list, use_hardware: bool = False):
    """
    Start LiveKit agent on main thread, Display in background thread.
    """
    from agent import set_eye_display, run_agent
    
    print("👁  Starting Procedural Eyes + LiveKit Agent")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Create display
    DisplayClass = get_display_class(use_hardware)
    display = DisplayClass()

    # Register with agent module
    set_eye_display(display)

    # Start display in background thread
    def run_display():
        # run_loop handles its own initialization (window or SPI)
        display.run_loop()

    display_thread = threading.Thread(target=run_display, daemon=True)
    display_thread.start()
    print("✅ Display thread started")

    # Run LiveKit agent on main thread (needs main thread for signals)
    # Patch sys.argv so the LiveKit CLI picks up the right args
    sys.argv = [sys.argv[0]] + (livekit_args if livekit_args else ["start"])
    run_agent()

    # If agent exits, stop display too
    display.stop()
    print("👋 Agent stopped. Exiting.")


if __name__ == "__main__":
    main()
