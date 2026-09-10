"""
config.py

Central place for settings that get reused across the STT scripts
(debug_volume_monitor.py, live_capture.py, main.py).

Why this file exists:
- Pepper's IP address was previously hardcoded in every script.
- If you change robots, networks, or mic settings, you'd have to edit
  three files instead of one. This file fixes that.
"""

# --- Network settings ---
PEPPER_IP = "192.168.1.100"   # Replace with your Pepper's actual IP
PEPPER_PORT = 9559            # Default NAOqi port, rarely needs changing

# --- Audio capture settings ---
SAMPLE_RATE = 16000            # 16kHz is standard for speech recognition models like Whisper
CHANNEL_MODE = 3               # ALAudioDevice channel arg: 3 = front mic only (see NAOqi docs)
DEINTERLEAVED = 0              # 0 = not deinterleaved, matches ALAudioDevice defaults

# --- Silence detection settings ---
# These control when the robot decides you've stopped talking and
# automatically stops recording. Tune these using debug_volume_monitor.py
# BEFORE trusting them in the real capture script.
SILENCE_THRESHOLD = 500        # Volume level below which audio counts as "silence"
SILENCE_DURATION = 1.5         # Seconds of continuous silence before auto-stop

# --- Whisper (speech-to-text model) settings ---
WHISPER_MODEL_SIZE = "small"   # Options: tiny, base, small, medium, large — bigger = more accurate but slower
WHISPER_DEVICE = "cpu"         # Change to "cuda" if you have an NVIDIA GPU available
WHISPER_COMPUTE_TYPE = "int8"  # int8 = faster/lighter, good default for CPU