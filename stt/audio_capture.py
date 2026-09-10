"""
live_capture.py

PURPOSE:
Defines the AudioCapture class — the reusable piece that streams live
audio from Pepper's microphone straight into memory (no .wav file, no
scp file transfer needed).

This file does NOT connect to Pepper or run anything by itself — it's
meant to be imported by main.py (and later, by the llm module, once
speech-to-text output starts feeding an AI model instead of just being
repeated back by Pepper).

HOW IT WORKS (high level):
1. We register this class as a NAOqi "service" using Pepper's ALAudioDevice.
2. Pepper repeatedly calls our processRemote() method with small chunks
   of raw audio as you speak.
3. We store each chunk in a list (self.buffer).
4. We measure the volume of each chunk. If it stays quiet for long enough
   (SILENCE_DURATION seconds below SILENCE_THRESHOLD), we assume you've
   stopped talking and automatically stop recording.
5. Once stopped, get_audio_float32() converts all the buffered chunks into
   one continuous array in the format Whisper expects.
"""

import time
import numpy as np

from config import (
    SAMPLE_RATE,
    CHANNEL_MODE,
    DEINTERLEAVED,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
)


class AudioCapture:
    def __init__(self, app):
        """
        'app' is a qi.Application instance (created in main.py).
        We pull the audio service out of its session here so we don't
        have to pass the session around separately everywhere.
        """
        self.app = app
        self.session = app.session
        self.audio_service = self.session.service("ALAudioDevice")

        self.buffer = []            # list of raw audio chunks (numpy arrays)
        self.is_recording = False
        self.silence_start = None   # timestamp of when silence began, or None

    def start(self):
        """Begin listening: reset state and subscribe to Pepper's mic stream."""
        self.buffer = []
        self.is_recording = True
        self.silence_start = None

        # setClientPreferences(name, sample_rate, channels, deinterleaved)
        # This tells Pepper how we want the audio formatted before it's sent to us.
        self.audio_service.setClientPreferences(
            "AudioCapture", SAMPLE_RATE, CHANNEL_MODE, DEINTERLEAVED
        )
        self.audio_service.subscribe("AudioCapture")

    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timestamp, buffer):
        """
        REQUIRED method name/signature — NAOqi calls this automatically
        whenever a new chunk of audio is ready. Don't rename it.
        """
        if not self.is_recording:
            # Ignore any stray callbacks that arrive after we've already stopped
            return

        # Convert raw bytes into actual numbers we can work with
        audio = np.frombuffer(buffer, dtype=np.int16)
        self.buffer.append(audio.copy())

        # --- Silence detection ---
        # Measure how "loud" this chunk is (same technique as debug_volume_monitor.py)
        volume = np.abs(audio).mean()

        if volume < SILENCE_THRESHOLD:
            # This chunk is quiet. Start (or continue) a silence timer.
            if self.silence_start is None:
                self.silence_start = time.time()
            elif time.time() - self.silence_start > SILENCE_DURATION:
                # We've been quiet for long enough — assume the person is done talking
                self.stop()
        else:
            # Any sound above the threshold resets the silence timer
            self.silence_start = None

    def stop(self):
        """Stop listening and unsubscribe from Pepper's mic stream."""
        if not self.is_recording:
            return  # already stopped, nothing to do
        self.is_recording = False
        self.audio_service.unsubscribe("AudioCapture")

    def get_audio_float32(self):
        """
        Combine all the recorded chunks into a single array, converted to
        the float32 format (values between -1.0 and 1.0) that Whisper expects.

        Raw mic audio comes in as int16 (-32768 to 32767), so we divide by
        32768.0 to rescale it into that -1.0 to 1.0 range.
        """
        if not self.buffer:
            return np.array([], dtype=np.float32)

        full_recording = np.concatenate(self.buffer)
        return full_recording.astype(np.float32) / 32768.0