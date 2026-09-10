"""
main.py

PURPOSE:
This is the script you actually RUN. It proves the full loop works:

    Pepper hears you  -->  we transcribe it  -->  Pepper repeats it back

It ties together:
- config.py         (settings)
- live_capture.py   (streams audio from Pepper into memory)
- faster-whisper     (turns that audio into text)
- ALTextToSpeech     (Pepper speaks the transcript back to you)

Once this is working reliably, the next step (not in this file yet) is
to send the transcript to the `llm` module instead of straight to
tts.say(), so Pepper responds intelligently instead of just echoing you.

HOW TO RUN:
    python main.py
Then just talk — it will listen, detect when you've stopped, transcribe,
and speak the transcript back.
"""

import sys
import time
import qi

from config import (
    PEPPER_IP,
    PEPPER_PORT,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)
from audio_capture import AudioCapture


def main():
    # --- 1. Connect to Pepper ---
    # qi.Application (not qi.Session) is required because AudioCapture needs
    # to REGISTER itself as a service that Pepper calls back into.
    app = qi.Application(sys.argv, url=f"tcp://{PEPPER_IP}:{PEPPER_PORT}")
    app.start()
    session = app.session

    try:
        tts = session.service("ALTextToSpeech")
    except RuntimeError:
        print("Could not reach ALTextToSpeech — check Pepper's IP/connection.")
        sys.exit(1)

    # --- 2. Set up live audio capture ---
    capture = AudioCapture(app)
    session.registerService("AudioCapture", capture)

    print("Listening... speak now")
    capture.start()

    # This loop just waits until AudioCapture decides (via silence detection)
    # that you've stopped talking. capture.is_recording flips to False
    # automatically inside live_capture.py when that happens.
    while capture.is_recording:
        time.sleep(0.1)

    print("Stopped listening, transcribing...")

    # --- 3. Pull the recorded audio out of memory ---
    # This is a numpy array of floats — no .wav file was ever created,
    # nothing was written to disk on Pepper or on your machine.
    audio = capture.get_audio_float32()

    # --- 4. Transcribe with Whisper ---
    # faster-whisper is imported here (not at the top) so that this script
    # can still be read/understood even if faster-whisper isn't installed yet.
    from faster_whisper import WhisperModel

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )

    # transcribe() returns a generator of "segments" (chunks of recognized speech)
    segments, _ = model.transcribe(audio, language="en")
    transcript = " ".join(segment.text for segment in segments).strip()

    print(f"Heard: {transcript}")

    # --- 5. Prove it worked: have Pepper say back what it heard ---
    if transcript:
        tts.say(f"You said: {transcript}")
    else:
        tts.say("Sorry, I didn't catch anything.")


if __name__ == "__main__":
    main()