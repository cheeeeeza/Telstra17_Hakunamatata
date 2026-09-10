import time
import numpy as np
import sounddevice as sd

from config import SAMPLE_RATE, SILENCE_THRESHOLD, SILENCE_DURATION

buffer = []
silence_start = None
is_recording = True


def callback(indata, frames, time_info, status):
    """
    sounddevice calls this automatically with new audio chunks —
    same idea as Pepper's processRemote(), just a different source.
    """
    global silence_start, is_recording

    if not is_recording:
        return

    # indata comes in as float32 already (range -1.0 to 1.0), so no
    # conversion needed here, unlike Pepper's int16 raw bytes.
    audio_chunk = indata[:, 0].copy()  # take first channel only
    buffer.append(audio_chunk)

    # Use the same silence-detection approach as live_capture.py,
    # scaled to match float32 range instead of int16.
    volume = np.abs(audio_chunk).mean() * 32768  # rescale for comparable numbers
    print(f"volume: {volume:.1f}")

    if volume < SILENCE_THRESHOLD:
        if silence_start is None:
            silence_start = time.time()
        elif time.time() - silence_start > SILENCE_DURATION:
            is_recording = False
    else:
        silence_start = None


def main():
    global is_recording

    print("Listening on your laptop mic... speak now")

    # Open a live input stream from the laptop's default microphone
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=callback
    ):
        while is_recording:
            time.sleep(0.1)

    print("Stopped listening, transcribing...")

    # Combine all recorded chunks into one array, same as get_audio_float32()
    # in live_capture.py
    full_recording = np.concatenate(buffer) if buffer else np.array([], dtype=np.float32)

    from faster_whisper import WhisperModel
    from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

    model = WhisperModel(
        WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
    )
    segments, _ = model.transcribe(full_recording, language="en")
    transcript = " ".join(segment.text for segment in segments).strip()

    print(f"Heard: {transcript}")


if __name__ == "__main__":
    main()