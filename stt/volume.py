"""
volume.py

DESCRIPTION:
this script is used to test the sensitivity of volume of haku's mic.
- it is used to decide when you should decide that you've stopped speaking 

INSTRUCTIONS:
1. run the script
2. stay quiet and note the numbers
3. talk and note the new numbers
-> use these numbers as your threshold, set something a little higher than the quiet_num and lower than the talk_num

regards,
Aiza
"""

import sys
import time
import numpy as np
import qi

from config import PEPPER_IP, PEPPER_PORT, SAMPLE_RATE, CHANNEL_MODE, DEINTERLEAVED


class VolumeMonitor:
    """
    A minimal NAOqi "service" object.

    Pepper's ALAudioDevice doesn't return audio to you when you ask for it —
    instead, YOU register as a service, and Pepper repeatedly CALLS INTO you
    (via processRemote) whenever new audio is available. That's why this is
    a class with a processRemote method, not just a function.
    """

    def __init__(self, app):
        # 'app' is the qi.Application — it holds the connection session
        self.session = app.session
        self.audio_service = self.session.service("ALAudioDevice")

    def start(self):
        """Tell Pepper to start sending us audio chunks."""
        # setClientPreferences(name, sample_rate, channels, deinterleaved)
        self.audio_service.setClientPreferences(
            "VolumeMonitor", SAMPLE_RATE, CHANNEL_MODE, DEINTERLEAVED
        )
        self.audio_service.subscribe("VolumeMonitor")

    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timestamp, buffer):
        """
        This method name and signature are REQUIRED by NAOqi — don't rename it.
        Pepper calls this automatically every time a new chunk of audio arrives.

        'buffer' is raw audio bytes. We convert it to numbers (int16 samples)
        so we can measure how "loud" this chunk is.
        """
        audio = np.frombuffer(buffer, dtype=np.int16)

        # A simple way to estimate loudness: average the absolute value of
        # all the samples in this chunk. Silence hovers near 0, speech spikes up.
        volume = np.abs(audio).mean()

        print(f"volume: {volume:.1f}")

    def stop(self):
        """Tell Pepper to stop sending us audio."""
        self.audio_service.unsubscribe("VolumeMonitor")


def main():
    # qi.Application (not just qi.Session) is required here because we need
    # to REGISTER a service that Pepper calls back into — a plain Session
    # can only call OUT to Pepper's services, not receive callbacks.
    app = qi.Application(sys.argv, url=f"tcp://{PEPPER_IP}:{PEPPER_PORT}")
    app.start()

    monitor = VolumeMonitor(app)
    app.session.registerService("VolumeMonitor", monitor)
    monitor.start()

    print("Watching volume levels. Stay quiet, then talk. Ctrl+C to stop.")
    try:
        app.run()  # blocks here forever; processRemote() fires in the background
    except KeyboardInterrupt:
        print("\nStopping...")
        monitor.stop()


if __name__ == "__main__":
    main()