import pygame
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import threading

class AudioController:
    def __init__(self, freq, buff):
        pygame.mixer.init(frequency=freq, buffer=buff)
        self.kick_sample = pygame.mixer.Sound('Audio/Samples/Kick.wav')
        self.clap_sample = pygame.mixer.Sound('Audio/Samples/Clap.wav')
        self.recording = False
        self.recorded_audio = []
        self.sample_rate = freq
        self.stream = None
        self.lock = threading.Lock()

    def play_kick(self):
        self._start_recording()
        self.kick_sample.play()

    def play_clap(self):
        self._start_recording()
        self.clap_sample.play()

    def _start_recording(self):
        if not self.recording:
            self.recorded_audio.clear()
            self.recording = True
            self.stream = sd.InputStream(samplerate=self.sample_rate,
                                         channels=2,
                                         callback=self._callback)
            self.stream.start()
            print("Audio stream started.")

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Stream status: {status}")
        with self.lock:
            self.recorded_audio.append(indata.copy())

    def stop_recording_and_save(self, silence_duration=5):
        threading.Timer(silence_duration, self._stop_recording).start()

    def _stop_recording(self):
        if self.recording:
            print("Stopping audio stream and saving.")
            self.stream.stop()
            self.stream.close()
            self.recording = False
            with self.lock:
                final_audio = np.concatenate(self.recorded_audio, axis=0)
            write('output.wav', self.sample_rate, final_audio)
            print("Audio saved to output.wav")
