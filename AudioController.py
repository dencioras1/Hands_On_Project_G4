import pygame
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading

class AudioController:
    def __init__(self, freq, buff):
        pygame.mixer.init(frequency=freq, buffer=buff)
        self.kick_sample = pygame.mixer.Sound('assets/Samples/Kick.wav')
        self.clap_sample = pygame.mixer.Sound('assets/Samples/Clap.wav')
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
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=2,
                callback=self._callback,
                device=10  # Change this if needed
            )

            print("Audio stream started.")
            self.stream.start()

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Stream status: {status}")
        
        with self.lock:
            # Append a copy of stereo data
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

                # Convert stereo to mono by averaging channels
                if final_audio.ndim == 2 and final_audio.shape[1] == 2:
                    final_audio = np.mean(final_audio, axis=1)

                # final_audio = self.normalize_audio(final_audio)

            sf.write('output.wav', final_audio, self.sample_rate, subtype='FLOAT')
            print("Audio saved to output.wav")

    def normalize_audio(self, audio):
        """Normalize audio to the range of -1.0 to 1.0."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
