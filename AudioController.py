import pygame
import sounddevice as sd
import soundfile as sf  # For saving audio as high-quality float32 WAV
import numpy as np
import threading

class AudioController:
    def __init__(self, freq, buff):
        pygame.mixer.init(frequency=freq, buffer=buff)
        pygame.mixer.music.set_volume(0)
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
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=2,
                callback=self._callback,
                device=10
            )
            self.stream.start()
            print("Audio stream started.")

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Stream status: {status}")
        
        with self.lock:
            # Ensure that we are appending the correct shape for mono audio
            self.recorded_audio.append(indata.copy())  # Append the 1D indata for mono audio


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
                final_audio = self.trim_silence(final_audio)  # Trim the silence
                
                # Normalize audio to -1.0 to 1.0
                final_audio = self.normalize_audio(final_audio)

            # Save using Soundfile (float32)
            sf.write('output.wav', final_audio, self.sample_rate, subtype='FLOAT')
            print("Audio saved to output.wav")

    def trim_silence(self, audio, threshold=0.01):
        """Trim silence from the audio based on energy threshold."""
        energy = np.abs(audio)
        mask = energy > threshold

        # Check if there's any non-silent audio
        indices = np.where(mask)[0]
        if len(indices) == 0:
            print("No audio above threshold, returning original audio.")
            return audio  # No audio above threshold, return the original
        
        # If there is valid audio, trim it
        trimmed_audio = audio[indices[0]:indices[-1]+1]
        return trimmed_audio



    def normalize_audio(self, audio):
        """Normalize audio to the range of -1.0 to 1.0."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
