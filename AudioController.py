import pygame

class AudioController:

    def __init__(self, freq, buff):
        pygame.mixer.init(frequency=freq, buffer=buff)
        self.kick_sample = pygame.mixer.Sound('Audio/Samples/Kick.wav')
        self.clap_sample = pygame.mixer.Sound('Audio/Samples/Clap.wav')

    def play_kick(self):
        self.kick_sample.play()

    def play_clap(self):
        self.clap_sample.play()

    def generate_waveform(self):
        pass

