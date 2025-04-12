import os

import keras
import pygame, sys

from Animation import Animation
from GenreClassifier import GenreClassifier
from assets.button import Button
moving_sprites = pygame.sprite.Group()
animation = Animation(640, 260, "assets/Animations/loading")
moving_sprites.add(animation)
clock = pygame.time.Clock()
genre_classification = GenreClassifier()
local_path = "saved_models/model1.keras"
pygame.mixer.init()
pygame.init()


class Game:

    def __init__(self, genre):
        self.genre = genre
        self.classification = None
        self.SCREEN = None
        self.current_genre_number = 0
        self.current_genre = None
        self.base_path = "Audio"

    def get_font(self, size):  # Returns Press-Start-2P in the desired size
        return pygame.font.Font("assets/Courier_New.ttf", size)

    def get_model(self):
        return keras.models.load_model(local_path)

    def start_game(self, SCREEN):
        # start at the first genre (0)
        self.current_genre = genre_classification.labels[self.current_genre_number]
        genre_path = os.path.join(self.base_path, self.current_genre)

        first_wav = None
        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                first_wav = os.path.join(genre_path, file)
                break  # Stop after the first .wav file

        pygame.mixer.music.load(first_wav)
        pygame.mixer.music.play()
        self.SCREEN = SCREEN

        # wait length of file
        while pygame.mixer.music.get_busy():
            SCREEN.fill("black")
            animation.is_animating = True
            PLAY_TEXT = pygame.font.Font("assets/Courier_New.ttf", 45).render(f"This is the genre: {self.current_genre}", True,
                                                                              "White")
            PLAY_RECT = PLAY_TEXT.get_rect(center=(640, 160))
            SCREEN.blit(PLAY_TEXT, PLAY_RECT)

            moving_sprites.draw(SCREEN)
            moving_sprites.update()
            pygame.display.flip()




        animation.is_animating = False
        SCREEN.fill("black")
        self.current_genre_number += 1



    def classification(self, file_path):
        model = self.get_model()
        predicted_genre = genre_classification.predict_file(model=model, file_path=file_path)

    def update_game(self, SCREEN):
        key = pygame.key.get_pressed()

        PLAY_TEXT = pygame.font.Font("assets/Courier_New.ttf", 60).render(f"Your turn to try: {self.current_genre}",
                                                                          True,
                                                                          "White")
        PLAY_RECT = PLAY_TEXT.get_rect(center=(640, 160))
        SCREEN.blit(PLAY_TEXT, PLAY_RECT)

        # get input from TSP

        # classify it


        PLAY_BACK = Button(image=None, pos=(320, 650),
                           text_input="MENU", font=self.get_font(60), base_color="White", hovering_color="#74b8ab")
        QUIT = Button(image=None, pos=(960, 650),
                           text_input="QUIT", font=self.get_font(60), base_color="White", hovering_color="#74b8ab")

        PLAY_BACK.update(SCREEN)
        QUIT.update(SCREEN)

        if key[pygame.K_i]:
            pygame.quit()
            sys.exit()





