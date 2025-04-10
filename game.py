import pygame, sys
from GenreClassifier import GenreClassifier
from assets.button import Button


class Game:

    def __init__(self, genre):
        self.genre = genre

    def update_game(self, SCREEN):

        SCREEN.fill("black")


        PLAY_TEXT = pygame.font.Font("assets/Courier_New.ttf", 45).render(f"This is the genre: {self.genre}", True, "White")
        PLAY_RECT = PLAY_TEXT.get_rect(center=(640, 260))
        SCREEN.blit(PLAY_TEXT, PLAY_RECT)




