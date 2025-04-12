import keras
import numpy as np
import pygame, sys
from matplotlib import pyplot as plt

from game import Game
from assets.button import Button
from GenreClassifier import GenreClassifier

pygame.init()

SCREEN = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("Menu")
BG = pygame.image.load("assets/glow.jpg")
game = Game(None)
clock = pygame.time.Clock()
genre_classifier = GenreClassifier()
local_path = "/Users/kveen/Documents/GitHub/Hands_On_Project_G4/saved_models/model1.keras"
model = None



def get_font(size):  # Returns Press-Start-2P in the desired size
    return pygame.font.Font("assets/Courier_New.ttf", size)


def get_title_font(size):
    return pygame.font.Font("assets/shagade.ttf", size)


def play():
    game.start_game(SCREEN)

    while True:
        key = pygame.key.get_pressed()
        game.update_game(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if key[pygame.K_u]:
                main_menu()

        pygame.display.update()


def main_menu():
    while True:
        key = pygame.key.get_pressed()
        SCREEN.blit(BG, (0, 0))

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_title_font(100).render("Test your tunes", True, "#74b8ab")
        MENU_RECT = MENU_TEXT.get_rect(center=(850, 100))

        PLAY_BUTTON = Button(image=pygame.image.load("assets/Button_Background.png"), pos=(640, 400),
                             text_input="Play", font=get_font(50), base_color="#2e4e3d", hovering_color="White")
        QUIT_BUTTON = Button(image=None, pos=(1000, 600),
                             text_input="QUIT", font=get_font(50), base_color="#d7fcd4", hovering_color="White")

        SCREEN.blit(MENU_TEXT, MENU_RECT)

        for button in [PLAY_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if key[pygame.K_u]:
                play()
            if key[pygame.K_i]:
                pygame.quit()
                sys.exit()

        pygame.display.update()




def save_model(model_local):
    model_local.save(local_path)




if __name__ == '__main__':

    # model_history, model_trained, X_test, y_test = genre_classifier.run_classifier()
    # save_model(model_trained)
    input("Press Enter to continue...")
    main_menu()


# def prediction_genre():
#     predicted_genre = genre_classifier.predict_file("assets/BossaNovaOffset_Seed5.wav")
#     genre = predicted_genre
#     return genre


# print(type(model_trained))
# print(type(model_history))
# save_model(model_trained)


# clock.tick(60)
