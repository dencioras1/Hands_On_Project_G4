import numpy as np
import pygame, sys
from game import Game
from assets.button import Button
from GenreClassifier import GenreClassifier

pygame.init()

SCREEN = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("Menu")
BG = pygame.image.load("assets/glow.jpg")
game = Game(None)


def get_font(size):  # Returns Press-Start-2P in the desired size
    return pygame.font.Font("assets/Courier_New.ttf", size)

def get_title_font(size):
    return pygame.font.Font("assets/shagade.ttf", size)


def play():
    while True:
        PLAY_MOUSE_POS = pygame.mouse.get_pos()
        game.genre = "Rock"  # Here will the randomizer go and finally this will be used to check with the prediction
        game.update_game(SCREEN)
        PLAY_BACK = Button(image=None, pos=(640, 650),
                           text_input="BACK", font=get_font(75), base_color="White", hovering_color="Green")

        PLAY_BACK.changeColor(PLAY_MOUSE_POS)
        PLAY_BACK.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BACK.checkForInput(PLAY_MOUSE_POS):
                    main_menu()

        pygame.display.update()


def options():
    while True:
        OPTIONS_MOUSE_POS = pygame.mouse.get_pos()

        SCREEN.fill("white")

        OPTIONS_TEXT = get_font(45).render("This is the OPTIONS screen.", True, "Black")
        OPTIONS_RECT = OPTIONS_TEXT.get_rect(center=(640, 260))
        SCREEN.blit(OPTIONS_TEXT, OPTIONS_RECT)

        OPTIONS_BACK = Button(image=None, pos=(640, 460),
                              text_input="BACK", font=get_font(75), base_color="Black", hovering_color="Green")

        OPTIONS_BACK.changeColor(OPTIONS_MOUSE_POS)
        OPTIONS_BACK.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if OPTIONS_BACK.checkForInput(OPTIONS_MOUSE_POS):
                    main_menu()

        pygame.display.update()


def main_menu():
    while True:
        SCREEN.blit(BG, (0, 0))


        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_title_font(100).render("Test your tunes", True, "#74b8ab")
        MENU_RECT = MENU_TEXT.get_rect(center=(850, 100))

        PLAY_BUTTON = Button(image=pygame.image.load("assets/Button_Background.png"), pos=(640, 400),
                             text_input="Play", font=get_font(50), base_color="#2e4e3d", hovering_color="White")
        OPTIONS_BUTTON = Button(image=None, pos=(1000, 1000),
                                text_input="", font=get_font(50), base_color="#d7fcd4", hovering_color="White")
        QUIT_BUTTON = Button(image=None, pos=(1000, 600),
                             text_input="QUIT", font=get_font(50), base_color="#d7fcd4", hovering_color="White")

        SCREEN.blit(MENU_TEXT, MENU_RECT)

        for button in [PLAY_BUTTON, OPTIONS_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                    play()
                if OPTIONS_BUTTON.checkForInput(MENU_MOUSE_POS):
                    options()
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()

def prediction_genre():
    predicted_genre = genre_classifier.predict_file("assets/BossaNovaOffset_Seed5.wav")
    genre = predicted_genre
    return genre

genre_classifier = GenreClassifier()
genre_classifier.run_classifier()
# genre_classifier.load_model_and_encoder()

main_menu()
