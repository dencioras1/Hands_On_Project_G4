import keras
import numpy as np
import pygame, sys
from matplotlib import pyplot as plt
import serial
from game import Game
from assets.button import Button
from GenreClassifier import GenreClassifier
from ButtonController import ButtonController

serial_com = '/dev/tty.usbmodem142101'
serial_baud = 9600
serial_timeout = 0.1

button_controller = ButtonController(serial_com, serial_baud, serial_timeout)

pygame.init()

SCREEN = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("Menu")
BG = pygame.image.load("assets/glow.jpg")
game = Game(None)
clock = pygame.time.Clock()
genre_classifier = GenreClassifier()
local_path = "saved_models/model1.keras"
model = None

def get_font(size):  # Returns Press-Start-2P in the desired size
    return pygame.font.Font("assets/Courier_New.ttf", size)

def get_title_font(size):
    return pygame.font.Font("assets/shagade.ttf", size)

def play():
    game.start_introduction(SCREEN)
    button_controller.set_in_game_true()
    user_recording = None

    # Game Loop
    while True:

        game.update_screen(SCREEN, True)
        pygame.display.update()


        input = button_controller.handle_serial_input()

        # Check if game is done, ie there is a recording
        if button_controller.get_has_recorded_and_saved():
            print("User has recorded, stopping in game loop")
            break



    # here we can process, but mke sure to start loading animation so the user knows

        #step 1, tell game object it time to process user output
    game.classify_input()
    # Post-Game Loop - Show results and give option to restart/menu

    button_controller.set_in_game_false()
    while True:

        game.update_screen(SCREEN, False)
        pygame.display.update()

        input = button_controller.handle_serial_input()

        # Handle input for top left/right buttons
        # Top left handles: play
        # Top right handle: going back
        if input == "TL":
            print("Top Left Pressed!")
            play()
            break

        if input == "TR":
            print("Top Right Pressed!")
            main_menu()
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()






def main_menu():
    button_controller.set_in_game_false()

    while True:

        input = button_controller.handle_serial_input()

        # key = pygame.key.get_pressed()

        SCREEN.blit(BG, (0, 0))

        # MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_title_font(100).render("Test your tunes", True, "#74b8ab")
        MENU_RECT = MENU_TEXT.get_rect(center=(850, 100))

        PLAY_BUTTON = Button(image=pygame.image.load("assets/Button_Background.png"), pos=(640, 400),
                             text_input="Play", font=get_font(50), base_color="#2e4e3d", hovering_color="White")
        QUIT_BUTTON = Button(image=None, pos=(1000, 600),
                             text_input="QUIT", font=get_font(50), base_color="#d7fcd4", hovering_color="White")

        SCREEN.blit(MENU_TEXT, MENU_RECT)

        for button in [PLAY_BUTTON, QUIT_BUTTON]:
            button.update(SCREEN)
        
        if input == "TL":
            play()
        if input == "TR":
            pygame.quit()
            sys.exit()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()




def save_model(model_local):
    model_local.save(local_path)




if __name__ == '__main__':

    # model_history, model_trained, X_test, y_test = genre_classifier.run_classifier()
    # save_model(model_trained)
    main_menu()


# clock.tick(60)
