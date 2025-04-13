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
        if self.current_genre_number == 9:
            self.current_genre_number = 0
        else:
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


    def genre_switch(self, genre):
        # Information from https://en.wikipedia.org/wiki/Main_Page
        string_text = ""
        artists = ""
        if genre == "BoomBap":
            string_text = "a subgenre of hiphop that was prominent in the East Coast during the golden age of hiphop" \
                          "from the late 1980s to the early 1990s"
            artists = "DJ Premier, Pete Rock, The Notorious B.I.G."

        elif genre == "BossaNova":
            string_text = "with roots in Rio de Janeiro, Brazil, this relaxed style of samba developed in the late " \
                          "1950s and early 1960s with influences from Jazz"
            artists = "João Gilberto, Quincy Jones, Laufey"

        elif genre == "BrazilianFunk":
            string_text = "otherwise known as funk carioca, this is a Brazilian hip hop-influenced music genre from " \
                          "the favelas of Rio de Janeiro around the 1980s, with influences from Miami bass and " \
                          "freestyle"
            artists = "DJ Marlboro, MIA, Major Lazer"

        elif genre == "Dancehall":
            string_text = "a genre of Jamaican pop music from the late 1970s, named after Jamaican dance halls" \
                          "in which popular recordings were played by local sound systems, originated in Kingston" \
                          "among lower and working-class people from the inner city."
            artists = "Yellowman, Sean Paul, Rihanna"

        elif genre == "Dnb":
            string_text = "also known as Drum and Bass a genre of electronic dance music with fast beats which grew " \
                          "out of the UK's jungle scene" \
                          "in the 1990s, with diverse influences from for example Jamaican dub and reggae this genre " \
                          "is affiliated with the ecstatic rave scene."
            artists = "Goldie, Noisia, Netsky "

        elif genre == "Dubstep":
            string_text = "a genre of electronic dance music from South London in the early 2000s, it became more " \
                          "commercially successful towards the early 2010s when it influenced several pop artists' work"

            artists = "Benga, Skrillex, A$AP Rocky"

        elif genre == "House":
            string_text = "a genre of electronic dance music created by DJs from Chicago's underground club culture" \
                          "when DJs began altering disco songs to give them a more mechanical beat, started in the " \
                          "early 1980s and became mainstream early 1988"
            artists = "Frankie Knuckles, Lady Gaga, Honey Dijon"

        elif genre == "JerseyClub":
            string_text = "a style of electronic club music that originated in Newark, New jersey in the late 1990s" \
                          "/early 2000s, inspired by Baltimore club's upbeat hybrid of house and hip hop and pushed by" \
                          "young producers in the late 2000s bringing it to global audiences"
            artists = "DJ Tameil, DJ Sliink, R3LL"

        elif genre == "Reggaeton":
            string_text = "a modern style of popular and electronic music that originated in Panama during the late " \
                          "1980s which rose to prominence in the late 1990s and early 2000s due to a lot of Puerto" \
                          " Rican musicians, evolved from Dancehall with elements of hiphop and Caribbean music"
            artists = "DJ Nelson, Daddy Yankee, Don Omar"

        if string_text != "":
            INFO_TEXT = pygame.font.Font("assets/Courier_New.ttf", 30).render((self.current_genre + ": " +
                                                                               string_text), True,
                                                                              "White")
            ARTIST_TEXT = pygame.font.Font("assets/Courier_New.ttf", 15).render("Artists using this genre include:"
                                                                                + artists, True,
                                                                                "White")
            return INFO_TEXT, ARTIST_TEXT
        return None



