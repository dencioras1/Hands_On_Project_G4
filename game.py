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
genre_classifier = GenreClassifier()

pygame.mixer.init()
pygame.init()



class Game:

    def __init__(self, genre):
        self.genre = genre
        self.classification = None
        self.SCREEN = None
        self.current_genre_number = 1
        self.current_genre = None
        self.base_path = "Audio"
        self.in_game = False
        self.classified_values_output = None
        self.BGA_game = pygame.image.load("assets/BGA.jpg")

    def get_font(self, size):  # Returns Press-Start-2P in the desired size
        return pygame.font.Font("assets/Courier_New.ttf", size)

    def start_introduction(self, SCREEN):
        if self.current_genre_number == 9:
            self.current_genre_number = 0
        else:
            self.current_genre_number += 1

        # start at the first genre (0)
        self.current_genre = genre_classifier.labels[self.current_genre_number]
        genre_path = os.path.join(self.base_path, self.current_genre)

        in_game = True

        first_wav = None
        for file in os.listdir(genre_path):
            if file.endswith("Quantized.wav"):
                first_wav = os.path.join(genre_path, file)
                break  # Stop after the first .wav file

        pygame.mixer.music.load(first_wav)
        pygame.mixer.music.play()
        self.SCREEN = SCREEN

        # wait length of file
        while pygame.mixer.music.get_busy():
            SCREEN.fill("Black")
            animation.is_animating = True
            PLAY_TEXT = pygame.font.Font("assets/Courier_New.ttf", 45).render(f"This is the genre: {self.current_genre}"
                                                                              , True,
                                                                              "White")
            PLAY_RECT = PLAY_TEXT.get_rect(center=(640, 160))
            SCREEN.blit(PLAY_TEXT, PLAY_RECT)

            moving_sprites.draw(SCREEN)
            moving_sprites.update()
            pygame.display.flip()

        animation.is_animating = False
        SCREEN.fill("black")

    def update_screen(self, screen_size, ingame):
        screen_size.fill("black")
        PLAY_TEXT = pygame.font.Font("assets/Courier_New.ttf", 60).render(f"Your turn to try: {self.current_genre}",
                                                                          True,
                                                                          "White")
        PLAY_RECT = PLAY_TEXT.get_rect(center=(640, 160))
        screen_size.blit(PLAY_TEXT, PLAY_RECT)

        # Time to try it >> self.recording = True
        if not ingame:
            screen_size.blit(self.BGA_game, (0, -40))

            self.genre_switch(self.current_genre, screen_size)

            self.show_classification(screen=screen_size)

            PLAY_BACK = Button(image=None, pos=(320, 650),
                               text_input="AGAIN", font=self.get_font(60), base_color="#2e4e3d",
                               hovering_color="#74b8ab")
            QUIT = Button(image=None, pos=(960, 650),
                          text_input="MENU", font=self.get_font(60), base_color="#2e4e3d", hovering_color="#74b8ab")

            PLAY_BACK.update(screen_size)
            QUIT.update(screen_size)

    def classify_input(self):
        genre_classifier.load_model()
        r1, r2 = genre_classifier.predict_file("output.wav")
        self.classification = r1
        r2 = r2.flatten()
        self.classified_values_output = r2
        print(self.classified_values_output)

    def show_classification(self, screen):
        self.current_genre = genre_classifier.labels[self.current_genre_number]
        percentage_actual = self.classified_values_output[self.current_genre_number]
        genre_probs = list(zip(genre_classifier.labels, self.classified_values_output))
        sorted_genres = sorted(genre_probs, key=lambda x: x[1], reverse=True)
        # Create text strings
        actual_line = f"{self.current_genre}: {percentage_actual * 100:.2f}%"
        other_lines = [
            f"{genre}: {prob * 100:.2f}%"
            for genre, prob in sorted_genres
            if genre != self.current_genre
        ]

        # Fonts
        font_info = pygame.font.Font("assets/Courier_New.ttf", 25)

        # Background card
        pygame.draw.rect(screen, (30, 30, 30), (200, 390, 500, 150), border_radius=10)
        pygame.draw.rect(screen, (200, 200, 200), (200, 390, 500, 150), 2, border_radius=10)

        # Render the actual genre line
        final_text = font_info.render(actual_line, True, "White")
        screen.blit(final_text, final_text.get_rect(topleft=(220, 400)))

        # Render the next 3 genres
        for i, line in enumerate(other_lines[:3]):
            line_surface = font_info.render(line, True, "White")
            screen.blit(line_surface, line_surface.get_rect(topleft=(220, 440 + i * 30)))

    def genre_switch(self, genre, screen_size):
        # Information from https://en.wikipedia.org/wiki/Main_Page
        string_text = ""
        artists = ""
        if genre == "BoomBap":
            string_text = "a subgenre of hiphop that was prominent in the East Coast during the golden age of hiphop" \
                          "from the late 1980s to the early 1990s"
            artists = "DJ Premier, Pete Rock, The Notorious B.I.G. ✝"

        elif genre == "BossaNova":
            string_text = "with roots in Rio de Janeiro, Brazil, this relaxed style of samba developed in the late " \
                          "1950s and early 1960s with influences from Jazz"
            artists = "João Gilberto ✝, Quincy Jones, Laufey"

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

        elif genre == "DnB":
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
            artists = "Frankie Knuckles ✝, Lady Gaga, Honey Dijon"

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
        elif genre == "Trap":
            string_text = "This subgenre of hip-hop music originated at the start of the 1990s in the southern part of " \
                          "the USA and gets its name" \
                          " from the Atlanta slang term 'trap house', a house used exclusively to sell drugs. It became" \
                          " mainstream in the 2010 and became one of the most popular forms of American music. "
            artists = "Kendrik Lamar, Post Malone,  XXXTentacion ✝"

        if string_text != "":
            font_info = pygame.font.Font("assets/Courier_New.ttf", 25)
            font_artist = pygame.font.Font("assets/Courier_New.ttf", 20)

            # Wrap the info text to fit within 1000 pixels
            wrapped_lines = self.wrap_text(self.current_genre + ": " + string_text, font_info, max_width=1000)

            # Render and blit each line with spacing
            start_y = 120
            for i, line in enumerate(wrapped_lines):
                rendered_line = font_info.render(line, True, "Black")
                line_rect = rendered_line.get_rect(center=(640, start_y + i * 30))
                screen_size.blit(rendered_line, line_rect)

            # Add the artist line a bit further down
            artist_text = "Artists using this genre include: " + artists
            ARTIST_TEXT = font_artist.render(artist_text, True, "#2e4e3d")
            ARTIST_RECT = ARTIST_TEXT.get_rect(center=(640, start_y + len(wrapped_lines) * 30 + 20))
            self.SCREEN.blit(ARTIST_TEXT, ARTIST_RECT)

    def wrap_text(self, text, font, max_width):
        words = text.split(' ')
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + word + " "
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        lines.append(current_line.strip())
        return lines
