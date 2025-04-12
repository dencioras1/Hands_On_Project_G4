import os
import sys

import pygame


class Animation(pygame.sprite.Sprite):

    def __init__(self, pos_x, pos_y, path):
        super().__init__()
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.path = path
        self.sprites = []
        self.is_animating = False
        for filename in sorted(os.listdir(path)):
            full_path = os.path.join(path, filename)
            self.sprites.append(pygame.image.load(full_path))
        self.current_sprite = 0
        self.image = self.sprites[self.current_sprite]

        self.rect = self.image.get_rect()
        self.rect.midtop = [pos_x, pos_y]

    def animate(self):
        self.is_animating = True

    def update(self):
        if self.is_animating:
            self.current_sprite += 0.2
            if self.current_sprite >= len(self.sprites):
                self.current_sprite = 0

            self.image = self.sprites[int(self.current_sprite)]



# pygame.init()
# clock = pygame.time.Clock()
# screen = pygame.display.set_mode((400, 400))
# moving_sprites = pygame.sprite.Group()
# animation = Animation(10, 10, "assets/Animations/loading")
# moving_sprites.add(animation)
#
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()
#
#     screen.fill("white")
#     moving_sprites.draw(screen)
#     moving_sprites.update()
#     pygame.display.flip()
#     clock.tick(60)


