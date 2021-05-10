from astropy.time import Time
import numpy as np
import pygame
import graphics

# testing
# you can change the method here

pygame.init()
screen = pygame.display.set_mode((1200, 800))
menu = graphics.Menu(screen)
parameters = menu.run()
if not parameters[len(parameters) - 1]:
    load = graphics.LoadingWindow(screen, parameters)
    a = load.run()
    if a:
        anim = graphics.Animation(load.mas_t, load.mas_x, load.position, screen, 1)
        anim.run()
pygame.quit()
