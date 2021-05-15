import pygame
import graphics

pygame.init()
screen = pygame.display.set_mode((1200, 800))
menu = graphics.Menu(screen)
parameters = menu.run()
if not parameters[-1]:
    load = graphics.LoadingWindow(screen, parameters)
    load.run()
    axis = ['x, m', 'y, m', 'z, m', 'vx, m/s', 'vy, m/s', 'vz, m/s', 't, s']
    anim = graphics.Animation(load.mas_x, load.mas_y, load.position, load.output, screen, 1,
                              axis[load.x_axis], axis[load.y_axis])
    anim.run()
pygame.quit()
