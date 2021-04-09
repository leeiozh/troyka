import pygame
import time


def animation(x, y, dt):
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))

    finished = False
    for r in range(len(x)):
        pygame.draw.circle(screen, [0, 0, 0], [600, 400], 20)
        pygame.draw.circle(screen, [0, 0, 0], [600 + x[r], 400 + y[r]], 5)
        # time.sleep(dt)
        pygame.display.update()
        screen.fill([255, 255, 255])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finished = True
        if finished:
            break

    pygame.quit()
