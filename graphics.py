import pygame
import numpy as np
FPS = 120


def animation(coordinates, dt):
    pygame.init()
    display_size = (1200, 800)
    screen = pygame.display.set_mode(display_size)

    matrix = np.array([[0.7, -0.5], [1, -0.3], [1, -1]])
    finished = False
    r = 0
    clock = pygame.time.Clock()
    while not finished:
        clock.tick(FPS)
        new_co = coordinates[r] @ matrix
        print(new_co)
        pygame.draw.polygon(screen, [0, 0, 0], ([display_size[0] / 2, 0], [display_size[0], 0],
                                                [display_size[0], display_size[1]], [display_size[0] / 2, display_size[1]]))
        pygame.draw.circle(screen, [255, 255, 255], [display_size[0]*0.75, display_size[1] / 2], 20)
        pygame.draw.circle(screen, [255, 255, 255], [display_size[0]*0.75 + new_co[0], display_size[1] / 2 + new_co[1]], 10)
        # time.sleep(dt)
        pygame.display.update()
        screen.fill([255, 255, 255])
        r += 15
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finished = True
        if r > len(coordinates) - 1:
            finished = True

    pygame.quit()

