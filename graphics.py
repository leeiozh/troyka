import pygame
import matplotlib.pyplot as plt
import numpy as np
FPS = 60


def animation(x, y, coordinates, dt):
    pygame.init()
    display_size = (1200, 800)
    screen = pygame.display.set_mode(display_size)
    matrix = np.array([[0, -0.5], [-1, -0.3], [1, 1]])
    finished = False
    dx = 0
    dy = 0
    r = 120
    dr = 30
    clock = pygame.time.Clock()
    plt.xlim(min(x) * 1.2, max(x) * 1.2)
    plt.ylim(min(y) * 1.2, max(y) * 1.2)
    plt.xlabel("time, sec")
    plt.ylabel("x, m")
    plt.grid()
    while not finished:
        if r % 120 == 0:
            plt.plot(x[r - 4 * dr:r], y[r - 4 * dr:r], 'r')
            plt.savefig("plot.png")
            clock.tick(FPS)
            plot_surf = pygame.image.load("plot.png")

        plot_rect = plot_surf.get_rect(
            bottomright=(625, 475))
        screen.blit(plot_surf, plot_rect)

        new_co = coordinates[r] @ matrix
        pygame.draw.polygon(screen, [0, 0, 0], ([display_size[0] / 2, 0], [display_size[0], 0],
                                                [display_size[0], display_size[1]], [display_size[0] / 2, display_size[1]]))
        pygame.draw.circle(screen, [255, 255, 255], [display_size[0]*0.75 + dx, display_size[1] / 2 + dy], 20)
        pygame.draw.circle(screen, [255, 255, 255], [display_size[0]*0.75 + new_co[0] + dx,
                                                     display_size[1] / 2 + new_co[1] + dy], 10)
        # time.sleep(dt)
        pygame.display.update()
        screen.fill([255, 255, 255])
        r += dr
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finished = True

        keys = pygame.key.get_pressed()

        if keys[pygame.K_DOWN]:
            dy += 2
        if keys[pygame.K_UP]:
            dy -= 2
        if keys[pygame.K_RIGHT]:
            dx += 2
        if keys[pygame.K_LEFT]:
            dx -= 2

        if r > len(coordinates) - 1:
            finished = True

    pygame.quit()

