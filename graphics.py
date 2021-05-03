import pygame
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from trasfomation import to_polar, to_kepler
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


def animation_map(q):

    pygame.init()
    display_size = (1200, 800)
    screen = pygame.display.set_mode(display_size)
    finished = False
    scale = 0.00002
    dx = 0
    dy = 0
    r = 1
    dr = 10
    coord_x = [to_polar(q[0])[0]]
    coord_y = [to_polar(q[0])[1]]
    clock = pygame.time.Clock()
    while not finished:
        coord_x.append(to_polar(q[r])[0])
        coord_y.append(to_polar(q[r])[1])
        if coord_x[-1] * coord_x[-2] < 0:
            coord_x.clear()
            coord_y.clear()
            coord_x.append(to_polar(q[r])[0])
            coord_y.append(to_polar(q[r])[1])

        fig, ax = plt.subplots(figsize=(6, 5))
        make_plot(fig, ax, coord_x, coord_y, clock)
        plot_surf = pygame.image.load("plot_map.png")

        plot_rect = plot_surf.get_rect(bottomright=(625, 475))
        screen.blit(plot_surf, plot_rect)

        new_co = q[r][:2]
        pygame.draw.polygon(screen, [0, 0, 0], ([display_size[0] / 2, 0], [display_size[0], 0],
                                                [display_size[0], display_size[1]],
                                                [display_size[0] / 2, display_size[1]]))
        pygame.draw.circle(screen, [124, 252, 0], [display_size[0] * 0.75 + dx, display_size[1] / 2 + dy], 20)
        pygame.draw.circle(screen, [255, 255, 255], [display_size[0] * 0.75 + int(scale * new_co[0]) + dx,
                                                     display_size[1] / 2 + int(scale * new_co[1]) + dy], 10)

        print_gcrs_coord(screen, display_size, q[r])
        print_kepler_coord(screen, display_size, q[r])

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

        if r > len(q) - 1:
            finished = True

    pygame.quit()


def make_plot(fig, ax, coord_x, coord_y, clock):
    """
    draw a plot
    :param coord_x: satellite's longitude
    :param coord_y: satellite's latitude
    :param clock: time
    :return: save plot in a folder
    """
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, color="gray")
    ax.set(xlabel="Longitude(Degrees)",  ylabel="Latitude(Degrees)")
    for i in range(-90, 91, 15):
        ax.plot([i * 2, i * 2], [-90, 90], color="lightgray", linewidth=0.4)
        ax.plot([-180, 180], [i, i], color="lightgray", linewidth=0.4)
    plt.plot(coord_x, coord_y, color="red")
    plt.savefig("plot_map.png")
    clock.tick(FPS)


def print_gcrs_coord(screen, display_size, q):
    font = pygame.font.Font(None, 25)
    font_color = (255, 255, 255)
    screen.blit(font.render("Satellite's coordinates in GCRS:", True, font_color), [display_size[0] * 0.55, display_size[1] * 0.05])
    text_surface = [font.render("X = {0:0.2f} m".format(q[0]), True, font_color),
                    font.render("Y = {0:0.2f} m".format(q[1]), True, font_color),
                    font.render("Z = {0:0.2f} m".format(q[2]), True, font_color),
                    font.render("Vx = {0:0.2f} m/s".format(q[3]), True, font_color),
                    font.render("Vy = {0:0.2f} m/s".format(q[4]), True, font_color),
                    font.render("Vz = {0:0.2f} m/s".format(q[5]), True, font_color)]
    for i in range(6):
        screen.blit(text_surface[i], [display_size[0] * 0.6, display_size[1] * 0.1 + i * 20])


def print_kepler_coord(screen, display_size, q):
    font = pygame.font.Font(None, 25)
    font_color = (255, 255, 255)
    screen.blit(font.render("Satellite's coordinates in Kepler:", True, font_color), [display_size[0] * 0.55, display_size[1] * 0.7])

    q = to_kepler(q)

    text_surface = [font.render("Semimajor axis a = {0:0.2f} m".format(q[0]), True, font_color),
                    font.render("Eccentricity e = {0:0.2f}".format(q[1]), True, font_color),
                    font.render("Inclination i = {0:0.2f}".format(q[2] / np.pi * 180), True, font_color),
                    font.render("Longitude of the ascending node O = {0:0.2f}".format(q[3] / np.pi * 180), True, font_color),
                    font.render("Argument of periapsis o = {0:0.2f}".format(q[4] / np.pi * 180), True, font_color),
                    font.render("True anomaly th = {0:0.2f}".format(q[5] / np.pi * 180), True, font_color)]
    for i in range(6):
        screen.blit(text_surface[i], [display_size[0] * 0.6, display_size[1] * 0.75 + i * 20])