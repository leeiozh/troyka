import pygame
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from abc import abstractmethod
from trasfomation import to_kepler, to_polar
import time
FPS = 30

YELLOW = (255, 200, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FIELD_WIDTH = 200
FIELD_HEIGHT = 40


class Window:

    @abstractmethod
    def run(self):
        pass

    def create_text(self, text, color, position, size, screen):
        f1 = pygame.font.Font(None, size)
        text1 = f1.render(text, True,
                          color, BLACK)
        screen.blit(text1, position)


class Menu(Window):

    def __init__(self, screen):
        self.buttons = []
        self.positions = []
        self.screen = screen
        self.create_text("x:", WHITE, (250, 150), 50, self.screen)
        self.create_text("y:", WHITE, (250, 250), 50, self.screen)
        self.create_text("z:", WHITE, (250, 350), 50, self.screen)
        self.create_text("vx:", WHITE, (700, 150), 50, self.screen)
        self.create_text("vy:", WHITE, (700, 250), 50, self.screen)
        self.create_text("vz:", WHITE, (700, 350), 50, self.screen)
        self.field_x = InsertField(0, 300, 150, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_y = InsertField(0, 300, 250, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_z = InsertField(0, 300, 350, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_vx = InsertField(0, 750, 150, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_vy = InsertField(0, 750, 250, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_vz = InsertField(0, 750, 350, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_integrator = ChoiceField(250, 450, ["EulerMethod", "RK4Method", "DormandPrinceMethod"], screen)
        self.insert_fields = np.array([self.field_x, self.field_y, self.field_z, self.field_vx, self.field_vy,
                                       self.field_vz])
        self.choice_fields = np.array([self.field_integrator])

    def run(self):
        finished = False
        clock = pygame.time.Clock()
        while not finished:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    finished = True
                    return 0
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for f in self.insert_fields:
                        if f.check_mouse():
                            f.activate()
                        else:
                            f.disactivate()

                    for f in self.choice_fields:
                        f.check_mouse()

                if event.type == pygame.KEYDOWN:
                    for f in self.insert_fields:
                        if event.key == 13:
                            f.disactivate()
                        if event.key == pygame.K_BACKSPACE:
                            if f.is_active and f.value != "":
                                f.value = f.value[:-2]
                                f.value += "|"
                        else:
                            if len(f.value) < 13:
                                f.insert(event.unicode)

            mouse_pos = pygame.mouse.get_pos()
            self.draw_objects()
            pygame.display.update()
            self.screen.fill(BLACK)

    def draw_objects(self):
        self.create_text("x:", WHITE, (250, 150), 50, self.screen)
        self.create_text("y:", WHITE, (250, 250), 50, self.screen)
        self.create_text("z:", WHITE, (250, 350), 50, self.screen)
        self.create_text("vx:", WHITE, (690, 150), 50, self.screen)
        self.create_text("vy:", WHITE, (690, 250), 50, self.screen)
        self.create_text("vz:", WHITE, (690, 350), 50, self.screen)
        self.create_text("Launch your satellite!", WHITE, (250, 50), 100, self.screen)
        for f in self.insert_fields:
            f.draw()
        for f in self.choice_fields:
            f.draw()

class Animation(Window):

    def __init__(self, x, y, coordinates, screen, delta_t):
        self.matrix = np.array([[0, -0.5], [-1, -0.3], [1, 1]])
        self.finished = False
        self.dt = delta_t
        self.dx = 0
        self.dy = 0
        self.counter = 0
        self.plot_step = 10
        self.acceleration = 10
        self.x = x
        self.y = y
        self.coordinates = coordinates
        self.screen = screen

    def run(self):
        finished = False
        clock = pygame.time.Clock()
        while not finished:
            clock.tick(FPS)
            # if self.counter % self.plot_step == 0:
            #     self.plot()
            self.draw_objects()
            #time.sleep(self.dt)
            self.counter += self.acceleration
            pygame.display.update()
            self.screen.fill(WHITE)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    finished = True

            keys = pygame.key.get_pressed()

            if keys[pygame.K_DOWN]:
                self.dy += 2
            if keys[pygame.K_UP]:
                self.dy -= 2
            if keys[pygame.K_RIGHT]:
                self.dx += 2
            if keys[pygame.K_LEFT]:
                self.dx -= 2

            if self.counter > len(self.coordinates) - 1:
                finished = True

    def plot(self):
        plt.cla()
        plt.xlim(min(self.x) * 1.2, max(self.x) * 1.2)
        plt.ylim(min(self.y) * 1.2, max(self.y) * 1.2)
        plt.xlabel("time, sec")
        plt.ylabel("x, m")
        plt.grid()
        plt.plot(self.x[0:self.counter], self.y[0:self.counter], 'r')
        plt.savefig("plot.png")

    def draw_objects(self):
        # plot_surf = pygame.image.load("plot.png")
        # plot_rect = plot_surf.get_rect(
        #     bottomright=(625, 475))
        # self.screen.blit(plot_surf, plot_rect)
        new_co = self.coordinates[self.counter] @ self.matrix
        pygame.draw.polygon(self.screen, [0, 0, 0], ([self.screen.get_width() / 2, 0], [self.screen.get_width(), 0],
                                                         [self.screen.get_width(), self.screen.get_height()],
                                                         [self.screen.get_width() / 2, self.screen.get_height()]))
        pygame.draw.circle(self.screen, [255, 255, 255], [self.screen.get_width() * 0.75 + self.dx,
                                                          self.screen.get_height() / 2 + self.dy], 20)
        pygame.draw.circle(self.screen, [255, 255, 255], [self.screen.get_width() * 0.75 + new_co[0] + self.dx,
                                                          self.screen.get_height() / 2 + new_co[1] + self.dy], 10)


class LoadingWindow(Window):

    def __init__(self, screen, integrator):
        self.screen = screen
        self.mas_x = np.array([])
        self.mas_vx = np.array([])
        self.mas_vy = np.array([])
        self.mas_vz = np.array([])
        self.clock = 0
        self.mas_t = np.array([0])
        self.mas_dt = np.array([])
        self.integrator = integrator
        self.q = np.array([6400000, 0, 0, 0, 6000, 0])
        self.position = np.array([[self.q[0], self.q[1], self.q[2]]], dtype=object)

    def run(self):
        finished = False
        clock = pygame.time.Clock()
        while not finished:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0

            pos = np.array([[self.q[0] / 25000, self.q[1] / 25000, self.q[2] / 25000]])
            self.position = np.append(self.position, pos, axis=0)
            # print(pos)
            self.mas_x = np.append(self.mas_x, self.q[0])
            self.mas_vx = np.append(self.mas_vx, self.q[3])
            self.mas_vy = np.append(self.mas_vx, self.q[4])
            self.mas_vz = np.append(self.mas_vx, self.q[5])
            self.clock += self.integrator.dt
            self.mas_t = np.append(self.mas_t, self.clock)
            self.mas_dt = np.append(self.mas_dt, self.integrator.dt)
            self.q = self.integrator.calc_next_step(self.q, 0)
            if max(self.mas_t) > 10000:
                return 1

            self.create_text("Loading " + str(round(max(self.mas_t) / 10000 * 100)) + "%", WHITE, (350, 350),
                             100, self.screen)
            pygame.display.update()
            self.screen.fill(BLACK)


class Field:

    @abstractmethod
    def draw(self):
        pass

    def create_text(self, text, color, position, size, screen):
        f1 = pygame.font.Font(None, size)
        text1 = f1.render(text, True,
                          color, WHITE)
        screen.blit(text1, position)


class InsertField(Field):

    def __init__(self, value, x, y, width, height, screen):
        self.is_active = False
        self.value = str(value)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.screen = screen

    def draw(self):
        pygame.draw.rect(self.screen, WHITE, (self.x, self.y, self.width, self.height))
        self.create_text(self.value, BLACK, (self.x + 7, self.y + 7), 40, self.screen)

    def insert(self, char):
        if self.is_active:
            self.value = self.value[:-1]
            self.value += str(char)
            self.value += "|"

    def activate(self):
        if not self.is_active:
            self.is_active = True
            self.value += "|"

    def disactivate(self):
        if self.is_active:
            self.value = self.value[:-1]
            self.is_active = False

    def check_mouse(self):
        if self.x < pygame.mouse.get_pos()[0] < self.x + self.width and self.y < pygame.mouse.get_pos()[1] < self.y + self.height:
            return True
        else:
            return False


class ChoiceField(Field):

    def __init__(self, x, y, elements, screen):
        self.choice = 0
        self.elements = elements
        self.x = x
        self.y = y
        self.screen = screen

    def check_mouse(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.x < mouse_pos[0] < self.x + 15 and self.y < mouse_pos[1] < self.y + 40:
            if self.choice > 0:
                self.choice -= 1
            else:
                self.choice = len(self.elements) - 1
        elif self.x + 15 * (len(self.elements[self.choice]) + 3) < mouse_pos[0] < self.x + 15 * (len(self.elements[self.choice]) + 6) and self.y < mouse_pos[1] < self.y + 40:
            if self.choice < len(self.elements) - 1:
                self.choice += 1
            else:
                self.choice = 0

    def draw(self):
        self.create_text("< " + self.elements[self.choice] + " >", WHITE, (self.x, self.y), 40, self.screen)

    def create_text(self, text, color, position, size, screen):
        f1 = pygame.font.Font(None, size)
        text1 = f1.render(text, True,
                          color, BLACK)
        screen.blit(text1, position)


def animation_map(q, dt):
    """
    draw main window with map and 3d visualization
    :param q: satellite's coordinates
    :return: picture
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    pygame.init()
    display_size = (1200, 800)
    screen = pygame.display.set_mode(display_size)
    finished = False
    scale = 0.00002
    dx = 0
    dy = 0
    dr = dt
    r = dr
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
        print_speed(dr, screen, display_size)

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
        if keys[pygame.K_KP_PLUS]:
            dr += 20
        if keys[pygame.K_KP_MINUS]:
            dr -= 20
            if dr < 20:
                dr = 1

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
    """
    print satellite's coordinates in GCRS
    :param screen: output's surface
    :param display_size: display's size
    :param q: satellite's coordinates
    :return: print a coordinates
    """
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
    """
    convert and print satellite's coordinates in Keplerian parameters
    :param screen: output's surface
    :param display_size: display's size
    :param q: satellite's coordinates
    :return: print a coordinates
    """
    font = pygame.font.Font(None, 25)
    font_color = (255, 255, 255)
    screen.blit(font.render("Satellite's coordinates in Kepler:", True, font_color),
                [display_size[0] * 0.55, display_size[1] * 0.7])

    q = to_kepler(q)

    text_surface = [font.render("Semimajor axis a = {0:0.2f} m".format(q[0]), True, font_color),
                    font.render("Eccentricity e = {0:0.2f}".format(q[1]), True, font_color),
                    font.render("Inclination i = {0:0.2f}".format(q[2] / np.pi * 180), True, font_color),
                    font.render("Longitude of the ascending node O = {0:0.2f}".format(q[3] / np.pi * 180),
                                True, font_color),
                    font.render("Argument of periapsis o = {0:0.2f}".format(q[4] / np.pi * 180), True, font_color),
                    font.render("True anomaly th = {0:0.2f}".format(q[5] / np.pi * 180), True, font_color)]
    for i in range(6):
        screen.blit(text_surface[i], [display_size[0] * 0.6, display_size[1] * 0.75 + i * 20])


def print_speed(dr, screen, display_size):
    """
    print visualization's speed
    :param dr: visualization's step
    :param screen: screen
    :param display_size: display size
    :return: printed speed
    """
    font = pygame.font.Font(None, 25)
    font_color = (0, 0, 0)
    screen.blit(font.render("Speed visualization: {}".format(dr), True, font_color),
                [display_size[0] * 0.05, display_size[1] * 0.05])
