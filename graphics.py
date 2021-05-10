import pygame
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from abc import abstractmethod
from trasfomation import to_kepler, to_polar
from forces import ResistForce, GravityForce, SunForce, TestForce
from IntegrateMethods import RK4Method, EulerMethod1, EulerMethod2, DormandPrinceMethod
from astropy.time import Time
FPS = 30

YELLOW = (255, 200, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FIELD_WIDTH = 200
FIELD_HEIGHT = 40
col1_x = 150
col2_x = 250
col3_x = 700
col4_x = 810

mass = 1
square = 0.2
cx = 1
time = Time('1999-01-01T00:00:00.123456789', format='isot', scale='utc')

integrator1 = EulerMethod2(0, mass, 0)
integrator2 = RK4Method(0, mass, 0)
integrator3 = DormandPrinceMethod(0, mass, 0)

integrators = [integrator1, integrator2, integrator3]

g_force = GravityForce(mass)
air_force = ResistForce(square, cx)
sun_force = SunForce(square)

forces = [g_force, air_force, sun_force]


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
        self.field_x = InsertField(6700000, col2_x, 150, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_y = InsertField(0, col2_x, 225, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_z = InsertField(0, col2_x, 300, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_t = InsertField(5000, col2_x, 375, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_vx = InsertField(0, col4_x, 150, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_vy = InsertField(6780, col4_x, 225, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_vz = InsertField(0, col4_x, 300, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_step = InsertField(10, col4_x, 375, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_integrator = ChoiceField(col1_x + 200, 455, ["EulerMethod", "RK4Method", "DorPrMethod"], screen, 1)
        self.field_xplot = ChoiceField(col4_x + 200, 455, ["x", "y", "z", "vx", "vy", "vz", "time"], screen, 6)
        self.field_yplot = ChoiceField(col4_x + 200, 505, ["x", "y", "z", "vx", "vy", "vz", "time"], screen)
        self.air_force_click = ClickField(col1_x, 575, self.screen)
        self.sun_force_click = ClickField(col1_x, 625, self.screen)
        self.insert_fields = np.array([self.field_x, self.field_y, self.field_z, self.field_vx, self.field_vy,
                                       self.field_vz, self.field_t, self.field_step])
        self.choice_fields = np.array([self.field_integrator, self.field_xplot, self.field_yplot])
        self.click_fields = np.array([self.air_force_click, self.sun_force_click])
        self.start_button = Button(500, 700, 200, 70, "Start!", screen)

    def run(self):
        """
        Returns parameters as an np.array([x, y, z, vx, vy, vz, time, step,
         x-axis, y-axis, air_force, sun_force, integrator, is_finished)
         axis: 0 -- x, 1 -- y, 2 -- z, 3 -- vx, 4 -- vy, 5 -- vz, 6 -- time
         forces: bool variables
         integrator: 0 -- Euler, 1 -- RK4, 2 -- Dormand-Prince
         is_finished: True if stop, False if go next
        :return:
        """
        finished = False
        clock = pygame.time.Clock()
        while not finished:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    finished = True
                    return np.array([True])
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for f in self.insert_fields:
                        if f.check_mouse():
                            f.activate()
                        else:
                            f.disactivate()

                    for f in self.choice_fields:
                        f.check_mouse()

                    for f in self.click_fields:
                        f.check_mouse()

                    if self.start_button.check_mouse():
                        answer = [float(self.field_x.value), float(self.field_y.value), float(self.field_z.value),
                                  float(self.field_vx.value), float(self.field_vy.value), float(self.field_vz.value),
                                  float(self.field_t.value), float(self.field_step.value),
                                  self.field_xplot.choice, self.field_yplot.choice,
                                  self.air_force_click.is_active, self.sun_force_click.is_active,
                                  self.field_integrator.choice, False]
                        print(answer)
                        return answer

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
            self.start_button.check_mouse()
            self.draw_objects()
            pygame.display.update()
            self.screen.fill(BLACK)

    def draw_objects(self):
        self.create_text("x:", WHITE, (col1_x, 150), 50, self.screen)
        self.create_text("y:", WHITE, (col1_x, 225), 50, self.screen)
        self.create_text("z:", WHITE, (col1_x, 300), 50, self.screen)
        self.create_text("vx:", WHITE, (col3_x, 150), 50, self.screen)
        self.create_text("vy:", WHITE, (col3_x, 225), 50, self.screen)
        self.create_text("vz:", WHITE, (col3_x, 300), 50, self.screen)
        self.create_text("time:", WHITE, (col1_x, 375), 50, self.screen)
        self.create_text("step:", WHITE, (col3_x, 375), 50, self.screen)
        self.create_text("m", WHITE, (col2_x + FIELD_WIDTH + 20, 150), 50, self.screen)
        self.create_text("m", WHITE, (col2_x + FIELD_WIDTH + 20, 225), 50, self.screen)
        self.create_text("m", WHITE, (col2_x + FIELD_WIDTH + 20, 300), 50, self.screen)
        self.create_text("s", WHITE, (col2_x + FIELD_WIDTH + 20, 375), 50, self.screen)
        self.create_text("m/s", WHITE, (col4_x + FIELD_WIDTH + 20, 150), 50, self.screen)
        self.create_text("m/s", WHITE, (col4_x + FIELD_WIDTH + 20, 225), 50, self.screen)
        self.create_text("m/s", WHITE, (col4_x + FIELD_WIDTH + 20, 300), 50, self.screen)
        self.create_text("s", WHITE, (col4_x + FIELD_WIDTH + 20, 375), 50, self.screen)
        self.create_text("Launch your satellite!", WHITE, (330, 25), 80, self.screen)
        self.create_text("Choose start parameters:", YELLOW, (150, 100), 50, self.screen)
        self.create_text("integrator: ", YELLOW, (col1_x, 450), 50, self.screen)
        self.create_text("Plot:", YELLOW, (col3_x, 450), 50, self.screen)
        self.create_text("x-axis ", WHITE, (col4_x, 450), 50, self.screen)
        self.create_text("y-axis ", WHITE, (col4_x, 500), 50, self.screen)
        self.create_text("Additional forces:", YELLOW, (col1_x, 525), 50, self.screen)
        self.create_text("Atmosphere resistance", WHITE, (col1_x + 60, 575), 50, self.screen)
        self.create_text("Light pressure", WHITE, (col1_x + 60, 625), 50, self.screen)
        self.start_button.draw()
        for f in self.insert_fields:
            f.draw()
        for f in self.choice_fields:
            f.draw()
        for f in self.click_fields:
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

    def __init__(self, screen, param):
        # param = np.array([x, y, z, vx, vy, vz, time, step,
        #          x-axis, y-axis, air_force, sun_force, integrator, is_finished)
        self.screen = screen
        self.mas_x = np.array([])
        self.mas_vx = np.array([])
        self.mas_vy = np.array([])
        self.mas_vz = np.array([])
        self.clock = 0
        self.mas_t = np.array([0])
        self.mas_dt = np.array([])
        self.integrator = integrators[param[12]]
        self.integrator.dt = param[7]
        self.integrator.forces = np.array([])
        force_numbers = np.array([0])
        if param[10]:
            force_numbers = np.append(force_numbers, 1)
        if param[11]:
            force_numbers = np.append(force_numbers, 2)
        for i in force_numbers:
            self.integrator.forces = np.append(self.integrator.forces, forces[i])
        self.duration = param[6]
        self.q = np.array([param[0], param[1], param[2], param[3], param[4], param[5]])
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
            if max(self.mas_t) > self.duration:
                return 1

            self.create_text("Loading " + str(round(max(self.mas_t) / self.duration * 100)) + "%", WHITE, (350, 350),
                             100, self.screen)
            pygame.display.update()
            self.screen.fill(BLACK)


class Field:

    @abstractmethod
    def draw(self):
        pass

    def create_text(self, text, color1, position, size, screen, color2=WHITE):
        f1 = pygame.font.Font(None, size)
        text1 = f1.render(text, True,
                          color1, color2)
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

    def __init__(self, x, y, elements, screen, start_choice=0):
        self.choice = start_choice
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
        elif self.x + 15 * (len(self.elements[self.choice]) + 2) < mouse_pos[0] < self.x + 15 * (len(self.elements[self.choice]) + 6) and self.y < mouse_pos[1] < self.y + 40:
            if self.choice < len(self.elements) - 1:
                self.choice += 1
            else:
                self.choice = 0

    def draw(self):
        self.create_text("< " + self.elements[self.choice] + " >", WHITE, (self.x, self.y), 40, self.screen, BLACK)


class ClickField(Field):

    def __init__(self, x, y, screen):
        self.r = 14
        self.x = x + self.r
        self.y = y + 1.2 * self.r
        self.screen = screen
        self.is_active = False

    def draw(self):
        pygame.draw.circle(self.screen, WHITE, (self.x, self.y), self.r)
        if self.is_active:
            pygame.draw.circle(self.screen, BLACK, (self.x, self.y), self.r / 2)

    def change(self):
        self.is_active = not self.is_active

    def check_mouse(self):
        mouse_pos = pygame.mouse.get_pos()
        if np.sqrt((mouse_pos[0] - self.x) ** 2 + (mouse_pos[1] - self.y) ** 2) < self.r:
            self.change()


class Button(Field):

    def __init__(self, x, y, w, h, text, screen, color=YELLOW):
        self.x = x
        self.y = y
        self.screen = screen
        self.color = color
        self.text = text
        self.w = w
        self.h = h
        self.is_active = False

    def draw(self):
        pygame.draw.rect(self.screen, self.color, (self.x, self.y, self.w, self.h))
        self.create_text(self.text, BLACK, (self.x + (self.w - len(self.text) * 22) / 2, self.y + self.h * 0.2),
                         self.h, self.screen, self.color)

    def check_mouse(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.x < mouse_pos[0] < self.x + self.w and self.y < mouse_pos[1] < self.y + self.h:
            self.is_active = True
            self.color = (255, 235, 0)
            return True
        else:
            self.is_active = False
            self.color = YELLOW
            return False


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
