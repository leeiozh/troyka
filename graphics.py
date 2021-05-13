import pygame
import numpy as np
from abc import abstractmethod
from trasfomation import to_kepler, to_polar
from forces import ResistForce, GravityForce, SunForce
from IntegrateMethods import RK4Method, EulerMethod2, DormandPrinceMethod
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

mass = 100
square = 0.2
cx = 1
time = Time('2021-05-10T00:00:00.10', format='isot', scale='utc')

integrator1 = EulerMethod2(0, mass, 0)
integrator2 = RK4Method(0, mass, 0)
integrator3 = DormandPrinceMethod(0, mass, 0)

integrators = [integrator1, integrator2, integrator3]

g_force = GravityForce(mass)
air_force = ResistForce(square, cx)
sun_force = SunForce(square)

forces = [g_force, air_force, sun_force]


class Window:
    """
    abstract class for all program's windows
    """

    @abstractmethod
    def run(self):
        """
        abstract method, uses for runtime functions
        :return: smth useful
        """
        pass

    @staticmethod
    def create_text(text, color, position, size, screen, background=BLACK):
        """
            abstract method, create text on screen
            :return: text
        """
        f1 = pygame.font.Font(None, size)
        text1 = f1.render(text, True,
                          color, background)
        screen.blit(text1, position)


class Menu(Window):
    """
    class of first window of program, where inits all parameters
    """

    def __init__(self, screen):
        """
        init function
        :param screen: surface
        """
        self.buttons = []
        self.positions = []
        self.screen = screen
        self.field_x = InsertField(6700000, col2_x, 150, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_y = InsertField(100000, col2_x, 225, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_z = InsertField(100000, col2_x, 300, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_t = InsertField(5000, col2_x, 375, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_vx = InsertField(1000, col4_x, 150, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_vy = InsertField(6780, col4_x, 225, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_vz = InsertField(1000, col4_x, 300, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_step = InsertField(1, col4_x, 375, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_integrator = ChoiceField(col1_x + 200, 455, ["EulerMethod", "RK4Method", "DorPrMethod"], screen, 1)
        self.field_xplot = ChoiceField(col4_x + 200, 455, ["x", "y", "z", "vx", "vy", "vz", "time"], screen, 6)
        self.field_yplot = ChoiceField(col4_x + 200, 505, ["x", "y", "z", "vx", "vy", "vz", "time"], screen)
        self.air_force_click = ClickField(col1_x, 575, self.screen)
        self.sun_force_click = ClickField(col1_x, 625, self.screen)
        self.insert_fields = np.array([self.field_x, self.field_y, self.field_z, self.field_vx, self.field_vy,
                                       self.field_vz, self.field_t, self.field_step])
        self.choice_fields = np.array([self.field_integrator, self.field_xplot, self.field_yplot])
        self.click_fields = np.array([self.air_force_click, self.sun_force_click])
        self.start_button = Button(500, 670, 200, 70, "Start!", screen)

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

            self.start_button.check_mouse()
            self.draw_objects()
            pygame.display.update()
            self.screen.fill(BLACK)

    def draw_objects(self):
        """
        draws objects from menu
        :return: none
        """
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
        self.create_text("Choose start parameters:", YELLOW, (150, 90), 50, self.screen)
        self.create_text("Integrator: ", YELLOW, (col1_x, 450), 50, self.screen)
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
    """
    class animation, third window of program, which outputs all visualization in runtime
    """

    def __init__(self, x, y, coordinates, output, screen, delta_t, x_axis, y_axis):
        """
        init function
        :param x: array user's x axis
        :param y: array user's y axis
        :param coordinates: satellite's coordinates
        :param output: satellite's coordinates and velocities
        :param screen: surface
        :param delta_t: time step
        :param x_axis: user's x axis title
        :param y_axis: user's y axis title
        """
        self.matrix = np.array([[0, -0.5], [-1, -0.3], [1, 1]])
        self.display = (1200, 800)
        self.finished = False
        self.dt = delta_t
        self.dx = 0
        self.dy = 0
        self.counter = 0
        self.plot_step = 10
        self.acceleration = 6
        self.x = x
        self.y = y
        self.coordinates = coordinates
        self.input = output
        self.screen = screen
        self.plot_w = 400
        self.plot_h = 300
        self.plot_x = 70
        self.plot_y = 50
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.long = np.zeros(len(self.input))
        self.lang = np.zeros(len(self.input))
        for i in range(len(self.input)):
            self.long[i] = to_polar(self.input[i])[0]
            self.lang[i] = to_polar(self.input[i])[1]

    def run(self):
        """
        runtime function, which draw all pictures
        :return: image
        """
        finished = False
        clock = pygame.time.Clock()
        while not finished:
            clock.tick(FPS)
            self.plot_map(self.screen, self.counter)
            self.draw_objects()
            self.counter += self.acceleration
            self.plot(self.x, self.y, self.counter, self.x_axis, self.y_axis)
            self.print_kepler_coord(self.screen, self.display, self.input[self.counter])
            self.print_gcrs_coord(self.screen, self.display, self.input[self.counter])
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

            if self.counter > len(self.input) - 5:
                self.acceleration = 0
                self.print_done(self.screen, self.display)

    def plot(self, x, y, counter, x_title="", y_title=""):
        """
        draw plot with user's axes
        :param x: user's x ax
        :param y: user's y ax
        :param counter: global counter
        :param x_title: user's x title
        :param y_title: user's y title
        :return: plot image
        """
        self.create_text(y_title, BLACK, (self.plot_x, self.plot_y - 30), 30, self.screen, WHITE)
        self.create_text(x_title, BLACK, (self.plot_x + self.plot_w - 5 * len(x_title),
                                          self.plot_y + self.plot_h + 20), 30, self.screen, WHITE)
        pygame.draw.line(self.screen, BLACK, (self.plot_x, self.plot_y), (self.plot_x, self.plot_y + self.plot_h), 5)
        pygame.draw.line(self.screen, BLACK, (self.plot_x, self.plot_y + self.plot_h),
                         (self.plot_x + self.plot_w, self.plot_y + self.plot_h), 5)
        pygame.draw.line(self.screen, BLACK, (self.plot_x, self.plot_y), (self.plot_x - 10, self.plot_y + 15), 5)
        pygame.draw.line(self.screen, BLACK, (self.plot_x, self.plot_y), (self.plot_x + 10, self.plot_y + 15), 5)
        pygame.draw.line(self.screen, BLACK, (self.plot_x + self.plot_w, self.plot_y + self.plot_h),
                         (self.plot_x + self.plot_w - 15, self.plot_y + self.plot_h + 10), 5)
        pygame.draw.line(self.screen, BLACK, (self.plot_x + self.plot_w, self.plot_y + self.plot_h),
                         (self.plot_x + self.plot_w - 15, self.plot_y + self.plot_h - 10), 5)
        for i in range(6):
            pygame.draw.line(self.screen, BLACK, (self.plot_x + i * self.plot_w / 5, self.plot_y),
                             (self.plot_x + i * self.plot_w / 5, self.plot_y + self.plot_h), 1)
            pygame.draw.line(self.screen, BLACK, (self.plot_x, self.plot_y + self.plot_h - i * self.plot_h / 5),
                             (self.plot_x + self.plot_w, self.plot_y + self.plot_h - i * self.plot_h / 5), 1)
        kx = 0.8 * self.plot_w / (max(x) - min(x))
        ky = 0.8 * self.plot_h / (max(y) - min(y))
        x = self.plot_x + self.plot_w * 0.1 + kx * x - kx * min(x)
        y = self.plot_y + self.plot_h - ky * y - self.plot_h * 0.1 + ky * min(y)
        x = x[0:counter]
        y = y[0:counter]
        for i in range(len(x) - 2):
            pygame.draw.line(self.screen, (255, 0, 0), (x[i], y[i]), (x[i + 1], y[i + 1]), 5)

    def draw_objects(self):
        """
        draws Earth's and satellite's circles
        :return: image
        """
        new_co = self.coordinates[self.counter] @ self.matrix
        pygame.draw.polygon(self.screen, [0, 0, 0], ([self.screen.get_width() / 2, 0], [self.screen.get_width(), 0],
                                                     [self.screen.get_width(), self.screen.get_height()],
                                                     [self.screen.get_width() / 2, self.screen.get_height()]))
        pygame.draw.circle(self.screen, [0, 255, 255], [self.screen.get_width() * 0.75 + self.dx,
                                                        self.screen.get_height() / 2 + self.dy], 20)
        pygame.draw.circle(self.screen, [255, 255, 255], [self.screen.get_width() * 0.75 + new_co[0] + self.dx,
                                                          self.screen.get_height() / 2 + new_co[1] + self.dy], 10)

    @staticmethod
    def print_done(screen, display_size):
        font = pygame.font.Font(None, 10)
        font_color = (0, 255, 0)
        text = font.render("ALL DONE", True, font_color)
        screen.blit(text, [display_size[0] * 0.5, display_size[1] * 0.5])

    @staticmethod
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
        screen.blit(font.render("Satellite's coordinates in GCRS:", True, font_color),
                    [display_size[0] * 0.55, display_size[1] * 0.05])
        text_surface = [font.render("X = {0:0.2f} m".format(q[0]), True, font_color),
                        font.render("Y = {0:0.2f} m".format(q[1]), True, font_color),
                        font.render("Z = {0:0.2f} m".format(q[2]), True, font_color),
                        font.render("Vx = {0:0.2f} m/s".format(q[3]), True, font_color),
                        font.render("Vy = {0:0.2f} m/s".format(q[4]), True, font_color),
                        font.render("Vz = {0:0.2f} m/s".format(q[5]), True, font_color)]
        for i in range(6):
            screen.blit(text_surface[i], [display_size[0] * 0.6, display_size[1] * 0.1 + i * 20])

    @staticmethod
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

    def plot_map(self, screen, counter):
        """
        draw satellite's dot on Earth
        :param counter: global counter
        :param screen: surface
        :return: picture
        """
        plot_surf = pygame.image.load("plot_map.png")
        plot_surf = pygame.transform.scale(plot_surf, (int(500 * 1.2), int(300 * 1.1)))
        plot_rect = plot_surf.get_rect(bottomright=(610, 750))
        screen.blit(plot_surf, plot_rect)
        for i in range(0, counter - 1):
            pygame.draw.line(self.screen, (0, 255, 0), (310 + self.lang[i], 575 + self.long[i] * 0.6),
                             (310 + self.lang[i + 1], 575 + self.long[i + 1] * 0.6), 4)


class LoadingWindow(Window):
    """
    class loading, second window, where make integration and output load line
    """

    def __init__(self, screen, param):
        """
        init function
        :param screen: surface
        :param param: zero's satellite's parameters
        """
        # param = np.array([x, y, z, vx, vy, vz, time, step,
        #          x-axis, y-axis, air_force, sun_force, integrator, is_finished)
        self.screen = screen
        self.duration = param[6]
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
        self.curr_x = 0
        self.curr_y = 0
        self.curr_t = 0
        self.curr_dt = self.integrator.dt
        self.curr_q = np.array([param[0], param[1], param[2], param[3], param[4], param[5]])
        self.mas_x = np.zeros(int(self.duration / self.integrator.dt))
        self.mas_y = np.zeros(int(self.duration / self.integrator.dt))
        self.x_axis = param[8]
        self.y_axis = param[9]
        if self.x_axis < 6:
            self.mas_x[0] = self.curr_q[self.x_axis]
        if self.y_axis < 6:
            self.mas_y[0] = self.curr_q[self.y_axis]
        self.clock = 0
        self.counter = 0
        # self.mas_t = np.array([0])
        # self.mas_dt = np.array([])
        self.output = np.zeros(int(self.duration / self.integrator.dt) * 6)
        self.output.shape = (int(self.duration / self.integrator.dt), 6)
        self.output[0] = self.curr_q
        self.position = np.zeros(int(self.duration / self.integrator.dt) * 3)
        self.position.shape = (int(self.duration / self.integrator.dt), 3)
        self.position[0] = self.curr_q[:3]

    def run(self):
        """
        runtime function, which build all integration
        :return: all ballistics parameters
        """
        finished = False
        i = 0
        while not finished:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0

            self.counter += 1
            self.clock += self.integrator.dt
            self.curr_q = self.integrator.calc_next_step(self.curr_q, 0)
            np.save("load/ballistic%s.npy" % self.counter, self.curr_q)
            if self.clock > self.duration:
                return 1

            if i % 300 == 0:
                self.create_text("Loading " + str(round(self.clock / self.duration * 100)) + "%", WHITE,
                                 (350, 350),
                                 100, self.screen)
                pygame.display.update()
                self.screen.fill(BLACK)
            i += 1

    def check(self):
        for i in range(1, self.counter - 1):
            f = np.load("load/ballistic%s.npy" % i)
            self.output[i] = f
            if self.x_axis < 6:
                self.mas_x[i] = f[self.x_axis]
            else:
                self.mas_x[i] = i * self.curr_dt
                # self.mas_x = self.mas_t
            if self.y_axis < 6:
                self.mas_y[i] = f[self.y_axis]
            else:
                self.mas_y[i] = i * self.curr_dt
            self.position[i] = f[0:3] / 25000
        return True


class Field:

    @abstractmethod
    def draw(self):
        pass

    @staticmethod
    def create_text(text, color1, position, size, screen, color2=WHITE):
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
        if self.x < pygame.mouse.get_pos()[0] < self.x + self.width and self.y < pygame.mouse.get_pos()[1] < self.y \
                + self.height:
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
        elif self.x + 15 * (len(self.elements[self.choice]) + 2) < mouse_pos[0] < self.x + 15 * (
                len(self.elements[self.choice]) + 6) and self.y < mouse_pos[1] < self.y + 40:
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
