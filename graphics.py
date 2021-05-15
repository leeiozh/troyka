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
GREY = (100, 100, 100)
FIELD_WIDTH = 255
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
    class of first window of program, where init all parameters
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
        self.field_filename1 = InsertField("load", col4_x + 50, 600, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_filename2 = InsertField("save", col4_x + 50, 700, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_mass = InsertField("100", col4_x, 75, FIELD_WIDTH, FIELD_HEIGHT, self.screen)
        self.field_integrator = ChoiceField(col1_x + 200, 455, ["EulerMethod", "RK4Method", "DorPrMethod"], screen, 1)
        self.field_x_plot = ChoiceField(col4_x + 200, 455, ["x", "y", "z", "vx", "vy", "vz", "time"], screen, 6)
        self.field_y_plot = ChoiceField(col4_x + 200, 495, ["x", "y", "z", "vx", "vy", "vz", "time"], screen)
        self.air_force_click = ClickField(col1_x, 575, self.screen)
        self.sun_force_click = ClickField(col1_x, 625, self.screen)
        self.load_click = ClickField(col3_x, 550, self.screen)
        self.save_click = ClickField(col3_x, 650, self.screen)
        self.insert_fields = np.array([self.field_x, self.field_y, self.field_z, self.field_vx, self.field_vy,
                                       self.field_vz, self.field_t, self.field_step, self.field_filename2,
                                       self.field_filename1, self.field_mass])
        self.choice_fields = np.array([self.field_integrator, self.field_x_plot, self.field_y_plot])
        self.click_fields = np.array([self.air_force_click, self.sun_force_click, self.load_click, self.save_click])
        self.start_button = Button(200, 680, 200, 70, "Start!", self.screen)

        self.text1 = Text("x:", WHITE, (col1_x, 150), 50, self.screen)
        self.text2 = Text("y:", WHITE, (col1_x, 225), 50, self.screen)
        self.text3 = Text("z:", WHITE, (col1_x, 300), 50, self.screen)
        self.text4 = Text("vx:", WHITE, (col3_x, 150), 50, self.screen)
        self.text5 = Text("vy:", WHITE, (col3_x, 225), 50, self.screen)
        self.text6 = Text("vz:", WHITE, (col3_x, 300), 50, self.screen)
        self.text7 = Text("time:", WHITE, (col1_x, 375), 50, self.screen)
        self.text8 = Text("step:", WHITE, (col3_x, 375), 50, self.screen)
        self.text9 = Text("mass:", WHITE, (col3_x, 75), 50, self.screen)
        self.text10 = Text("m", WHITE, (col2_x + FIELD_WIDTH + 20, 150), 50, self.screen)
        self.text11 = Text("m", WHITE, (col2_x + FIELD_WIDTH + 20, 225), 50, self.screen)
        self.text12 = Text("m", WHITE, (col2_x + FIELD_WIDTH + 20, 300), 50, self.screen)
        self.text13 = Text("s", WHITE, (col2_x + FIELD_WIDTH + 20, 375), 50, self.screen)
        self.text14 = Text("m/s", WHITE, (col4_x + FIELD_WIDTH + 20, 150), 50, self.screen)
        self.text15 = Text("m/s", WHITE, (col4_x + FIELD_WIDTH + 20, 225), 50, self.screen)
        self.text16 = Text("m/s", WHITE, (col4_x + FIELD_WIDTH + 20, 300), 50, self.screen)
        self.text17 = Text("s", WHITE, (col4_x + FIELD_WIDTH + 20, 375), 50, self.screen)
        self.text18 = Text("kg", WHITE, (col4_x + FIELD_WIDTH + 20, 75), 50, self.screen)
        self.text19 = Text("Launch your satellite!", WHITE, (330, 12), 70, self.screen)
        self.text20 = Text("Choose start parameters:", YELLOW, (150, 75), 50, self.screen)
        self.text21 = Text("Integrator: ", YELLOW, (col1_x, 450), 50, self.screen)
        self.text22 = Text("Plot:", YELLOW, (col3_x, 450), 50, self.screen)
        self.text23 = Text("x-axis ", WHITE, (col4_x, 450), 50, self.screen)
        self.text24 = Text("y-axis ", WHITE, (col4_x, 490), 50, self.screen)
        self.text25 = Text("Additional forces:", YELLOW, (col1_x, 525), 50, self.screen)
        self.text26 = Text("Atmosphere resistance", WHITE, (col1_x + 60, 575), 50, self.screen)
        self.text27 = Text("Light pressure", WHITE, (col1_x + 60, 625), 50, self.screen)
        self.text28 = Text("Open from file:", YELLOW, (col3_x + 40, 550), 50, self.screen)
        self.text29 = Text("filename", WHITE, (col3_x, 600), 50, self.screen)
        self.text30 = Text("Save to file:", YELLOW, (col3_x + 40, 650), 50, self.screen)
        self.text31 = Text("filename", WHITE, (col3_x, 700), 50, self.screen)

        self.texts = np.array([self.text1, self.text2, self.text3, self.text4, self.text5, self.text6, self.text7,
                               self.text8, self.text9, self.text10, self.text11, self.text12, self.text13, self.text14,
                               self.text15, self.text16, self.text17, self.text18, self.text19, self.text20,
                               self.text21, self.text22, self.text23, self.text24, self.text25, self.text26,
                               self.text27, self.text28, self.text29, self.text30, self.text31])

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
                        if f.check_mouse() and not self.load_click.is_active:
                            f.activate()
                        else:
                            f.deactivate()

                    if self.field_filename1.check_mouse():
                        self.field_filename1.activate()
                    else:
                        self.field_filename1.deactivate()

                    for f in self.choice_fields:
                        f.check_mouse()

                    for f in self.click_fields:
                        f.check_mouse()

                    for text in [self.text1, self.text2, self.text3, self.text4, self.text5, self.text6, self.text7,
                                 self.text8, self.text9, self.text10, self.text11, self.text12, self.text13,
                                 self.text14, self.text15, self.text16, self.text17, self.text18, self.text20,
                                 self.field_integrator, self.text25, self.text26, self.text27, self.air_force_click,
                                 self.sun_force_click, self.text21, self.save_click, self.text30, self.text31]:
                        if self.load_click.is_active:
                            text.deactivate()
                        else:
                            text.activate()

                    if self.start_button.check_mouse():
                        answer = [float(self.field_x.value), float(self.field_y.value), float(self.field_z.value),
                                  float(self.field_vx.value), float(self.field_vy.value),
                                  float(self.field_vz.value),
                                  float(self.field_t.value), float(self.field_step.value),
                                  self.field_x_plot.choice, self.field_y_plot.choice,
                                  self.air_force_click.is_active, self.sun_force_click.is_active,
                                  self.field_integrator.choice, int(self.field_mass.value),
                                  self.load_click.is_active, self.save_click.is_active, self.field_filename1.value,
                                  self.field_filename2.value, False]
                        return answer

                if event.type == pygame.KEYDOWN:
                    for f in self.insert_fields:
                        if event.key == 13:
                            f.deactivate()
                        if event.key == pygame.K_BACKSPACE:
                            if f.is_active and f.value != "":
                                f.value = f.value[:-2]
                                f.value += "|"
                                f.text.set_text(f.value)
                        else:
                            if len(f.value) < 15:
                                f.insert(event.unicode)

            self.start_button.check_mouse()
            self.draw_objects()
            pygame.display.update()
            self.screen.fill(BLACK)

    def draw_objects(self):
        """Draws all objects in the window"""
        self.start_button.draw()
        for text in self.texts:
            text.draw()
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

        :param x: data for x-axis of plot
        :param y: data for y-axis of plot
        :param coordinates: array of coordinate-vectors: [x, y, z, vx, vy, vz]
        :param output:
        :param screen: screen to draw
        :param delta_t: step of integration
        :param x_axis: name for x-axis
        :param y_axis: name for y-axis
        """
        self.matrix = np.array([[1, 0], [0, 1], [0, 0]])
        self.display = (1200, 800)
        self.scale = 1
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
            self.plot_map(self.screen, self.counter, self.display)
            self.draw_objects()
            # time.sleep(self.dt)
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
                self.dy -= 2
            elif keys[pygame.K_UP]:
                self.dy += 2
            elif keys[pygame.K_RIGHT]:
                self.dx -= 2
            elif keys[pygame.K_LEFT]:
                self.dx += 2
            elif keys[pygame.K_w]:
                self.scale += 0.02
            elif keys[pygame.K_s]:
                self.scale -= 0.02

            if self.counter > len(self.input) - 5:
                self.acceleration = 0

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
                                                        self.screen.get_height() / 2 + self.dy], self.scale * 20)
        pygame.draw.circle(self.screen, [255, 255, 255], [self.screen.get_width() * 0.75 + self.scale * new_co[0]
                                                          + self.dx,
                                                          self.screen.get_height() / 2 + self.scale * new_co[1]
                                                          + self.dy],
                           self.scale * 10)

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

    def plot_map(self, screen, counter, display_size):
        """
        draw satellite's dot on Earth
        :param display_size: display size
        :param counter: global counter
        :param screen: surface
        :return: picture
        """
        plot_surf = pygame.image.load("plot_map.png")
        plot_surf = pygame.transform.scale(plot_surf, (
            int(500 / 1200 * display_size[0] * 1.2), int(300 / 800 * display_size[1] * 1.1)))
        plot_rect = plot_surf.get_rect(bottomright=(610 / 1200 * display_size[0], 750 / 800 * display_size[1]))
        screen.blit(plot_surf, plot_rect)
        for i in range(0, counter - 1):
            if (self.long[i] * self.long[i + 1]) < -10:
                continue
            else:
                pygame.draw.line(self.screen, (0, 255, 0),
                                 ((310 / 1200 * display_size[0] + self.lang[i] + i * 0.005) / 1200
                                  * display_size[0], (570 + self.long[i] * 0.65) / 800 * display_size[1]),
                                 ((310 + self.lang[i + 1] + i * 0.005) / 1200 * display_size[0],
                                  (570 + self.long[i + 1] * 0.65) / 800 * display_size[1]), 4)


class LoadingWindow(Window):
    """
    class loading, second window, where make integration and output load line
    """

    def __init__(self, screen, param):
        """
        init function
        :param screen: surface
        :param param: np.array([x, y, z, vx, vy, vz, time, step,
        #          x-axis, y-axis, air_force, sun_force, integrator, mass, is_load,
        is_save, file1, file2, is_finished)
        """
        self.screen = screen
        self.display = (1200, 800)
        self.duration = param[6]
        self.mas_x = np.array([])
        self.mas_y = np.array([])
        self.x_axis = param[8]
        self.y_axis = param[9]
        self.clock = 0
        self.mas_t = np.array([0])
        self.mas_dt = np.array([])
        self.integrator = integrators[param[12]]
        self.integrator.dt = param[7]
        self.integrator.forces = np.array([])
        self.integrator.m = param[13]
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
        self.output = np.zeros(int(self.duration / self.integrator.dt) * 6)
        self.output.shape = (int(self.duration / self.integrator.dt), 6)
        self.output[0] = self.curr_q
        self.position = np.zeros(int(self.duration / self.integrator.dt) * 3)
        self.position.shape = (int(self.duration / self.integrator.dt), 3)
        self.position[0] = self.curr_q[:3]
        self.is_load = param[-5]
        self.is_save = param[-4]
        self.load_file = param[-3]
        self.save_file = param[-2]

    def run(self):
        """
        runtime function, which build all integration
        :return: all ballistics parameters
        """
        finished = False
        while not finished:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0

            self.clock += self.integrator.dt
            self.curr_q = self.integrator.calc_next_step(self.curr_q, 0)
            if self.is_load:
                self.check(self.load_file)
                return 1
            if self.is_save:
                np.save(self.save_file + "/ballistic%s.npy" % self.counter, self.curr_q)
            self.output[self.counter] = self.curr_q
            if self.x_axis < 6:
                self.mas_x[self.counter] = self.curr_q[self.x_axis]
            else:
                self.mas_x[self.counter] = self.counter * self.curr_dt
            if self.y_axis < 6:
                self.mas_y[self.counter] = self.curr_q[self.y_axis]
            else:
                self.mas_y[self.counter] = self.counter * self.curr_dt
            self.position[self.counter] = self.curr_q[0:3] / 25000
            if self.clock >= self.duration:
                return 1

            if self.counter % 50 == 0:
                self.create_text("Loading " + str(round(self.clock / self.duration * 100)) + "%", WHITE,
                                 (350, 350),
                                 100, self.screen)
                self.draw_rocket(self.screen, self.display, (self.clock / self.duration))
                pygame.display.update()
                self.screen.fill(BLACK)
            self.counter += 1

    def check(self, name):
        """
        load ballistic from file
        :param name: file name
        :return: coordinates and velocities
        """
        for i in range(0, int(self.duration / self.integrator.dt)):
            f = np.load(name + "/ballistic%s.npy" % (i + 1))
            self.output[i] = f
            if self.x_axis < 6:
                self.mas_x[i] = f[self.x_axis]
            else:
                self.mas_x[i] = i * self.curr_dt
            if self.y_axis < 6:
                self.mas_y[i] = f[self.y_axis]
            else:
                self.mas_y[i] = i * self.curr_dt
            self.position[i] = f[0:3] / 25000

    @staticmethod
    def draw_rocket(screen, display_size, speed):
        """
        draw rocket on the screen
        :param screen: screen
        :param display_size: display size
        :param speed: rocket position
        :return: picture
        """
        plot_surf = pygame.image.load("rocket.png")
        plot_surf = pygame.transform.scale(plot_surf, (120, 120))
        plot_rect = plot_surf.get_rect(bottomright=((display_size[0] + 124) * speed, 0.7 * display_size[1]))
        screen.blit(plot_surf, plot_rect)


class Text:
    """
    class for working with text fields
    """

    def __init__(self, text, color, position, size, screen, background=BLACK):
        """
        init function
        :param text: text
        :param color: color
        :param position: text position
        :param size: field size
        :param screen: surface
        :param background: background
        """
        self.text = text
        self.base_color = color
        self.current_color = color
        self.x = position[0]
        self.y = position[1]
        self.size = size
        self.screen = screen
        self.background = background
        self.is_active = True

    def draw(self):
        """
        draw text on screen
        :return:
        """
        f1 = pygame.font.Font(None, self.size)
        text1 = f1.render(self.text, True,
                          self.current_color, self.background)
        self.screen.blit(text1, (self.x, self.y))

    def activate(self):
        """
        change active fields color
        :return:
        """
        if not self.is_active:
            self.is_active = True
            self.current_color = self.base_color

    def deactivate(self):
        """
        chnge deactive fields in grey color
        :return:
        """
        if self.is_active:
            self.is_active = False
            self.current_color = (100, 100, 100)

    def set_text(self, text):
        """
        set text
        :param text: text
        :return:
        """
        self.text = text


class Field:
    """
    base class for all fields
    """

    @abstractmethod
    def draw(self):
        """
        abstract method for drawing
        :return:
        """
        pass


class InsertField(Field):
    """
    class for inserting
    """

    def __init__(self, value, x, y, width, height, screen):
        """
        init function
        :param value: value
        :param x: x position on screen
        :param y: y position on screen
        :param width: width
        :param height: height
        :param screen: surface
        """
        self.is_active = False
        self.value = str(value)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.screen = screen
        self.text = Text(self.value, BLACK, (self.x + 7, self.y + 7), 40, self.screen, WHITE)

    def draw(self):
        """
        drawing text on screen
        :return:
        """
        pygame.draw.rect(self.screen, WHITE, (self.x, self.y, self.width, self.height))
        self.text.draw()

    def insert(self, char):
        """
        set text in field
        :param char: symbol
        :return:
        """
        if self.is_active:
            self.value = self.value[:-1]
            self.value += str(char)
            self.value += "|"
            self.text.set_text(self.value)

    def activate(self):
        """
        activate field
        :return:
        """
        if not self.is_active:
            self.is_active = True
            self.value += "|"
            self.text.set_text(self.value)

    def deactivate(self):
        """
        disactivate field
        :return:
        """
        if self.is_active:
            self.value = self.value[:-1]
            self.is_active = False
            self.text.set_text(self.value)

    def check_mouse(self):
        """
        check mouse position
        :return:
        """
        if self.x < pygame.mouse.get_pos()[0] < self.x + self.width and self.y < pygame.mouse.get_pos()[1] < self.y \
                + self.height:
            return True
        else:
            return False


class ChoiceField(Field):
    """
    class for choosing buttons
    """

    def __init__(self, x, y, elements, screen, start_choice=0):
        """
        init function
        :param x: x position
        :param y: y position
        :param elements: elements
        :param screen: surface
        :param start_choice: start parameter
        """
        self.choice = start_choice
        self.elements = elements
        self.x = x
        self.y = y
        self.screen = screen
        self.text = Text("< > " + self.elements[self.choice], WHITE, (self.x, self.y), 40, self.screen, BLACK)
        self.is_alive = True

    def check_mouse(self):
        """
        check mouse position
        :return:
        """
        if self.is_alive:
            mouse_pos = pygame.mouse.get_pos()
            if self.x < mouse_pos[0] < self.x + 15 and self.y < mouse_pos[1] < self.y + 40:
                if self.choice > 0:
                    self.choice -= 1
                else:
                    self.choice = len(self.elements) - 1
                self.text.set_text("< > " + self.elements[self.choice])
            elif self.x + 25 < mouse_pos[0] < self.x + 40 and self.y < mouse_pos[1] < self.y + 40:
                if self.choice < len(self.elements) - 1:
                    self.choice += 1
                else:
                    self.choice = 0
                self.text.set_text("< > " + self.elements[self.choice])

    def draw(self):
        """
        draw text in the screen
        :return:
        """
        self.text.draw()

    def deactivate(self):
        """
        deactivate field
        :return:
        """
        self.is_alive = False

    def activate(self):
        """
        activate field
        :return:
        """
        self.is_alive = True


class ClickField(Field):
    """
    class for click fields
    """

    def __init__(self, x, y, screen):
        """
        init function
        :param x: x position
        :param y: y position
        :param screen: surface
        """
        self.r = 14
        self.x = x + self.r
        self.y = y + 1.2 * self.r
        self.screen = screen
        self.is_active = False
        self.is_alive = True

    def draw(self):
        """
        draw text on the screen
        :return:
        """
        if self.is_alive:
            pygame.draw.circle(self.screen, WHITE, (self.x, self.y), self.r)
            if self.is_active:
                pygame.draw.circle(self.screen, BLACK, (self.x, self.y), self.r / 2)
        else:
            pygame.draw.circle(self.screen, GREY, (self.x, self.y), self.r)

    def change(self):
        """
        change button's clickable
        :return:
        """
        if self.is_alive:
            self.is_active = not self.is_active

    def check_mouse(self):
        """
        check mouse position
        :return:
        """
        mouse_pos = pygame.mouse.get_pos()
        if np.sqrt((mouse_pos[0] - self.x) ** 2 + (mouse_pos[1] - self.y) ** 2) < self.r:
            self.change()

    def deactivate(self):
        """
        deactivate field
        :return:
        """
        self.is_alive = False
        self.is_active = False

    def activate(self):
        """
        activate field
        :return:
        """
        self.is_alive = True


class Button(Field):
    """
    class for buttons
    """

    def __init__(self, x, y, w, h, text, screen, color=YELLOW):
        """
        init function
        :param x: x position
        :param y: y position
        :param w: width
        :param h: height
        :param text: text
        :param screen: surface
        :param color: color
        """
        self.x = x
        self.y = y
        self.screen = screen
        self.color = color
        self.text = text
        self.w = w
        self.h = h
        self.is_active = False
        self.text_class = Text(self.text, BLACK, (self.x + (self.w - len(self.text) * 22) / 2, self.y + self.h * 0.2),
                               self.h, self.screen, self.color)

    def draw(self):
        """
        draw text on the screen
        :return:
        """
        pygame.draw.rect(self.screen, self.color, (self.x, self.y, self.w, self.h))
        self.text_class.draw()

    def check_mouse(self):
        """
        check mouse position
        :return:
        """
        mouse_pos = pygame.mouse.get_pos()
        if self.x < mouse_pos[0] < self.x + self.w and self.y < mouse_pos[1] < self.y + self.h:
            self.is_active = True
            self.color = (255, 235, 0)
            self.text_class = Text(self.text, BLACK,
                                   (self.x + (self.w - len(self.text) * 22) / 2, self.y + self.h * 0.2),
                                   self.h, self.screen, self.color)
            return True
        else:
            self.is_active = False
            self.color = YELLOW
            self.text_class = Text(self.text, BLACK,
                                   (self.x + (self.w - len(self.text) * 22) / 2, self.y + self.h * 0.2),
                                   self.h, self.screen, self.color)
            return False
