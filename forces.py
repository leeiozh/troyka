from abc import ABC, abstractmethod
from astropy.coordinates import get_sun
from astropy.time import Time
import numpy as np


class BaseForce(ABC):
    """abstract class, which init a forces array"""
    Force = np.zeros(3, float)
    Time = Time('1999-01-01T00:00:00.123456789', format='isot', scale='utc')

    @abstractmethod
    def calc(self, *args):
        """virtual method, which take some parameters and return a force"""
        pass


class CenterForce(BaseForce):
    """
    class centrifugal force
    """
    mass = 0

    def __init__(self, mass):
        """
        take satellite's mass
        """
        self.mass = mass

    def calc(self, q, time):
        """
        calculate force
        :param q: satellite's coordinates and velocity
        :param time: current time
        :return: force
        """
        r = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2) ** 0.5
        v = (q[3] ** 2 + q[4] ** 2 + q[5] ** 2) ** 0.5
        self.Force[0] = self.mass * v ** 2 * q[0] / r / r
        self.Force[1] = self.mass * v ** 2 * q[1] / r / r
        self.Force[2] = self.mass * v ** 2 * q[2] / r / r

        return self.Force


class ResistForce(BaseForce):
    """
    class air resist force
    """
    Atmosphere = np.empty(211, float)
    input_atmosphere = open('atmosphere.csv', 'r')
    i = 0
    for line in input_atmosphere:
        Atmosphere[i] = float(line.split()[2])
        i += 1
    Square = 0
    Cx = 0

    def __init__(self, Square, Cx):
        """
        take satellite's square and mass
        """
        self.Square = Square
        self.Cx = Cx

    def get_atmosphere(self, height):
        """
        :param height: orbit's height
        :return: air's density
        """

        if height < 80000:
            return self.Atmosphere[0]
        elif height > 1200000:
            return self.Atmosphere[-1]
        elif 80000 < height < 500000:
            x = int(height / 6000 - 40 / 3)
            return self.Atmosphere[x] * (height / 6000 - (x + 40 / 3)) + self.Atmosphere[x + 1] * (
                    1 - (height - (x + 43 / 3) * 6000)) / 6000
        else:
            x = int(height / 5000 - 30)
            return self.Atmosphere[x] * (height / 5000 - (x + 30)) + self.Atmosphere[x + 1] * (
                        1 - (height - (x + 31) * 5000)) / 5000

    def calc(self, q, time):
        """
        calculate force
        :param q: satellite's coordinates and velocity
        :param time: current time
        :return: force
        """
        height = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2) ** 0.5 - 6.37e6
        rho = self.get_atmosphere(height)
        self.Force[0] = - 0.5 * self.Cx * self.Square * rho * (q[3] ** 2)
        self.Force[1] = - 0.5 * self.Cx * self.Square * rho * (q[4] ** 2)
        self.Force[2] = - 0.5 * self.Cx * self.Square * rho * (q[5] ** 2)

        return self.Force


class GravityForce(BaseForce):
    GM = 4e14
    mass = 0

    def __init__(self, mass):
        self.mass = mass

    def calc(self, q, time):
        r = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2) ** 0.5
        self.Force[0] = - self.GM * self.mass * q[0] / r ** 3
        self.Force[1] = - self.GM * self.mass * q[1] / r ** 3
        self.Force[2] = - self.GM * self.mass * q[2] / r ** 3
        return self.Force


class SunForce(BaseForce):
    """
    class sunlight pressure force
    """
    Square = 0
    Wc = 4.556e-3

    def __init__(self, Square):
        """
        :param Square: satellite's square
        """
        self.Square = Square

    def calc(self, q, time):
        """
        calculate force
        :param q: satellite's coordinates and velocity
        :param time: current time
        :return: force
        """
        sun = get_sun(time).obsgeoloc.xyz
        r = ((q[0] - sun[0].value) ** 2 + (q[1] - sun[1].value) ** 2 + (q[2] - sun[2].value) ** 2) ** 0.5
        self.Force[0] = self.Wc * self.Square * (q[0] - sun[0].value) / r
        self.Force[1] = self.Wc * self.Square * (q[1] - sun[1].value) / r
        self.Force[2] = self.Wc * self.Square * (q[2] - sun[2].value) / r
        return self.Force


class OtherForce(BaseForce):
    def calc(self, *args):
        return self.Force


class TestForce(BaseForce):
    """Force for testing integrators"""

    def calc(self, q, time):
        self.Force[0] = - 15 * q[0] - 0.1 * q[3]
        self.Force[1] = 0
        self.Force[2] = 0
        return self.Force
