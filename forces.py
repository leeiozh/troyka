from abc import ABC, abstractmethod
from astropy.coordinates import get_sun
import numpy as np


class BaseForce(ABC):
    Force = np.zeros(3, float)

    @abstractmethod
    def calc(self, *args):
        pass


class ResistForce(BaseForce):
    Atmosphere = np.array([1.22500, 1.21913, 1.21328, 1.20746, 1.20165, 1.19587, 1.19011, 1.18437, 1.17865, 1.17295])
    Square = 0
    Cx = 0

    def __init__(self, Square, Cx):
        self.Square = Square
        self.Cx = Cx

    def get_atmosphere(self, q):
        height = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2) ** 0.5 // 10000
        return self.Atmosphere[height]

    def calc(self, q):
        rho = self.get_atmosphere(self, q)
        self.Force[0] = - 0.5 * self.Cx * self.Square * rho * (q[3] ** 2)
        self.Force[1] = - 0.5 * self.Cx * self.Square * rho * (q[4] ** 2)
        self.Force[2] = - 0.5 * self.Cx * self.Square * rho * (q[5] ** 2)

        return self.Force


class GravityForce(BaseForce):
    GM = 4e14

    def calc(self, q):
        r = (q[0] ** 2 + q[1] ** 2 + q[2] ** 3) ** 0.5
        self.Force[0] = - self.GM * q[0] / r ** 3
        self.Force[1] = - self.GM * q[1] / r ** 3
        self.Force[2] = - self.GM * q[2] / r ** 3
        return self.Force


class SunForce(BaseForce):
    Square = 0
    Wc = 4.556e-3

    def __init__(self, Square):
        self.Square = Square

    def calc(self, q, time):
        sun = get_sun(time).obsgeoloc.xyz
        r = ((q[0] - sun[0].value) ** 2 + (q[1] - sun[1].value) ** 2 + (q[2] - sun[2].value) ** 2) ** 0.5
        self.Force[0] = self.Wc * self.Square * (q[0] - sun[0].value) / r
        self.Force[1] = self.Wc * self.Square * (q[1] - sun[1].value) / r
        self.Force[2] = self.Wc * self.Square * (q[2] - sun[2].value) / r
        return self.Force


class OtherForce(BaseForce):
    def calc(self, *args):
        return self.Force
