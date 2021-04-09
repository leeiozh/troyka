import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from forces import TestForce


class Integrator:

    def __init__(self, dt, m, forces):
        self.m = m
        self.dt = dt
        self.forces = forces

    @abstractmethod
    def calc_next_step(self, state, time):
        """
        Возвращает состояние через шаг dt
        :param state: (x, y, z, vx, vy, vz)
        :param time:
        :return: (x, y, z, vx, vy, vz)
        """

    def calc_resultant_force(self, state, time):
        resultant = np.array([0, 0, 0])
        for count in range(len(self.forces)):
            force = self.forces[count].calc(state, time)
            for j in range(3):
                resultant[j] += force[j]
        return resultant


class EulerMethod1(Integrator):

    def calc_next_step(self, state, time):
        """
        Возвращает состояние через шаг dt
        :param state: (x, y, z, vx, vy, vz)
        :param time:
        :return: (x, y, z, vx, vy, vz)
        """
        forces = self.calc_resultant_force(state, time)
        derivative = np.array([state[3], state[4], state[5], forces[0] / self.m, forces[1] / self.m,
                               forces[2] / self.m])
        new_state = np.array([state[0] + derivative[0] * self.dt, state[1] + derivative[1] * self.dt,
                              state[2] + derivative[2] * self.dt, state[3] + derivative[3] * self.dt,
                              state[4] + derivative[4] * self.dt, state[5] + derivative[5] * self.dt])
        return new_state


# Симплектический метод Эйлера, почти то же самое, но координаты считаются после вычисления скоростей
class EulerMethod2(Integrator):

    def calc_next_step(self, state, time):
        forces = self.calc_resultant_force(state, time)
        derivative = np.array([state[3], state[4], state[5], forces[0] / self.m, forces[1] / self.m,
                               forces[2] / self.m])
        new_vx = state[3] + derivative[3] * self.dt
        new_vy = state[4] + derivative[4] * self.dt
        new_vz = state[5] + derivative[5] * self.dt
        new_state = np.array([state[0] + new_vx * self.dt, state[1] + new_vy * self.dt, state[2] + new_vz * self.dt,
                              new_vx, new_vy, new_vz])
        return new_state


class RK4Method(Integrator):

    def evaluate(self, state, derivative, dt, time):
        """
        Вспомогательная функция для integrate, вычисляет и возвращает массив производных через время dt,
        здесь же происходит вызов сил
        :param state: (x, y, z, vx, vy, vz)
        :param derivative: (vx, vy, vz, ax, ay, az)
        :param dt:
        :param time: текущее время
        :return: (vx, vy, vz, ax, ay, az)
        """
        new_state = np.array([])
        for count in range(6):
            new_state = np.append(new_state, state[count] + derivative[count] * dt)

        forces = self.calc_resultant_force(new_state, time)

        new_derivative = np.array([new_state[3], new_state[4], new_state[5], forces[0] / self.m,
                                   forces[1] / self.m, forces[2] / self.m])
        return new_derivative

    def calc_next_step(self, state, time):
        """
        Вычисляет и возвращает состояние через шаг dt
        :param state: (x, y, z, vx, vy, vz)
        :param time: текущее время
        :return: (x, y, z, vx, vy, vz)
        """
        forces = self.calc_resultant_force(state, time)
        derivative = np.array([state[3], state[4], state[5], forces[0] / self.m, forces[1] / self.m,
                               forces[2] / self.m])
        a = self.evaluate(state, derivative, 0, time)
        b = self.evaluate(state, a, 0.5 * self.dt, time)
        c = self.evaluate(state, b, 0.5 * self.dt, time)
        d = self.evaluate(state, c, self.dt, time)

        vx = (a[0] + 2 * (b[0] + c[0]) + d[0]) / 6
        vy = (a[1] + 2 * (b[1] + c[1]) + d[1]) / 6
        vz = (a[2] + 2 * (b[2] + c[2]) + d[2]) / 6

        ax = (a[3] + 2 * (b[3] + c[3]) + d[3]) / 6
        ay = (a[4] + 2 * (b[4] + c[4]) + d[4]) / 6
        az = (a[5] + 2 * (b[5] + c[5]) + d[5]) / 6

        x = state[0] + vx * self.dt
        y = state[1] + vy * self.dt
        z = state[2] + vz * self.dt

        vx = state[3] + ax * self.dt
        vy = state[4] + ay * self.dt
        vz = state[5] + az * self.dt

        return np.array([x, y, z, vx, vy, vz])


# тестирую интеграторы на гармоничесом осцилляторе с затуханием (можно запустить)

mass = 1
delta_t = 0.01
state0 = np.array([100, 0, 0, 0, 0, 0])
force1 = TestForce()
Forces = [force1]
integrator1 = RK4Method(delta_t, mass, Forces)
integrator2 = EulerMethod1(delta_t, mass, Forces)
integrator3 = EulerMethod2(delta_t, mass, Forces)

state1 = state0
mas_x = np.array([])
mas_v = np.array([])
mas_t = np.arange(10000) * 0.01

for i in range(10000):
    mas_x = np.append(mas_x, state1[0])
    mas_v = np.append(mas_v, state1[3])
    state1 = integrator1.calc_next_step(state1, 0)  # здесь можно поменять метод интегрирования

plt.plot(mas_t, mas_x)
plt.show()
