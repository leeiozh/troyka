import numpy as np
import matplotlib.pyplot as plt


class EulerMethod1:

    def __init__(self, dt, m):
        """
        Передаем шаг dt и массу m
        :param dt:
        :param m:
        """
        self.dt = dt
        self.m = m

    def integrate(self, state, derivative, time):
        """
        Возвращает состояние через шаг dt
        :param state: (x, y, z, vx, vy, vz)
        :param derivative: (vx, vy, vz, ax, ay, az)
        :param time:
        :return: (x, y, z, vx, vy, vz)
        """
        new_state = np.array([state[0] + derivative[0] * self.dt, state[1] + derivative[1] * self.dt,
                              state[2] + derivative[2] * self.dt, state[3] + derivative[3] * self.dt,
                              state[4] + derivative[4] * self.dt, state[5] + derivative[5] * self.dt])
        return new_state


# Симплектический метод Эйлера, почти то же самое, но координаты считаются после вычисления скоростей
class EulerMethod2:

    def __init__(self, dt, m):
        self.dt = dt
        self.m = m

    def integrate(self, state, derivative, time):
        new_vx = state[3] + derivative[3] * self.dt
        new_vy = state[4] + derivative[4] * self.dt
        new_vz = state[5] + derivative[5] * self.dt
        new_state = np.array([state[0] + new_vx * self.dt, state[1] + new_vy * self.dt, state[2] + new_vz * self.dt,
                              new_vx, new_vy, new_vz])
        return new_state


class RK4Method:

    def __init__(self, dt, m):
        """
        Передаем шаг dt и массу m
        :param dt:
        :param m:
        """
        self.dt = dt
        self.m = m

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
        new_state = np.append(new_state, state[0] + derivative[0] * dt)
        new_state = np.append(new_state, state[1] + derivative[1] * dt)
        new_state = np.append(new_state, state[2] + derivative[2] * dt)
        new_state = np.append(new_state, state[3] + derivative[3] * dt)
        new_state = np.append(new_state, state[4] + derivative[4] * dt)
        new_state = np.append(new_state, state[5] + derivative[5] * dt)

        # вычисление сил (в данном случае для осциллятора)
        new_forces = np.array([[- k * new_state[0] - new_state[3] * g, 0, 0]])

        forces = np.array([])
        for f in new_forces:
            for count in range(3):
                forces = np.append(forces, f[count])

        new_derivative = np.array([new_state[3], new_state[4], new_state[5], forces[0] / self.m,
                                   forces[1] / self.m, forces[2] / self.m])
        return new_derivative

    def integrate(self, state, derivative, time):
        """
        Вычисляет и возвращает состояние через шаг dt
        :param state: (x, y, z, vx, vy, vz)
        :param derivative: (vx, vy, vz, ax, ay, az)
        :param time: текущее время
        :return: (x, y, z, vx, vy, vz)
        """
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
k = 15  # жесткость пружины
g = 0.1  # коэффициент вязкого трения
mass = 1
delta_t = 0.01
state0 = np.array([100, 0, 0, 0, 0, 0])
f0 = - k * state0[0] - g * state0[3]
derivative0 = np.array([0, 0, 0, f0 / mass, 0, 0])
forces_ = np.array([[f0, 0, 0]])

integrator1 = RK4Method(delta_t, mass)
integrator2 = EulerMethod1(delta_t, mass)
integrator3 = EulerMethod2(delta_t, mass)

state1 = state0
derivative1 = derivative0
mas_x = np.array([])
mas_v = np.array([])
mas_t = np.arange(10000) * 0.01

for i in range(10000):
    mas_x = np.append(mas_x, state1[0])
    mas_v = np.append(mas_v, state1[3])
    state1 = integrator3.integrate(state1, derivative1, 0)  # здесь можно поменять метод интегрирования
    forces_ = np.array([[- k * state1[0] - state1[3] * g, 0, 0]])
    derivative1 = np.array([state1[3], state1[4], state1[5], forces_[0][0] / mass, forces_[0][1] / mass,
                           forces_[0][2] / mass])

plt.plot(mas_t, mas_x)
plt.show()
