import numpy as np
from abc import abstractmethod


class Integrator:
    """
    integrator's base class
    """

    def __init__(self, dt, m, forces):
        """
        init function
        :param dt: time step
        :param m: mass
        :param forces: type of forces
        """
        self.m = m
        self.dt = dt
        self.forces = forces
        self.b = np.array([])
        self.c = np.array([])
        self.k1 = np.array([])
        self.k2 = np.array([])
        self.k = np.array([[0, 0, 0, 0, 0, 0]])

    @abstractmethod
    def calc_next_step(self, state, time):
        """
        Return state after dt
        :param state: (x, y, z, vx, vy, vz)
        :param time:
        :return: (x, y, z, vx, vy, vz)
        """

    def calc_resultant_force(self, state, time):
        """
        calculate resultant force
        :param state: current q
        :param time: time
        :return:
        """
        resultant = np.array([0, 0, 0])
        for count in range(len(self.forces)):
            force = self.forces[count].calc(state, time)
            for j in range(3):
                resultant[j] += force[j]
        return resultant

    def evaluate(self, state, derivative, dt, time):
        """
        Help function for calc_next_step, computers and returns a massive of derivatives after dt
        :param state: (x, y, z, vx, vy, vz)
        :param derivative: (vx, vy, vz, ax, ay, az)
        :param dt:
        :param time: current time
        :return: (vx, vy, vz, ax, ay, az)
        """

        new_state = state + dt * derivative

        forces = self.calc_resultant_force(new_state, time + dt)
        new_derivative = np.array([new_state[3], new_state[4], new_state[5], forces[0] / self.m,
                                   forces[1] / self.m, forces[2] / self.m])
        return new_derivative


class EulerMethod1(Integrator):

    def calc_next_step(self, state, time):
        """
        Return state after dt
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


# ?????????????????????????????? ?????????? ????????????, ?????????? ???? ???? ??????????, ???? ???????????????????? ?????????????????? ?????????? ???????????????????? ??????????????????
class EulerMethod2(Integrator):

    def calc_next_step(self, state, time):
        """
        Return state after dt
        :param state: (x, y, z, vx, vy, vz)
        :param time:
        :return: (x, y, z, vx, vy, vz)
        """
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

    b = np.array([[], [0.5], [0, 0.5], [0, 0, 1]], dtype=object)
    c = np.array([0, 0.5, 0.5, 1])
    k1 = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])

    def calc_next_step(self, state, time):
        """
        Return state after dt
        :param state: (x, y, z, vx, vy, vz)
        :param time: ?????????????? ??????????
        :return: (x, y, z, vx, vy, vz)
        """

        self.k = np.array([np.zeros(6)])
        forces = self.calc_resultant_force(state, time)

        derivative = np.array([state[3], state[4], state[5], forces[0] / self.m, forces[1] / self.m,
                               forces[2] / self.m])
        self.k[0] = derivative
        for i in range(1, len(RK4Method.c)):
            new_k = self.evaluate(state, RK4Method.b[i] @ self.k, RK4Method.c[i] * self.dt, time)
            self.k = np.append(self.k, [new_k],
                               axis=0)

        new_derivative = RK4Method.k1 @ self.k

        new_state = state + self.dt * new_derivative

        return new_state


class DormandPrinceMethod(Integrator):

    b = np.array([[],
                  [1/5], [3/40, 9/40], [44/45, -56/15, 32/9],
                  [19372/6561, -25360/2187, 64448/6561, -212/729],
                  [9017/3168, -355/33, -46732/5247, 49/176, -5103/18656],
                  [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]], dtype=object)
    k1 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    k2 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

    def calc_next_step(self, state, time):
        """
        Return state after dt
        :param state: (x, y, z, vx, vy, vz)
        :param time: ?????????????? ??????????
        :return: (x, y, z, vx, vy, vz)
        """
        self.k = np.array([np.zeros(6)])
        forces = self.calc_resultant_force(state, time)

        derivative = np.array([state[3], state[4], state[5], forces[0] / self.m, forces[1] / self.m,
                               forces[2] / self.m])
        self.k[0] = derivative
        for i in range(1, len(DormandPrinceMethod.c)):
            new_k = self.evaluate(state, np.dot(DormandPrinceMethod.b[i], self.k), DormandPrinceMethod.c[i] * self.dt,
                                  time)
            self.k = np.append(self.k, [new_k],
                               axis=0)

        # calculating error
        next_step1 = state + self.dt * DormandPrinceMethod.k1 @ self.k
        next_step2 = state + self.dt * DormandPrinceMethod.k2 @ self.k

        error = np.abs(next_step1[0] - next_step2[0])
        eps = 1
        # s = (eps * self.dt / 2 / error) ** 0.2
        if error > eps:
            self.dt = 0.5 * self.dt
        if error < eps:
            self.dt = 2 * self.dt

        return next_step1
