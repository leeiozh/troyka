from forces import ResistForce, GravityForce, SunForce, TestForce
from IntegrateMethods import RK4Method, EulerMethod1, EulerMethod2, DormandPrinceMethod
from astropy.time import Time
import numpy as np
import graphics


mass = 1
square = 0.2
cx = 2
time = Time('1999-01-01T00:00:00.123456789', format='isot', scale='utc')
q = np.array([5000000, 0, 0, 0, 7800, 0])

air_force = ResistForce(square, cx)
g_force = GravityForce(mass)
sun_force = SunForce(square)
test_force = TestForce()

# testing

delta_t = 1
Forces = [g_force]
integrator1 = RK4Method(delta_t, mass, Forces)
integrator2 = DormandPrinceMethod(delta_t, mass, Forces)
integrator3 = EulerMethod1(delta_t, mass, Forces)
integrator4 = EulerMethod2(delta_t, mass, Forces)

integrator = integrator1   # you can change the method here

position = np.array([[q[0], q[1], q[2]]], dtype=object)

mas_x = np.array([])
mas_vx = np.array([])
mas_vy = np.array([])
mas_vz = np.array([])
clock = 0
mas_t = np.array([0])
mas_dt = np.array([])

while max(mas_t) < 10000:
    pos = np.array([[q[0] / 25000, q[1] / 25000, q[2] / 25000]])
    position = np.append(position, pos, axis=0)
    # print(pos)
    mas_x = np.append(mas_x, q[0])
    mas_vx = np.append(mas_vx, q[3])
    mas_vy = np.append(mas_vx, q[4])
    mas_vz = np.append(mas_vx, q[5])
    clock += integrator.dt
    mas_t = np.append(mas_t, clock)
    mas_dt = np.append(mas_dt, integrator.dt)
    q = integrator.calc_next_step(q, 0)


graphics.animation(mas_t, mas_x, position, delta_t)
