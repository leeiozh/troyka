from forces import ResistForce, GravityForce, SunForce, OtherForce, TestForce
from IntegrateMethods import RK4Method, EulerMethod1, EulerMethod2
from astropy.time import Time
import numpy as np
import graphics
import matplotlib.pyplot as plt

mass = 1000
square = 0.2
cx = 2
time = Time('1999-01-01T00:00:00.123456789', format='isot', scale='utc')
q = np.array([6500000, 0, 0, 0, 6000, 0])

air_force = ResistForce(square, cx)
print(air_force.calc(q, time))
g_force = GravityForce(mass)
print(g_force.calc(q, time))

sun_force = SunForce(square)
print(sun_force.calc(q, time))

test_force = TestForce()

# testing

delta_t = 1
Forces = [g_force]
integrator1 = RK4Method(delta_t, mass, Forces)
integrator2 = EulerMethod1(delta_t, mass, Forces)
integrator3 = EulerMethod2(delta_t, mass, Forces)

mas_x = np.array([])
mas_y = np.array([])
mas_v = np.array([])
mas_t = np.arange(30000) * delta_t

for i in range(30000):
    mas_x = np.append(mas_x, q[0])
    mas_y = np.append(mas_y, q[1])
    mas_v = np.append(mas_v, q[3])
    q = integrator1.calc_next_step(q, 0)  # you can change the method here


graphics.animation(mas_x / 50000, mas_y / 50000, delta_t)
