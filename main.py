from forces import ResistForce, GravityForce, SunForce, TestForce
from IntegrateMethods import RK4Method, EulerMethod1, EulerMethod2, DormandPrinceMethod
from astropy.time import Time
import numpy as np
import graphics


mass = 1
square = 0.2
cx = 1
time = Time('1999-01-01T00:00:00.123456789', format='isot', scale='utc')

q = [np.array([6700000, 0, 1000000, 0, 6000, 4000])]

air_force = ResistForce(square, cx)
g_force = GravityForce(mass)
sun_force = SunForce(square)
test_force = TestForce()

# testing

delta_t = 0.1
Forces = [g_force, air_force]
integrator1 = RK4Method(delta_t, mass, Forces)
integrator2 = DormandPrinceMethod(delta_t, mass, Forces)
integrator3 = EulerMethod1(delta_t, mass, Forces)
integrator4 = EulerMethod2(delta_t, mass, Forces)

integrator = integrator1   # you can change the method here

clock = 0
mas_t = np.array([0])

while mas_t[-1] < 1000:
    q.append(integrator.calc_next_step(q[-1], time))
    mas_t = np.append(mas_t, clock)
    clock += integrator.dt

graphics.animation_map(q, 100)
