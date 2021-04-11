from forces import ResistForce, GravityForce, SunForce, OtherForce, TestForce
from IntegrateMethods import RK4Method, EulerMethod1, EulerMethod2
from astropy.time import Time
import numpy as np
import graphics
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mass = 1000
square = 0.2
cx = 2
time = Time('1999-01-01T00:00:00.123456789', format='isot', scale='utc')
q = np.array([6600000, 0, 0, 0, 5000, 0])

air_force = ResistForce(square, cx)
g_force = GravityForce(mass)
sun_force = SunForce(square)
test_force = TestForce()

# testing

delta_t = 1
Forces = [g_force, air_force]
integrator1 = RK4Method(delta_t, mass, Forces)
integrator2 = EulerMethod1(delta_t, mass, Forces)
integrator3 = EulerMethod2(delta_t, mass, Forces)

mas_x = np.array([])
mas_y = np.array([])
mas_z = np.array([])
mas_vx = np.array([])
mas_vy = np.array([])
mas_vz = np.array([])
mas_t = np.arange(30000) * delta_t

for i in range(30000):
    mas_x = np.append(mas_x, q[0])
    mas_y = np.append(mas_y, q[1])
    mas_z = np.append(mas_z, q[2])
    mas_vx = np.append(mas_vx, q[3])
    mas_vy = np.append(mas_vx, q[4])
    mas_vz = np.append(mas_vx, q[5])
    q = integrator1.calc_next_step(q, 0)  # you can change the method here

# graphics.animation(mas_x / 50000, mas_y / 50000, delta_t)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


def anim(t):
    ax.clear()
    ax.plot(mas_x, mas_y, mas_z, color='red')
    ax.scatter(mas_x[t], mas_y[t], mas_z[t], color='blue')
    ax.scatter(0, 0, 0, color='green')


ani = FuncAnimation(fig, anim, frames=len(mas_x) - 1)
plt.show()

# ani.save('3D.gif', writer='imagemagick')
