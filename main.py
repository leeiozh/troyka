from forces import ResistForce, GravityForce, CenterForce, SunForce, OtherForce
from astropy.time import Time
import numpy as np

mass = 1000
square = 0.2
cx = 2
time = Time('1999-01-01T00:00:00.123456789', format='isot', scale='utc')
q = np.array([7000000, 0, 0, 0, 7000, 0])

air_force = ResistForce(square, cx)
print(air_force.calc(q, time))

g_force = GravityForce(mass)
print(g_force.calc(q, time))

c_force = CenterForce(mass)
print(c_force.calc(q, time))

sun_force = SunForce(square)
print(sun_force.calc(q, time))
