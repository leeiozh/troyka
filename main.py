from forces import ResistForce, GravityForce, SunForce, TestForce
from IntegrateMethods import RK4Method, EulerMethod1, EulerMethod2, DormandPrinceMethod
from astropy.time import Time
import numpy as np
import pygame
import graphics


mass = 1
square = 0.2
cx = 1
time = Time('1999-01-01T00:00:00.123456789', format='isot', scale='utc')


air_force = ResistForce(square, cx)
g_force = GravityForce(mass)
sun_force = SunForce(square)
test_force = TestForce()

# testing

delta_t = 10
Forces = [g_force]
integrator1 = RK4Method(delta_t, mass, Forces)
integrator2 = DormandPrinceMethod(delta_t, mass, Forces)
integrator3 = EulerMethod1(delta_t, mass, Forces)
integrator4 = EulerMethod2(delta_t, mass, Forces)

integrator = integrator1   # you can change the method here

pygame.init()
screen = pygame.display.set_mode((1200, 800))
menu = graphics.Menu(screen)
choice = menu.run()
if choice == 1:
    load = graphics.LoadingWindow(screen, integrator)
    a = load.run()
    if a:
        anim = graphics.Animation(load.mas_t, load.mas_x, load.position, screen, delta_t)
        anim.run()
pygame.quit()
