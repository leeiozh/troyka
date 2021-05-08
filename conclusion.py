import pygame
from trasfomation import print_kepler, to_kepler, to_polar

pygame.init()

white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
X = 400
Y = 400

display_surface = pygame.display.set_mode((X, Y))
pygame.display.set_caption('Show Text')
font = pygame.font.Font('freesansbold.ttf', 32)

text_1 = font.render("Semimajor axis a =" + q[0] / 1000 + "km", True, green, blue)
text_1Rect = text_1.get_rect()
text_1Rect.center = (X // 3, Y // 6)

text_2 = font.render("Eccentricity e =" + q[1], True, green, blue)
text_2Rect = text_2.get_rect()
text_2Rect.center = (X // 3, Y // 3)

text_3 = font.render("Inclination i =" + q[2] / np.pi * 180, True, green, blue)
text_3Rect = text_3.get_rect()
text_3Rect.center = (X // 3, Y // 2)

text_4 = font.render("Longitude of the ascending node O =" + q[3] / np.pi * 180, True, green, blue)
text_4Rect = text_4.get_rect()
text_4Rect.center = (X // 3, 2 * Y // 3)

text_5 = font.render("Argument of periapsis o =" + q[4] / np.pi * 180, True, green, blue)
text_5Rect = text_5.get_rect()
text_5Rect.center = (X // 3, 5 * Y // 6)

text_6 = font.render("True anomaly th =" + q[5] / np.pi * 180, True, green, blue)
text_6Rect = text_6.get_rect()
text_6Rect.center = (X // 3, Y)

while True:
    display_surface.fill(white)
    display_surface.blit(text, textRect)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        pygame.display.update()
