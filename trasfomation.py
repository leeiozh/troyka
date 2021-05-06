import numpy as np


def to_polar(q):
    """
    converter from xyz GCRS to polar coordinates
    :param q: satellite's coordinates in GCRS
    :return: satellite's coordinates in polar system
    """

    theta = np.arcsin(q[2] / np.linalg.norm(q[:3]))

    if q[0] > 0:
        phi = np.arctan(q[1] / q[0])
    elif q[0] < 0 <= q[1]:
        phi = np.arctan(q[1] / q[0]) + np.pi
    elif q[0] < 0 and q[1] < 0:
        phi = np.arctan(q[1] / q[0]) - np.pi
    elif q[0] == 0 and q[1] > 0:
        phi = 0.5 * np.pi
    elif q[0] == 0 and q[1] < 0:
        phi = -0.5 * np.pi
    else: phi = 0

    return np.array([phi * 180 / np.pi, theta * 180 / np.pi])


def to_kepler(q):
    """
    converter from xyz GCRS coordinates to Keplerian parameters
    :param q: satellite's coordinates in GCRS
    :return: satellite's coordinates in Kepler
    """

    mu = 3.986004e+14

    c1 = q[1] * q[5] - q[4] * q[2]
    c2 = q[3] * q[2] - q[0] * q[5]
    c3 = q[0] * q[4] - q[3] * q[1]

    c = np.sqrt(c1 ** 2 + c2 ** 2 + c3 ** 2)

    p = c ** 2 / mu

    i = np.arccos(c3 / c)

    cos_Omega = - c2 / c / np.sin(i)
    Omega = np.arcsin(c1 / c / np.sin(i))
    if cos_Omega < 0:
        Omega = 180 - Omega

    v2 = np.linalg.norm(q[3:]) ** 2

    r = np.linalg.norm(q[:3])

    h = v2 - 2 * mu / r

    a = - mu / h

    D = q[0] * q[3] + q[1] * q[4] + q[2] * q[5]
    Dd = mu / r + h

    f1 = Dd * q[0] - D * q[3]
    f2 = Dd * q[1] - D * q[4]
    f3 = Dd * q[2] - D * q[5]

    f = np.linalg.norm(np.array([f1, f2, f3]))

    e = f / mu

    if np.abs(e) > 1:
        e = 1

    omega = np.arccos((f1 * np.cos(Omega) + f2 * np.sin(Omega)) / f)
    sin_omega = np.arcsin(f3 / f / np.sin(i))
    if sin_omega < 0:
        omega *= -1

    if (p - r) / e / r > 1:
        theta = np.arccos(1)
    elif (p - r) / e / r < -1:
        theta = np.arccos(-1)
    else:
        theta = np.arccos((p - r) / e / r)

    sin_theta = D * np.sqrt(p / mu) / e / r
    if sin_theta < 0:
        theta *= -1

    if (e + np.cos(theta)) / (1 + e * np.cos(theta)) > 1:
        E = np.arccos(1)
    elif (e + np.cos(theta)) / (1 + e * np.cos(theta)) < -1:
        E = np.arccos(-1)
    else:
        E = np.arccos((e + np.cos(theta)) / (1 + e * np.cos(theta)))

    sin_E = np.sqrt(1 - e ** 2) * np.sin(theta) / (1 + e * np.cos(theta))
    if sin_E < 0:
        E *= -1

    M = E - e * sin_E

    return np.array([a, e, i, Omega, omega, M])


def print_kepler(q):
    """
    print keplerian coordinates
    :param q:
    :return:
    """
    print("Semimajor axis a =", q[0] / 1000, "km")
    print("Eccentricity e =", q[1])
    print("Inclination i =",  q[2] / np.pi * 180)
    print("Longitude of the ascending node O =", q[3] / np.pi * 180)
    print("Argument of periapsis o =", q[4] / np.pi * 180)
    print("True anomaly th =", q[5] / np.pi * 180)