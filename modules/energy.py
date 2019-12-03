import numpy as np

mb = 9.27 * (10 ** -21)
Mf = 10 * mb
Md = 5 * mb
l = (10 ** 4)/mb
Kf = 1.4 * (10 ** -16)
#TODO: подобрать параметры Kd, Kf
Kd = 1.4 * (10 ** -16)


def Heff (theta, h):
    return (h ** 2 + (l * Md) ** 2 - 2 * l * Md * np.cos(theta)) ** (1/2)


def dHeff(theta, h):
    return (l * Md * np.sin(theta))/(Heff(theta, h))


def cos_theta_f(theta, h):
    return (h - l * Md * np.cos(theta))/Heff(theta, h)


def energy(theta, *args):
    x, h = args
    return (-x * Mf * Heff(theta, h) -
           Md * h * (1-x) * np.cos(theta) -
           Kd * (1-x) * np.cos(theta) ** 2 -
           Kf * x * cos_theta_f(theta, h) ** 2) * 10 ** 14


def denergy(theta, *args):
    x, h = args
