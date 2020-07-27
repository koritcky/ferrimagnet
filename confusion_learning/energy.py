import numpy as np

mb = 9.27 * (10 ** -21)
Mf = 10 * mb
Md = 5 * mb
l = (10 ** 5) / mb
Kf = 2.55 * (10 ** -16)
# TODO: подобрать параметры Kd, Kf
Kd = 2.55 * (10 ** -16)

c = 10 ** 15  # Normalisation constant


def Heff(theta, h):
    return (h ** 2 + (l * Md) ** 2 - 2 * l * h * Md * np.cos(theta)) ** (1 / 2)


def dHeff(theta, h):
    """derivative of Heff"""
    return (l * Md * h * np.sin(theta)) / (Heff(theta, h))


def ddHeff(theta, h):
    """second derivative of Heff"""
    return (l * Md * h * np.cos(theta)) / (Heff(theta, h)) - \
           (l * Md * h * np.sin(theta) * dHeff(theta, h)) / (Heff(theta, h)) ** 2


def cos_f(theta, h):
    return (h - l * Md * np.cos(theta)) / Heff(theta, h)


def d_cos_f(theta, h):
    """derivative of cos_f"""
    return (l * Md * np.sin(theta)) / Heff(theta, h) - \
           (h - l * Md * np.cos(theta)) * dHeff(theta, h) / (Heff(theta, h) ** 2)


def dd_cos_f(theta, h):
    """second derivative of cos_f"""
    return l * Md * np.cos(theta) / Heff(theta, h) - \
           2 * l * Md * np.sin(theta) / (Heff(theta, h) ** 2) * dHeff(theta, h) + \
           (h - l * Md * np.cos(theta)) * \
           (2 * dHeff(theta, h) ** 2 / Heff(theta, h) ** 3 - ddHeff(theta, h) / (Heff(theta, h) ** 2))


def energy(theta, *args):
    x, h = args
    return (-x * Mf * Heff(theta, h) -
            Md * h * (1 - x) * np.cos(theta) -
            Kd * (1 - x) * np.cos(theta) ** 2 -
            Kf * x * cos_f(theta, h) ** 2) * c


def denergy(theta, *args):
    x, h = args
    return (-x * Mf * dHeff(theta, h) +
            Md * h * (1 - x) * np.sin(theta) +
            Kd * (1 - x) * np.sin(2 * theta) -
            Kf * x * 2 * cos_f(theta, h) * d_cos_f(theta, h)) * c

def ddenergy(theta, *args):
    x, h = args
    return (-x * Mf * ddHeff(theta, h) +
            Md * h * (1-x) * np.cos(theta) +
            2 * Kd * (1-x) * np.cos(2 * theta) -
            Kf * x * 2 * (d_cos_f(theta, h) ** 2 + cos_f(theta, h) * dd_cos_f(theta, h))) * c   