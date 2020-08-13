# This module helps finding best w-shape by given  data

import numpy as np


def th_wshape(a, b, c, n_dots):
    """
    Draws theoretical w-shape by given starting, ending and transition point
    a: start point
    b: end pint
    c: transition point
    n_dots: number of dots for the w-shape
    """
    X = np.linspace(a, b, n_dots)

    Y_left = 1 - (np.minimum(c - X, X - a)) / (b - a)
    Y_right = 1 - (np.minimum(b - X, X - c)) / (b - a)

    return np.minimum(Y_left, Y_right)


def w_find(a, b, w_shape):
    n_dots = len(w_shape)
    min_mse = 10 ** 10
    best_c = a
    for c in np.linspace(a, b, n_dots):
        theory = th_wshape(a, b, c, n_dots)
        mse = ((w_shape - theory) ** 2).mean()
        if mse < min_mse:
            min_mse = mse
            best_c = c

    return th_wshape(a, b, best_c, n_dots), best_c




