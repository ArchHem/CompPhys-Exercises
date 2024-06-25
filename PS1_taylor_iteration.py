import numpy as np
import matplotlib.pyplot as plt
import warnings

def inverse_approximator(x,g,d_stop = 10e-6, maxloops = 400):
    """
    Taylor expansion of 1/x around 1, i.e. 1/(x+g) = 1 - g/x + g^2/x^2 -g^3/x^3... etc.

    Uses iteration of the first order to handle the work
    :param x: expansion point
    :param g: distance from x
    :param d_stop: relative change stopping condition, value set to 10e-6
    :param maxloops: maximum number of iterative loops
    :return: estimate of said expansion
    """

    starting_value = 1.0/x
    g_current = starting_value
    """begin iterating"""
    relative_error = 1.0 #can be set to anything: will be overwritten
    iteration_counter = 0


    while relative_error > d_stop and maxloops > iteration_counter:
        iteration_counter = iteration_counter + 1
        g_current_plus = 1/x - g/x * g_current
        """current relative change"""
        relative_error = np.abs((g_current_plus-g_current) / g_current_plus)
        g_current = g_current_plus

    if iteration_counter >= maxloops:
        warnings.warn('Maximum number of iterations reached: result may significantly differ from exact result!')

    return [g_current, iteration_counter]

print(inverse_approximator(1,0.99),1/(1+0.99))