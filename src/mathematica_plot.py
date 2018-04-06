#!/usr/bin/env python3
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
# A Mathematica like plot
##################################

from sympy import lambdify
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import matplotlib.pyplot as plt


def Plot(expressions, arg, arg_min, arg_max, legends=False):
    """ A mathematica like plot function
        Usage: Plot(["sin(x)", "cos(x)"], 'x', 0, 2*np.pi, legends=True)
    """
    fig, ax = plt.subplots()
    n = len(expressions)
    funcs = [None] * n
    for i, expression in enumerate(expressions):
        funcs[i] = lambdify(parse_expr(arg, evaluate=False), \
                parse_expr(expression, evaluate=False))
    n_points = 100
    xs = np.linspace(arg_min, arg_max, n_points)
    for i, func in enumerate(funcs):
        func_v = np.vectorize(func)
        ys = func_v(xs)
        ax.plot(xs, ys, label=expressions[i])
    if legends:
        ax.legend(loc="best")
    plt.show()


def test():
    expressions = ['sin(x)', 'sin(2*x)', 'sin(3*x)']
    Plot(expressions, 'x', 0, 2*np.pi, legends=True)


if __name__ == "__main__":
    test()
