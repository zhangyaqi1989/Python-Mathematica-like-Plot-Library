#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
# A Mathematica like plot
##################################

from sympy import lambdify
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def Plot3D(expression, x_specs, y_specs):
    """ mimic Plot3D() function of Mathematica
        Usage: Plot3D("sin(x + y**2)", ['x', -3, 3], ['y', -2, 2])
    """
    x, x_min, x_max = x_specs
    y, y_min, y_max = y_specs
    n_points = 100
    xs = np.linspace(x_min, x_max, n_points)
    ys = np.linspace(y_min, y_max, n_points)
    func = lambdify([parse_expr(x, evaluate=False), parse_expr(y, evaluate=False)],
            parse_expr(expression, evaluate=False))
    X, Y = np.meshgrid(xs, ys)
    vec_func = np.vectorize(func)
    Z = vec_func(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, color='orange')
    plt.show()


def Plot(expressions, arg_specs, legends=False):
    """ mimic Plot() function of Mathematica
        Usage: Plot(["sin(x)", "cos(x)"], ['x', 0, 2*np.pi], legends=True)
    """
    fig, ax = plt.subplots()
    n = len(expressions)
    funcs = [None] * n
    arg, arg_min, arg_max = arg_specs
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


def test_Plot():
    """test Plot()"""
    expressions = ['sin(x)', 'sin(2*x)', 'sin(3*x)']
    Plot(expressions, ['x', 0, 2*np.pi], legends=True)


def test_Plot3D():
    """test Plot3D()"""
    expression = 'sin(x + y**2)'
    Plot3D(expression, ['x', -3, 3], ['y', -2, 2])


if __name__ == "__main__":
    # test_Plot()
    test_Plot3D()
