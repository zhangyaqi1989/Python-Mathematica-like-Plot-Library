#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
"""
A Python library that mimics Mathematica plot functionality
"""

# 3rd party library
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sympy import lambdify
from sympy.parsing.sympy_parser import parse_expr

sns.set()


def _lambdify_func(parameters, expression):
    """
    make a vectorized function based on expression

    Args:
        parameters: parameter list, e.g. ['u', 'v']
        expression: expression string, e.g. 'sin(u) + cos(u)'

    Returns:
        a vectorized function
    """
    func = lambdify([parse_expr(para, evaluate=False) for para in parameters],
                    parse_expr(expression, evaluate=False))
    vec_func = np.vectorize(func)
    return vec_func


def ParametricPlot3D(expression, u_specs, v_specs, n_points=100):
    """
    mimic ParametricPlot3D() in Mathematica

    Args:
        expression: parametric expressions
        u_specs: first parameter and its min and max
        v_specs: second parameter and its min and max
        n_points: number of sample points in each axis

    Returns:
        an axes object

    Example:
        expression = ['cos(u)', 'sin(u) + cos(v)', 'sin(v)']
        ParametricPlot3D(expression, ['u', 0, 2*np.pi], ['v', -np.pi, np.pi])
    """
    u, u_min, u_max = u_specs
    v, v_min, v_max = v_specs
    u_angles = np.linspace(u_min, u_max, n_points)
    v_angles = np.linspace(v_min, v_max, n_points)
    us, vs = np.meshgrid(u_angles, v_angles)
    vec_x_func = _lambdify_func([u, v], expression[0])
    vec_y_func = _lambdify_func([u, v], expression[1])
    vec_z_func = _lambdify_func([u, v], expression[2])
    xs = vec_x_func(us, vs)
    ys = vec_y_func(us, vs)
    zs = vec_z_func(us, vs)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xs, ys, zs, color='orange')
    return ax


def Plot3D(expression, x_specs, y_specs, n_points=100):
    """
    mimic Plot3D() function in Mathematica

    Args:
        expression: parametric expressions
        x_specs: first parameter and its min and max
        y_specs: second parameter and its min and max
        n_points: number of sample points in each axis

    Returns:
        an axes object

    Example:
        Plot3D("sin(x + y**2)", ['x', -3, 3], ['y', -2, 2])
    """
    x, x_min, x_max = x_specs
    y, y_min, y_max = y_specs
    xs = np.linspace(x_min, x_max, n_points)
    ys = np.linspace(y_min, y_max, n_points)
    func = _lambdify_func([x, y], expression)
    X, Y = np.meshgrid(xs, ys)
    vec_func = np.vectorize(func)
    Z = vec_func(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, color='orange')
    return ax


def ListLinePlot(lst):
    """
    mimic ListLinePlot() in Mathematica

    Args:
        lst: list of values

    Returns:
        an axes object

    Example:
        ListLinePlot([1, 1, 2, 3, 5, 8])
    """
    fig, ax = plt.subplots()
    ax.plot(lst)
    return ax


def ContourPlot(expression, arg_specs):
    """
    mimic ContourPlot() in Mathematica
    """
    pass


def PolarPlot(expressions, arg_specs, legends=False, n_points=100):
    """
    mimic PolarPlot() in Mathematica

    Args:
        expressions: expression string list
        arg_specs: argument string and its min and max

    Returns:
        an axes object

    Example:
        PolarPlot(["sin(t)"], ['t', 0, 2*np.pi], legends=True)
    """
    arg, arg_min, arg_max = arg_specs
    vec_funcs = (_lambdify_func(arg, expression) for expression in expressions)
    ts = np.linspace(arg_min, arg_max, n_points)
    fig, ax = plt.subplots()
    for i, vec_func in enumerate(vec_funcs):
        rs = vec_func(ts)
        xs = rs * np.cos(ts)
        ys = rs * np.sin(ts)
        ax.plot(xs, ys, label=expressions[i])
    if legends:
        ax.legend(loc="best")
    return ax


def ParametricPlot(expressions, arg_specs, legends=False, n_points=100):
    """
    mimic ParametricPlot() in Mathematica

    Args:
        expressions: expression string list
        arg_specs: argument string and its min and max

    Returns:
        an axes object

    Example: ParametricPlot([["cos(u)", "sin(u)"]],
       ["u", 0, 2*np.pi], legends=True)
    """
    arg, arg_min, arg_max = arg_specs
    vec_x_funcs = [_lambdify_func(arg, x_expression) for (x_expression, _) in
                   expressions]
    vec_y_funcs = [_lambdify_func(arg, y_expression) for (_, y_expression) in
                   expressions]
    ts = np.linspace(arg_min, arg_max, n_points)
    fig, ax = plt.subplots()
    for i, (vec_x_func, vec_y_func) in enumerate(
            zip(vec_x_funcs, vec_y_funcs)):
        xs = vec_x_func(ts)
        ys = vec_y_func(ts)
        ax.plot(xs, ys, label=expressions[i])
    if legends:
        ax.legend(loc="best")
    return ax


def Plot(expressions, arg_specs, legends=False, n_points=100):
    """
    mimic Plot() function of Mathematica

    Args:
        expressions: expression string list
        arg_specs: argument string and its min and max

    Returns:
        an axes object

    Example: Plot(["sin(x)", "cos(x)"], ['x', 0, 2*np.pi], legends=True)
    """
    arg, arg_min, arg_max = arg_specs
    vec_funcs = [_lambdify_func(arg, expression) for expression in expressions]
    xs = np.linspace(arg_min, arg_max, n_points)
    fig, ax = plt.subplots()
    for i, vec_func in enumerate(vec_funcs):
        ys = vec_func(xs)
        ax.plot(xs, ys, label=expressions[i])
    if legends:
        ax.legend(loc="best")
    return ax


if __name__ == "__main__":
    print("Hello Mathematica Plot")
