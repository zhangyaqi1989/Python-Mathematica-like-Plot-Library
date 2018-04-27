#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
# A Python library that mimics
# Mathematica plot functionality
##################################

from sympy import lambdify
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def ParametricPlot3D(expression, u_specs, v_specs):
    """
    mimic ParametricPlot3D()
    expression = ['cos(u)', 'sin(u) + cos(v)', 'sin(v)']
    ParametricPlot3D(expression, ['u', 0, 2*np.pi], ['v', -np.pi, np.pi])
    """
    u, u_min, u_max = u_specs
    v, v_min, v_max = v_specs
    n_points = 100
    u_angles = np.linspace(0, 2 * np.pi, n_points)
    v_angles = np.linspace(0, 2 * np.pi, n_points)
    us, vs = np.meshgrid(u_angles, v_angles)
    x_func = lambdify([parse_expr(u, evaluate=False),
                      parse_expr(v, evaluate=False)],
                      parse_expr(expression[0], evaluate=False))
    vec_x_func = np.vectorize(x_func)
    y_func = lambdify([parse_expr(u, evaluate=False),
                      parse_expr(v, evaluate=False)],
                      parse_expr(expression[1], evaluate=False))
    vec_y_func = np.vectorize(y_func)
    z_func = lambdify([parse_expr(u, evaluate=False),
                      parse_expr(v, evaluate=False)],
                      parse_expr(expression[2], evaluate=False))
    vec_z_func = np.vectorize(z_func)
    xs = vec_x_func(us, vs)
    ys = vec_y_func(us, vs)
    zs = vec_z_func(us, vs)
    # print(zs.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xs, ys, zs, color='orange')
    return ax


def Plot3D(expression, x_specs, y_specs):
    """ mimic Plot3D() function of Mathematica
        Usage: Plot3D("sin(x + y**2)", ['x', -3, 3], ['y', -2, 2])
    """
    x, x_min, x_max = x_specs
    y, y_min, y_max = y_specs
    n_points = 100
    xs = np.linspace(x_min, x_max, n_points)
    ys = np.linspace(y_min, y_max, n_points)
    func = lambdify([parse_expr(x, evaluate=False),
                    parse_expr(y, evaluate=False)],
                    parse_expr(expression, evaluate=False))
    X, Y = np.meshgrid(xs, ys)
    vec_func = np.vectorize(func)
    Z = vec_func(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, color='orange')
    return ax


def ListLinePlot(lst):
    """ mimic ListLinePlot() function of Mathematica
        Usage: ListLinePlot([1, 1, 2, 3, 5, 8])
    """
    fig, ax = plt.subplots()
    ax.plot(lst)
    return ax


def ContourPlot(expression, arg_specs):
    pass


def PolarPlot(expressions, arg_specs, legends=False):
    """ mimic PolarPlot() function of Mathematica
        Usage: PolarPlot(["sin(t)"], ['t', 0, 2*np.pi], legends=True)
    """
    n = len(expressions)
    funcs = [None] * n
    arg, arg_min, arg_max = arg_specs
    for i, expression in enumerate(expressions):
        funcs[i] = lambdify(parse_expr(arg, evaluate=False),
                            parse_expr(expression, evaluate=False))
        n_points = 100
    ts = np.linspace(arg_min, arg_max, n_points)
    fig, ax = plt.subplots()
    for i, func in enumerate(funcs):
        func_v = np.vectorize(func)
        rs = func_v(ts)
        xs = rs * np.cos(ts)
        ys = rs * np.sin(ts)
        ax.plot(xs, ys, label=expressions[i])
    if legends:
        ax.legend(loc="best")
    return ax


def ParametricPlot(expressions, arg_specs, legends=False):
    """mimic ParametricPlot() function of Mathematica
       Usage: ParametricPlot([["cos(u)", "sin(u)"]],
       ["u", 0, 2*np.pi], legends=True)
    """
    arg, arg_min, arg_max = arg_specs
    n = len(expressions)
    x_funcs = [None] * n
    y_funcs = [None] * n
    for i, (x_expression, y_expression) in enumerate(expressions):
        x_funcs[i] = lambdify(parse_expr(arg, evaluate=False),
                              parse_expr(x_expression, evaluate=False))
        y_funcs[i] = lambdify(parse_expr(arg, evaluate=False),
                              parse_expr(y_expression, evaluate=False))
    n_points = 100
    ts = np.linspace(arg_min, arg_max, n_points)
    fig, ax = plt.subplots()
    for i, (x_func, y_func) in enumerate(zip(x_funcs, y_funcs)):
        x_func_v = np.vectorize(x_func)
        y_func_v = np.vectorize(y_func)
        xs = x_func_v(ts)
        ys = y_func_v(ts)
        ax.plot(xs, ys, label=expressions[i])
    if legends:
        ax.legend(loc="best")
    return ax


def Plot(expressions, arg_specs, legends=False):
    """ mimic Plot() function of Mathematica
        Usage: Plot(["sin(x)", "cos(x)"], ['x', 0, 2*np.pi], legends=True)
    """
    n = len(expressions)
    funcs = [None] * n
    arg, arg_min, arg_max = arg_specs
    for i, expression in enumerate(expressions):
        funcs[i] = lambdify(parse_expr(arg, evaluate=False),
                            parse_expr(expression, evaluate=False))
    n_points = 100
    xs = np.linspace(arg_min, arg_max, n_points)
    fig, ax = plt.subplots()
    for i, func in enumerate(funcs):
        func_v = np.vectorize(func)
        ys = func_v(xs)
        ax.plot(xs, ys, label=expressions[i])
    if legends:
        ax.legend(loc="best")
    return ax


def test_Plot():
    """test Plot()"""
    expressions = ['sin(x)', 'sin(2*x)', 'sin(3*x)']
    return Plot(expressions, ['x', 0, 2 * np.pi], legends=True)


def test_ListLinePlot():
    """test ListLinePlot()"""
    return ListLinePlot([1, 1, 2, 3, 5, 8])


def test_ParametricPlot():
    """test ParametricPlot()"""
    expressions = [['2*cos(u)', '2*sin(u)'], ['2*cos(u)', 'sin(u)'],
                   ['cos(u)', '2*sin(u)'], ['cos(u)', 'sin(u)']]
    return ParametricPlot(expressions, ['u', 0, 2 * np.pi], legends=True)


def test_PolarPlot():
    """test PolarPlot()"""
    expressions = ['1', '1 + 1/10 * sin(10*t)']
    return PolarPlot(expressions, ['t', 0, 2 * np.pi])


def test_Plot3D():
    """test Plot3D()"""
    expression = 'sin(x + y**2)'
    return Plot3D(expression, ['x', -3, 3], ['y', -2, 2])


def test_ParametricPlot3D():
    """test ParametricPlot3D()"""
    expression = ['cos(u)', 'sin(u) + cos(v)', 'sin(v)']
    return ParametricPlot3D(
        expression, ['u', 0, 2 * np.pi], ['v', -np.pi, np.pi])


if __name__ == "__main__":
    # ax = test_Plot()
    # ax = test_Plot3D()
    # ax = test_ParametricPlot()
    ax = test_PolarPlot()
    # ax = test_ListLinePlot()
    # ax = test_ParametricPlot3D()
    # plt.axis('equal')
    plt.show()
