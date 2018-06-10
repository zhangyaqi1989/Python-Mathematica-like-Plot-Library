#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
"""
test mathematica_plot module
"""

# standard library
import sys

# 3rd party library
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# local library
from mathematica_plot import (Plot, ListLinePlot, ParametricPlot,
                              PolarPlot, Plot3D, ParametricPlot3D)


def test_Plot():
    """ test Plot() """
    expressions = ['sin(x)', 'sin(2*x)', 'sin(3*x)']
    return Plot(expressions, ['x', 0, 2 * np.pi], legends=True)


def test_ListLinePlot():
    """ test ListLinePlot() """
    return ListLinePlot([1, 1, 2, 3, 5, 8])


def test_ParametricPlot():
    """ test ParametricPlot() """
    expressions = [['2*cos(u)', '2*sin(u)'], ['2*cos(u)', 'sin(u)'],
                   ['cos(u)', '2*sin(u)'], ['cos(u)', 'sin(u)']]
    return ParametricPlot(expressions, ['u', 0, 2 * np.pi], legends=True)


def test_PolarPlot():
    """ test PolarPlot() """
    expressions = ['1', '1 + 1/10 * sin(10*t)']
    return PolarPlot(expressions, ['t', 0, 2 * np.pi], n_points=200)


def test_Plot3D():
    """ test Plot3D() """
    expression = 'sin(x + y**2)'
    return Plot3D(expression, ['x', -3, 3], ['y', -2, 2])


def test_ParametricPlot3D():
    """ test ParametricPlot3D() """
    expression = ['cos(u)', 'sin(u) + cos(v)', 'sin(v)']
    return ParametricPlot3D(
        expression, ['u', 0, 2 * np.pi], ['v', -np.pi, np.pi])


def command_line_runner():
    """command line runner"""
    funcs_names = ['test_Plot',
                   'test_Plot3D',
                   'test_ParametricPlot',
                   'test_PolarPlot',
                   'test_ListLinePlot',
                   ]
    funcs = [globals()[item] for item in funcs_names]
    if len(sys.argv) != 2:
        print('Usage: >> python {} <test_type ({}-{})>'.
              format(sys.argv[0], 0, len(funcs_names) - 1))
        for i, test in enumerate(funcs_names):
            _, funcs_name = test.split('_', 1)
            print("test type = {:d}: {}".format(i, funcs_name + '()'))
        sys.exit(1)
    test_type = int(sys.argv[1])
    assert(0 <= test_type < len(funcs_names))
    funcs[test_type]()
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    command_line_runner()
