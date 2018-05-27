#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
"""
test mathematica_plot module
"""

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


if __name__ == "__main__":
    # ax = test_Plot()
    # ax = test_Plot3D()
    # ax = test_ParametricPlot()
    # ax = test_PolarPlot()
    # ax = test_ListLinePlot()
    plt.axis('equal')
    ax = test_ParametricPlot3D()
    plt.show()
