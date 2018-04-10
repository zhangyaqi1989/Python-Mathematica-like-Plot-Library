#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
#
##################################


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    print("Hello World")
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    angle = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(angle, angle)
    X = np.cos(theta)
    Y = np.sin(theta) + np.cos(phi)
    Z = np.sin(phi)
    ax.plot_surface(X, Y, Z, color='orange')
    plt.axis('equal')
    plt.show()
