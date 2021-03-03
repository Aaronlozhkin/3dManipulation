# Developed by Aaron Lozhkin
# 2/20/21 - 2D Projection of Vector on Plane

import time
import matplotlib.pyplot as plt
import numpy as np

def orthogonal_projection_matrix(a, b, c):
    """Return the orthogonal projection matrix for a plane with equation ax + by + cz = 0"""
    vector1 = np.array([1, 0, (-a / c)])  # Creates a vector on the plane
    vector_perp = np.array([a, b, c])  # Creates a vector orthogonal to the plane
    vector2 = np.cross(vector1,
                       vector_perp)  # Crosses these two vectors to get a vector orhtogonal to both and on the plane
    orthonormal_basis = np.column_stack((vector1, vector2))
    projection_matrix = np.dot(
        np.dot(orthonormal_basis, np.linalg.inv(
            np.dot(np.transpose(orthonormal_basis), orthonormal_basis))),
            np.transpose(orthonormal_basis))
    plane_x, plane_y = np.meshgrid(range(10), range(10))
    plane_z = (-vector_perp[0] * plane_x - vector_perp[1] * plane_y) * 1. / vector_perp[2]
    AXIS.plot_surface(plane_x, plane_y, plane_z, alpha=0.2)  # Plots the plane
    return projection_matrix


def z_function(x, y):
    """Return a function f(x, y) = z"""
    return np.sqrt(x ** 2 + y ** 2)


X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
Z = z_function(X, Y)
LINEARRAY = np.stack((X, Y, Z))  # Creates a 3xn vector representing the values of f(x,y)

AXIS = plt.axes(projection="3d")

PROJECTIONMATRIX = orthogonal_projection_matrix(3, -2, 4)
PROJECTEDVECTOR = np.dot(PROJECTIONMATRIX, LINEARRAY)

AXIS.plot3D(LINEARRAY[0, :], LINEARRAY[1, :], LINEARRAY[2, :])
AXIS.plot3D(PROJECTEDVECTOR[0, :], PROJECTEDVECTOR[1, :], PROJECTEDVECTOR[2, :])
plt.show()
