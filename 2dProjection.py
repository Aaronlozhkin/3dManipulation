#Developed by Aaron Lozhkin
#2/20/21 - 2D Projection of Vector on Plane

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import time

def orthogonalProjectionMatrix(a, b, c): #Constructs the orthogonal projection matrix for a plane with equation ax + by + cz = 0
    vector1 = np.array([1, 0, (-a/c)])          #Creates a vector on the plane
    vectorPerp = np.array([a, b, c])            #Creates a vector orthogonal to the plane
    vector2 = np.cross(vector1, vectorPerp)     #Crosses these two vectors to get a vector orhtogonal to both and on the plane
    orthonormalBasis = np.column_stack((vector1, vector2))
    projectionMatrix = np.dot(np.dot(orthonormalBasis, np.linalg.inv(np.dot(np.transpose(orthonormalBasis), orthonormalBasis))), np.transpose(orthonormalBasis))
    xx, yy = np.meshgrid(range(10), range(10))
    zz = (-vectorPerp[0] * xx - vectorPerp[1] * yy) * 1. / vectorPerp[2]
    axis.plot_surface(xx, yy, zz, alpha=0.2)      #Plots the plane
    return projectionMatrix

def zFunction(x, y):                            #creates a function f(x, y) = z
    return np.sqrt(x**2 + y**2)

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
z = zFunction(x,y)

lineArray = np.stack((x, y, z))             #Creates a 3xn vector representing the values of f(x,y)

axis = plt.axes(projection="3d")

projectionMatrix = orthogonalProjectionMatrix(3, -2, 4)

graph = np.dot(projectionMatrix, lineArray)

axis.plot3D(lineArray[0,:], lineArray[1,:], lineArray[2,:])
axis.plot3D(graph[0,:], graph[1,:], graph[2,:])
plt.show()
