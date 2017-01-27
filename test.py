# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:39:06 2017

@author: Raman
"""
import numpy as np
from spline_interpolate import interpolate
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def test1d_1d():
    # 1D interpolation
    f = lambda x: x**2

    a = -1
    b = 1
    xp = np.linspace(a, b, 20)
    xi = np.linspace(a, b, 100)
    
    fi = interpolate(xi, xp, f(xp))

    plt.plot(xi, f(xi), 'bo',label='true')
    plt.plot(xi, fi, 'r-', label='interpolated')
    plt.legend()
    plt.show()
    
def test1d_2d():
    xp = np.arange(1,5)
    fp = np.array([xp*(i+1)**2 for i in range(8)])
    x = np.arange(1,4,0.5)
    ff = np.array([x*(i+1)**2 for i in range(8)])
    assert np.allclose(interpolate(x, xp, fp), ff)
    
def test1d_2d_2():
    xp = np.arange(1,5)
    fp = np.array([xp*(i+1)**2 for i in range(8)])
    x = np.arange(1,4,0.5)
    ff = np.array([x*(i+1)**2 for i in range(8)])
    assert np.allclose(interp1d(xp, fp, 'slinear')(x), ff)

if __name__ == '__main__':
#    test1d_1d()
    import timeit
    N = 300
#    print(test1d_2d())
#    test1d_2d()
    print(timeit.timeit("test1d_2d()", setup="from __main__ import test1d_2d", number=N)/N)
    print(timeit.timeit("test1d_2d_2()", setup="from __main__ import test1d_2d_2", number=N)/N)
    print(timeit.timeit("test1d_2d()", setup="from __main__ import test1d_2d", number=N)/N)
    print(timeit.timeit("test1d_2d_2()", setup="from __main__ import test1d_2d_2", number=N)/N)

#if not "2D interpolation":
#    # 2D interpolation
#    f2d = lambda x, z: x**2 + 2*x + 1 + z ** 0.5 + 3 * z
#    
#    a1, a2 = 0, 0
#    b1, b2 = 1, 1
#    n1, n2 = 49, 39  # n + 1 grid points (0,..., n)
#
#    h1, h2 = (b1 - a1)/n1, (b2 - a2)/n2
#    grid_x = np.arange(n1 + 1) * h1 + a1
#    grid_z = np.arange(n2 + 1) * h2 + a2
#
#    y = np.zeros((n1 + 1, n2 + 1))
#
#    for i, x in enumerate(grid_x):
#        for j, z in enumerate(grid_z):
#            y[i, j] = f2d(x, z)
#
#    alpha = 0
#    beta = 0
#
#    c_tmp = np.zeros((n1 + 3, n2 + 1))
#    get_coeffs(a1, b1, y, c_tmp)
#
#    c = np.zeros((n1 + 3, n2 + 3))
#    # NOTE: here you have to pass c_tmp.T and c.T
#    get_coeffs(a2, b2, c_tmp.T, c.T)
#
#    fhat = np.zeros((n1 + 1, n2 + 1))
#    for i, x in enumerate(grid_x):
#        for j, z in enumerate(grid_z):
#            fhat[i, j] = interpolate2d(x, z, a1, b1, a2, b2, c)
#
#    real_val = np.zeros((n1 + 1, n2 + 1))
#    for i, x in enumerate(grid_x):
#        for j, z in enumerate(grid_z):
#            real_val[i, j] = f2d(x, z)
#
#    from mayavi import mlab
#
#    def draw_3d(grid_x, grid_y, fval, title='pi'):
#        mlab.figure()
#        mlab.surf(grid_x, grid_y, fval)#, warp_scale="auto")
#        mlab.axes(xlabel='x', ylabel='z', zlabel=title)
#        mlab.orientation_axes(xlabel='x', ylabel='z', zlabel=title)
#        mlab.title(title)
#
#    draw_3d(grid_x, grid_z, fhat, title='interpolated')
#    draw_3d(grid_x, grid_z, real_val, title='real')
#    mlab.show()