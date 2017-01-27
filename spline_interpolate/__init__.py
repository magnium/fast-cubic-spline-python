# Fast-Cubic-Spline-Python provides an implementation of 1D and 2D fast spline
# interpolation algorithm (Habermann and Kindermann 2007) in Python.
# Copyright (C) 2012, 2013 Joon H. Ro

# This file is part of Fast-Cubic-Spline-Python.

# Fast-Cubic-Spline-Python is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Fast-Cubic-Spline-Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Fast-Cubic-Spline-Python provides an implementation of 1D and 2D fast spline
# interpolation algorithm (Habermann and Kindermann 2007) in Python.
# Copyright (C) 2012, 2013 Joon H. Ro

# This file is part of Fast-Cubic-Spline-Python.

# Fast-Cubic-Spline-Python is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Fast-Cubic-Spline-Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import numpy as np
#from scipy.linalg import solve_banded

from fast_cubic_spline import interpolate1d, interpolate2d, interpolate1dx

def interpolate(x, xp, fp):
    if np.ndim(xp) == 1:
        a, b = xp[0], xp[-1]
        if np.ndim(fp) == 1:
            c = get_coeffs(a, b, fp.T)
            return interpolate1d(x, a, b, c.T)
        elif np.ndim(fp) == 2:
            c = get_coeffs(a, b, fp.T)
#            return interpolate1dx(x, a, b, c.T)
            return np.array([interpolate1d(x, a, b, cc) for cc in c.T])
#            return np.array([interpolate1d(x, a, b, get_coeffs(a, b, fpp)) for fpp in fp])
    elif np.ndim(xp) == 2:
        xs, ys = xp # extract x-grid and y-grid
        a1, a2 = xs[0, 0], xs[0, -1]
        b1, b2 = ys[0, 0], xp[-1, 0]
        n1, n2 = np.shape(xs)
        
        c_tmp = np.zeros((n1 + 3, n2 + 1))
        get_coeffs(a1, b1, fp, c_tmp)

        c = np.zeros((n1 + 3, n2 + 3))
        # NOTE: here you have to pass c_tmp.T and c.T
        get_coeffs(a2, b2, c_tmp.T, c.T)
        
        interpolate2d(xs, ys, fp, a1, b1, a2, b2, c)

def get_coeffs(a, b, y, c=None, alpha=0, beta=0):
    '''
    Return spline coefficients 

    Parameters
    ----------
    a : float
        lower bound of the grid.
    b : float
        upper bound of the grid.
    y : ndarray
        actual function value at grid points.
    c : (y.shape[0] + 2, ) ndarray, optional
        ndarry to be written
    alpha : float
        Second-order derivative at a. Default is 0.
    beta : float
        Second-order derivative at b. Default is 0.

    Returns
    -------
    out : ndarray
        Array of coefficients.
    '''
    n = y.shape[0] - 1
    h = (b - a)/n
    
    shape = list(np.shape(y))
    shape[0] += 2

    if c is None:
        c = np.zeros(shape)
        ifreturn = True
    else:
        assert(c.shape[0] == n + 3)
        ifreturn = False

    c[1] = 1/6 * (y[0] - (alpha * h**2)/6)
    c[n+1] = 1/6 * (y[n] - (beta * h**2)/6)

#    # ab matrix here is just compressed banded matrix
#    ab = np.ones((3, n - 1))
#    ab[0, 0] = 0
#    ab[1, :] = 4
#    ab[-1, -1] = 0

    B = np.array(y[1:-1], dtype=float)
    B[0] -= c[1]
    B[-1] -=  c[n + 1]
    
    ab = np.diag(np.ones(n-1)*4,0)+np.diag(np.ones(n-2),-1)+np.diag(np.ones(n-2), 1)
    ab = np.tile(ab, y.shape[1:]+(1,1))

#    c[2:-2] = solve_banded((1, 1), ab, B)
    c[2:-2] = np.linalg.solve(ab, B.T).T

    c[0] = alpha * h**2/6 + 2 * c[1] - c[2]
    c[-1] = beta * h**2/6 + 2 * c[-2] - c[-3]

    if ifreturn:
        return(c)
