
#-----------------------------------------------------------------------------
# fitting
#-----------------------------------------------------------------------------

from scipy.spatial.kdtree import KDTree
from ..ImportAll import *
import numpy as _numpy


#-----------------------------------------------------------------------------
# poly fit
#-----------------------------------------------------------------------------

def poly2dfit_minimum_(x : T_ARRAY,y : T_ARRAY, fit_width : int = 5):
    
    CHECK_NDIM_(x, "x", 1)
    CHECK_NDIM_(y, "y", 1)

    idx = y[:].argmin()
    idx1 = idx - fit_width
    idx2 = idx + fit_width
    xfit = x[idx1:idx2]
    yfit = y[idx1:idx2]
    coe =  _numpy.polyfit(xfit,yfit, deg=2)
    return -0.5 * coe[1] / coe[0]

def poly2dfit_horizontal_line_(img : T_ARRAY, y0 : int, x1: int, x2 : int, yr : int = 20 , step_size : int = 50, average_size : int = 3, fit_width : int = 5):

    CHECK_NDIM_(img, "img", 2)

    xs : T_ARRAY = _numpy.arange(x1, x2, step_size)
    ys = _numpy.empty( xs.shape )
    
    for k in range(xs.size):
        x0 = xs[k]
        prof_y = img[y0-yr:y0+yr, x0-average_size:x0+average_size].mean(axis=1)
        prof_x = _numpy.arange(y0-yr, y0+yr)
        ys[k] = poly2dfit_minimum_(prof_x, prof_y, fit_width=fit_width)
        
    coe =  _numpy.polyfit(xs,ys, deg=2)
    ys_smooth = coe[0] * xs * xs + coe[1] * xs + coe[2]
    threshold = (ys_smooth - ys).std() * 2.0
    mask = ((ys_smooth - ys) < threshold) 
    coe = _numpy.polyfit(xs[mask],ys[mask], deg=2)
    ys_smooth = coe[0] * xs * xs + coe[1] * xs + coe[2]

    return xs, ys_smooth, coe

def poly2dfit_curve_line_(img : T_ARRAY, y0 : int, x1: int, x2 : int, yr : int = 20 , step_size : int = 50, average_size : int = 3, fit_width : int = 5, kind : int = 0):
    """
    Parameters
    -----------
    kind : int
        0 : horizontal line
        1 : vertical line
    """
    CHECK_NDIM_(img, "img", 2)
    CHECK_IN_(kind, "kind", (0,1))

    xs : T_ARRAY = _numpy.arange(x1, x2, step_size)
    ys = _numpy.empty( xs.shape )
    
    for k in range(xs.size):
        x0 = xs[k]
        if kind == 0:
            prof_y = img[y0-yr:y0+yr, x0-average_size:x0+average_size].mean(axis=1)
        else:
            prof_y = img[x0-average_size:x0+average_size, y0-yr:y0+yr].mean(axis=0)
        prof_x = _numpy.arange(y0-yr, y0+yr)
        try:
            ys[k] = poly2dfit_minimum_(prof_x, prof_y, fit_width=fit_width)
        except TypeError:
            ys[k] = ys[k-1]
        
    coe : T_ARRAY =  _numpy.polyfit(xs,ys, deg=2)
    ys_smooth : T_ARRAY = coe[0] * xs * xs + coe[1] * xs + coe[2]
    threshold = (ys_smooth - ys).std() * 2.0
    mask = ((ys_smooth - ys) < threshold) 
    coe = _numpy.polyfit(xs[mask],ys[mask], deg=2)
    ys_smooth = coe[0] * xs * xs + coe[1] * xs + coe[2]

    return xs, ys_smooth, coe

#-----------------------------------------------------------------------------
# find crossing of vertical and horizontal poly2d lines by data sampling
from scipy.spatial import KDTree as _KDTree
#-----------------------------------------------------------------------------

def poly2d_cross_(img : T_ARRAY, coe_h : T_ARRAY, coe_v : T_ARRAY, upsample : int = 5):
    """

    Returns
    --------
    cross : Tuple[int,int]
        coordinate (y,x) of the approximated cross point

    """
    CHECK_NDIM_(img, "img", 2)

    nrows, ncols = img.shape
    x_mesh = _numpy.linspace( 0, ncols-1, (ncols-1)*upsample + 1 )
    y_mesh = _numpy.linspace( 0, nrows-1, (nrows-1)*upsample + 1 )

    points_h = _numpy.empty((x_mesh.size,2))
    points_h[:,1] = x_mesh[:]
    points_h[:,0] = coe_h[0] * points_h[:,1] * points_h[:,1] + coe_h[1] * points_h[:,1] + coe_h[2]

    points_v = _numpy.empty((y_mesh.size,2))
    points_v[:,0] = y_mesh[:]
    points_v[:,1] = coe_v[0] * points_v[:,0] * points_v[:,0] + coe_v[1] * points_v[:,0] + coe_v[2]

    tree_v = _KDTree( points_v )
    dists, idxs = tree_v.query(points_h, k=1)
    idx_min_dist = _numpy.array(dists).argmin()
    point_selected_v = points_v[ idxs[idx_min_dist], : ]
    
    tree_h = _KDTree( points_h )
    dist, idx = tree_h.query(point_selected_v, k=1)
    point_selected_h = points_h[ idx ]

    cross = point_selected_h[0], point_selected_v[1]
    
    return cross

#-----------------------------------------------------------------------------
# numerically compute the arc length of function
#-----------------------------------------------------------------------------
    
from math import hypot as _hypot
import numba as _numba

@_numba.jit( nopython=True, cache=False, nogil=False )
def polynd_(coe : T_ARRAY, x : T_ARRAY):

    y = coe[0] * _numpy.ones_like(x)
    for k in range(1, coe.size):
        y[:] *= x[:]
        y[:] += coe[k]

    return y

@_numba.jit( nopython=True, cache=False, nogil=False )
def arc_length_(func : Callable, a : float, b : float, tol : float = 1E-6):
    """
    reference : https://stackoverflow.com/questions/46098157/how-to-calculate-the-length-of-a-curve-of-a-math-function
    Compute the arc length of function f(x) for a <= x <= b. 
    Stop when two consecutive approximations are closer than the value of tol.
    """
    nsteps = 1  # number of steps to compute
    oldlength = 1.E20
    length = 1.E10
    while abs(oldlength - length) >= tol:
        nsteps *= 2
        fx1 = func(a)
        xdel = (b - a) / nsteps  # space between x-values
        oldlength = length
        length = 0
        for i in range(1, nsteps + 1):
            fx0 = fx1  # previous function value
            fx1 = func(a + i * (b - a) / nsteps)  # new function value
            length += _hypot(xdel, fx1 - fx0)  # length of small line segment
    return length

