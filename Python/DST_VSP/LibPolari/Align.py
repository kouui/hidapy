
#-----------------------------------------------------------------------------
# alignment of spectra panel with orthogonal linear polarization
#-----------------------------------------------------------------------------

from functools import cache
from ..ImportAll import *
import numpy as _numpy
from collections import namedtuple as _nametuple


#-----------------------------------------------------------------------------
# rotate the image by setting the hair line vertical
from ..LibImage.Rotate import rotate2d_ as _rotate2d_
from ..LibImage.Shift import fft_shift_ as _fft_shift_
from ..LibImage.Measure import measure_shift1d_ as _measure_shift1d_
#-----------------------------------------------------------------------------

def _should_rotate_cloclwise_(shift_LR : float):

    return shift_LR < 0

def _should_stop_rotate_(shift_LR : float, threshold : float = 0.5):
    return abs(shift_LR) < threshold

def _measure_hair_shift_(img : T_ARRAY, coords : T_ARRAY, dx : int, dy : int):

    # upper left
    x0, y0 = coords[0,:]
    prof_UL = img[ y0-dy:y0+dy, x0-dx:x0+dx ].mean(axis=0)
    # lower left
    x0, y0 = coords[1,:]
    prof_LL = img[ y0-dy:y0+dy, x0-dx:x0+dx ].mean(axis=0)
    # upper right
    x0, y0 = coords[2,:]
    prof_UR = img[ y0-dy:y0+dy, x0-dx:x0+dx ].mean(axis=0)
    # lower right
    x0, y0 = coords[3,:]
    prof_LR = img[ y0-dy:y0+dy, x0-dx:x0+dx ].mean(axis=0)

    shift_L = _measure_shift1d_(prof_LL, prof_UL)
    shift_R = _measure_shift1d_(prof_LR, prof_UR)
    shift_LR = shift_L + shift_R

    return shift_LR


def rotate_image_to_hair_line_vertical_iter_(img : T_ARRAY, coords : T_ARRAY, dx : int = 20, dy : int = 10, max_iter : int = 30, threshold : float = 0.5 ):

    if img.ndim != 2:
        raise ValueError(f"img.ndim = {img.ndim}, must == 2")
    
    # [x,y]
    # [[upper left], [lower left], [upper right], [lower right]]
    if coords.shape != (4,2):
        raise ValueError(f"coords.shape = {coords.shape}, must == (4,2)")

    rotate_deg = 1.0
    #img_iter = img.copy()
    count = 1
    direction = -1
    img_iter = _rotate2d_(img, rotate_deg * direction)

    while True:

        # update value
        shift_LR =  _measure_hair_shift_( img_iter, coords, dx, dy )
        if _should_rotate_cloclwise_(shift_LR):
            direction_new = -1
        else:
            direction_new = +1
        
        if (direction_new + direction) == 0:
            rotate_deg_new = 0.8 * rotate_deg
        else:
            rotate_deg_new = 1.0 * rotate_deg

        print(shift_LR, rotate_deg_new * direction_new)

        # whether to break the loop
        if count > max_iter:
            print(f"exist reaching max_iter = {max_iter}")
            break
        
        if _should_stop_rotate_( shift_LR, threshold ):
            break
        img_iter = _rotate2d_(img_iter, rotate_deg_new * direction_new)
        
        # store previous value
        rotate_deg = rotate_deg_new
        direction = direction_new

        count += 1

    
    return img_iter, rotate_deg

def rotate_image_to_hair_line_vertical_(img : T_ARRAY, coords : T_ARRAY, dx : int = 20, dy : int = 10, deg_min : float = -0.5, deg_max: float = 0.5, nstep : int = 101):
    if img.ndim != 2:
        raise ValueError(f"img.ndim = {img.ndim}, must == 2")
    
    # [x,y]
    # [[upper left], [lower left], [upper right], [lower right]]
    if coords.shape != (4,2):
        raise ValueError(f"coords.shape = {coords.shape}, must == (4,2)")


    deg_arr = _numpy.linspace(deg_min, deg_max, nstep)

    shift_LR_arr = _numpy.empty_like( deg_arr )
    for k in range(nstep):
        img_rotated = _rotate2d_(img, deg_arr[k])
        shift_LR_arr[k] =  _measure_hair_shift_( img_rotated, coords, dx, dy )

    shift_LR_arr[:] = _numpy.abs( shift_LR_arr[:] )
    rotate_deg = deg_arr[ shift_LR_arr.argmin() ]
    img_rotated = _rotate2d_(img, rotate_deg)

    return img_rotated, rotate_deg

#-----------------------------------------------------------------------------
# split left and right panel with equal width
#-----------------------------------------------------------------------------

def _find_start_given_width_( profR : T_ARRAY, dx : int):

    dcounts = []
    for x1 in range(0, profR.size - dx - 1):
        x2 = x1 + dx
        dcounts.append( abs( profR[x1] - profR[x2] ) )
    dcounts = _numpy.array( dcounts )
    x1 = dcounts.argmin()
    
    return x1

def get_left_right_split_x_( img3d : T_ARRAY, center : Tuple[int,int], dy : int = 20, ratio : float = 0.2, inner_offset : int = 10 ):
    """
    Parameers
    ----------
    center : Tuple[int,int]
        (y,x) , a rough center point to split left and right panel
    """
    if img3d.ndim != 3:
        raise ValueError(f"img3d.ndim = {img3d.ndim}, must == 3")

    img = img3d.mean(axis=0)
    x0, y0 = center

    prof = img[y0-dy:y0+dy, :].mean(axis=0)
    threshold = _numpy.median( prof ) * ratio

    xL1 : int = _numpy.where( prof[0 : x0] > threshold )[0][0]
    xL2 : int = _numpy.where( prof[0 : x0] > threshold )[0][-1] + 1

    dx = xL2-xL1
    xR1 : int = _find_start_given_width_( prof[x0:], dx ) + x0 
    xR2 = xR1 + dx

#    ret = _nametuple("Split_Range", ["left", "right"]) (
#        (xL1 + inner_offset, xL2 - inner_offset), 
#        (xR1 + inner_offset, xR2 - inner_offset)
#        )
#
#    return ret

    return {
        "left" : (xL1 + inner_offset, xL2 - inner_offset), 
        "right": (xR1 + inner_offset, xR2 - inner_offset),
        }

#-----------------------------------------------------------------------------
# remove paddings after shifting panels
#-----------------------------------------------------------------------------

def remove_padding_(panel : T_ARRAY, panel_shift : T_ARRAY, shift : T_ARRAY):

    CHECK_NDIM_(panel, "panel", 2)
    CHECK_NDIM_(panel_shift, "panel_shift", 2)

    t_y, t_x = shift
    nrows, ncols = panel.shape
    
    # y
    if t_y > 0:
        yr = int( _numpy.ceil(t_y) ), nrows
    elif t_y < 0:
        yr = 0, nrows - (-1 * int( t_y ) + 1)
    else:
        yr = 0, nrows

    # x
    if t_x > 0:
        xr = int( _numpy.ceil(t_x) ), ncols
    elif t_x < 0:
        xr = 0, ncols - (-1 * int( t_x ) + 1)
    else:
        xr = 0, ncols

    return panel[yr[0]:yr[1], xr[0]:xr[1]], panel_shift[yr[0]:yr[1], xr[0]:xr[1]]


#-----------------------------------------------------------------------------
# given the distored fitted coordinates, 
# generate target rectangular coordinate 
from .Fit import arc_length_ as _arc_length_
#-----------------------------------------------------------------------------

def _sort_cross_coords_(cross_v1 : List[Tuple[float,float]], cross_v2 : List[Tuple[float,float]]):
    """
    y coordinates sorted in descending order.
    """
    x_v1, x_v2 = cross_v1[0][1], cross_v1[0][1]
    if x_v1 > x_v2: # v1 right, v2 left
        cross_L = sorted( cross_v2, key = lambda x : x[0], reverse=True )
        cross_R = sorted( cross_v1, key = lambda x : x[0], reverse=True )
    else: # v1 left, v2 right
        cross_L = sorted( cross_v1, key = lambda x : x[0], reverse=True )
        cross_R = sorted( cross_v2, key = lambda x : x[0], reverse=True )

    return cross_L, cross_R

def _pick_bigger_(x1 : float, x2 : float):
    return x1 if x1 > x2 else x2

def _pick_smaller_(x1 : float, x2 : float):
    return x1 if x1 < x2 else x2

import numba as _numba
def arc_length_ratio_( coe : T_ARRAY, xs : T_ARRAY):
    """
    xs should be sorted in ascending order
    """

    @_numba.jit( nopython=True, cache=False, nogil=False )
    def _poly2d_( _x : float ):
        return coe[0] * _x * _x + coe[1] * _x + coe[2]

    xs_target = xs[1:]
    xs_begin = xs[:-1]
    lengths = _numpy.empty( xs_target.size )
    for k, (x1, x2) in enumerate( zip( xs_begin, xs_target) ): 
        lengths[k] = _arc_length_(_poly2d_, x1, x2)

    return lengths

def hair_line_cross_ratio_( cross : List[Tuple[float,float]], coe : T_ARRAY ):
    
    ys = _numpy.array( [ c[0] for c in cross ] )
    lengths = arc_length_ratio_( coe, ys[::-1])
    ratios = lengths[::-1].cumsum()
    ratios[:] /= ratios[-1]
    
    return ratios

def cross_to_rects_coords_map_(cross_v1 : List[Tuple[float,float]], cross_v2 : List[Tuple[float,float]], coe_v1 : T_ARRAY, coe_v2 : T_ARRAY):

    cross_L, cross_R = _sort_cross_coords_( cross_v1, cross_v2 )
    if cross_L[0] == cross_v1[0]:
        coe_L = coe_v1
        coe_R = coe_v2
    elif cross_R[0] == cross_v1[0]:
        coe_R = coe_v1
        coe_L = coe_v2
    else:
        raise ValueError(f"Cannot decide Left/Right between v1/v2")

    c4 = {
        "ul" : cross_L[0], # upper left
        "ll" : cross_L[-1],# lower left
        "ur" : cross_R[0], # upper right
        "lr" : cross_R[-1],# lower right
    }

    # build inner rectangle
    # upper y coord
    y_u = _pick_smaller_( c4["ul"][0], c4["ur"][0] )
    y_l = _pick_bigger_( c4["ll"][0], c4["lr"][0] )
    x_r = _pick_smaller_( c4["ur"][1], c4["lr"][1] )
    x_l = _pick_bigger_( c4["ul"][1], c4["ll"][1] )


    # calculate arc length for left hair line
    ratio_L = hair_line_cross_ratio_(cross_L, coe_L)
    # calculate arc length for right hair line
    ratio_R = hair_line_cross_ratio_(cross_R, coe_R)
    # average raito calculated in left/right hair lines
    # ratio in an order of : large y -> small y
    ratio = 0.5 * ( ratio_L[:] + ratio_R[:] )

    
    ys_rects = [y_u,] + [ y_u - (y_u - y_l) * r for r in ratio[:-1] ] + [y_l,]

    coords_map : Dict[str, T_ARRAY] = {
        "L" : _numpy.empty( (len(cross_L),4) ),
        "R" : _numpy.empty( (len(cross_R),4) ),
    }
    
    for k, (cr, yrect) in enumerate( zip( cross_L, ys_rects ) ): 
        coords_map["L"][k,:2] =  cr
        coords_map["L"][k,2:] = (yrect,x_l)
    for k, (cr, yrect) in enumerate( zip( cross_R, ys_rects ) ): 
        coords_map["R"][k,:2] =  cr
        coords_map["R"][k,2:] = (yrect,x_r)

    return coords_map


#-----------------------------------------------------------------------------
# piecewise warping :
# 
# comparing to matlab's " PiecewiseLinearTransformation2D", skimage's "PiecewiseAffineTransform"
# only implements the first two steps (totally three), so generate the black border outside the control points.
# reference : https://forum.image.sc/t/equivalent-for-matlabs-piecewiselineartransformation2d/51035
# maybe poly skimage's "PolynomialTransform" and Zachary Pincus's "thin plate splines" is better
# 
# PolynomialTransform (does not work well)
# reference : https://scikit-image.org/docs/dev/api/skimage.transform.html#polynomialtransform
# 
# warp_image ( thin plate splines) by Zachary Pincus
# reference : https://github.com/kouui/celltool/blob/master/celltool/numerics/image_warp.py
from skimage.transform import PiecewiseAffineTransform as _PiecewiseAffineTransform
from skimage.transform import PolynomialTransform as _PolynomialTransform
from skimage.transform import warp as _warp
from ..LibImage.Warp import _make_inverse_warp
from scipy import ndimage as _ndimage
#-----------------------------------------------------------------------------

def calculate_warp_tform_piecewise_(src : T_ARRAY, dst : T_ARRAY):

    CHECK_NDIM_(src, "src", 2)
    CHECK_NDIM_(dst, "dst", 2)

    tform = _PiecewiseAffineTransform()
    #tform.estimate(src, dst)
    tform.estimate(dst, src)

    return tform

def calculate_warp_tform_polyn_(src : T_ARRAY, dst : T_ARRAY, order : int = 2):

    CHECK_NDIM_(src, "src", 2)
    CHECK_NDIM_(dst, "dst", 2)

    tform = _PolynomialTransform()
    #tform.estimate(src, dst, order)
    tform.estimate(dst, src, order)

    return tform

def coords_warp_skimage_(img : T_ARRAY, tform : _PiecewiseAffineTransform):

    CHECK_NDIM_(img, "img", 2)
    #return _warp(img, tform.inverse)
    return _warp(img, tform)


def calculate_warp_tform_spline_(src : T_ARRAY, dst : T_ARRAY, output_region : Tuple[int,int,int,int],  approximate_grid : int = 2):
    """Define a thin-plate-spline warping transform that warps from the from_points
    to the to_points, and then warp the given images by that transform. This
    transform is described in the paper: "Principal Warps: Thin-Plate Splines and
    the Decomposition of Deformations" by F.L. Bookstein.

    Parameters:
        - from_points and to_points: Nx2 arrays containing N 2D landmark points in (y,x) order.
        - output_region: the (ymin, xmin, ymax, xmax) region of the output
                image that should be produced. (Note: The region is inclusive, i.e.
                xmin <= x <= xmax)
        - approximate_grid: defining the warping transform is slow. If approximate_grid
                is greater than 1, then the transform is defined on a grid 'approximate_grid'
                times smaller than the output image region, and then the transform is
                bilinearly interpolated to the larger region. This is fairly accurate
                for values up to 10 or so.
    """
    return _make_inverse_warp(src, dst, output_region, approximate_grid)

def coords_warp_spline_(img : T_ARRAY, tform : List, interpolation_order : int = 1):
    """
    Parameters:
        - interpolation_order: if 1, then use linear interpolation; if 0 then use
                nearest-neighbor.
    """
    CHECK_NDIM_(img, "img", 2)
    return _ndimage.map_coordinates(_numpy.asarray(img), tform, order=interpolation_order)











    
