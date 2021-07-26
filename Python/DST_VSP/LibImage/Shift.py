
#-----------------------------------------------------------------------------
# image operation : shift (translation)
#-----------------------------------------------------------------------------

from ..ImportAll import *

import numpy as _numpy
#-----------------------------------------------------------------------------
# shift with opencv
import cv2 as _cv2
#-----------------------------------------------------------------------------

def cv2_shift2d_(src : T_ARRAY, tx : float, ty : float):

    if src.ndim != 2:
        raise ValueError(f"src.ndim = {src.ndim}, must == 2")
    rows, cols = src.shape

    M = _numpy.float32( [[1,0,tx],[0,1,ty]] )
    dst = _cv2.warpAffine( src, M, (cols,rows) )

    return dst


def cv2_shift1d_(src : T_ARRAY, t : float):

    if src.ndim != 1:
        raise ValueError(f"src.ndim = {src.ndim}, must == 1")
    rows = src.size
    cols = 1
    src2d = src.reshape(-1,1)

    M = _numpy.float32( [[1,0,1],[0,1,t]] )
    dst = _cv2.warpAffine( src, M, (cols,rows) )

    return dst.reshape(-1)

#-----------------------------------------------------------------------------
# shift with scipy.ndimage
from scipy.ndimage import fourier_shift as _fourier_shift
#-----------------------------------------------------------------------------

def fft_shift_( src : T_ARRAY, t : Union[int, float, T_ARRAY, Tuple[int,...],Tuple[float,...]] ):

    dst = _fourier_shift( _numpy.fft.fftn(src), t )
    dst = _numpy.fft.ifftn( dst )

    return dst.real



