
#-----------------------------------------------------------------------------
# measure shift in 1d array or 2d image
#-----------------------------------------------------------------------------

from ..ImportAll import *

#-----------------------------------------------------------------------------
# with skimage
from skimage.feature import register_translation as _register_translation
#-----------------------------------------------------------------------------

def measure_shift2d_( src : T_ARRAY, dst : T_ARRAY):
    """

    Returns
    -------
    shift : Tuple[float,float]
        [shfit_y, shift_x]
    """
    if src.ndim != 2:
        raise ValueError(f"src.ndim = {src.ndim}, must == 2")
    if dst.ndim != 2:
        raise ValueError(f"dst.ndim = {dst.ndim}, must == 2")

    shift, error, diffphase = _register_translation( src, dst, upsample_factor=10. , return_error=True )

    return -1 * shift

def measure_shift1d_( src : T_ARRAY, dst : T_ARRAY):

    if src.ndim != 1:
        raise ValueError(f"src.ndim = {src.ndim}, must == 1")
    if dst.ndim != 1:
        raise ValueError(f"dst.ndim = {dst.ndim}, must == 1")

    src2d = src.reshape(-1,1)
    dst2d = dst.reshape(-1,1)

    shift = measure_shift2d_( src2d, dst2d )

    return shift[0]
