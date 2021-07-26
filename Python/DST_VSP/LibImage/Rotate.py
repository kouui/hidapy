#-----------------------------------------------------------------------------
# image operation : Rotation
#-----------------------------------------------------------------------------

from ..ImportAll import *

#-----------------------------------------------------------------------------
# with opencv
import cv2 as _cv2
#-----------------------------------------------------------------------------

def rotate2d_(img : T_ARRAY, deg : float, 
            center : Union[ Tuple[int,int],Tuple[float,float], None ] = None):
    r"""
    deg : float
        rotation angle in degree,  
        positive : couter clockwise
        negative : clockwise
    """
    scale = 1.0
    if center is None:
        center  = (img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2

    (cX, cY) = center
    M = _cv2.getRotationMatrix2D(center=(cX, cY), angle=deg, scale=scale)
    img_rotated = _cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return img_rotated