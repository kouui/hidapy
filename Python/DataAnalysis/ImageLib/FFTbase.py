
#-----------------------------------------------------------------------------
# image operation
from image_registration.fft_tools.shift import shift2d
from image_registration import chi2_shift
from imreg_dft.imreg import transform_img
#-----------------------------------------------------------------------------

def shift1d(arr, delta):
    r""" """

    arr2d = arr.reshape(-1,1)
    deltax = 0
    deltay = delta
    arr2d_shifted = shift2d(arr2d, deltax, deltay)

    return arr2d_shifted.reshape(-1)
