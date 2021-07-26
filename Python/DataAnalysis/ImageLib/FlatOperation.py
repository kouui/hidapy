
import numpy

#-----------------------------------------------------------------------------
# detect position of local minimum of absorptoin line
#-----------------------------------------------------------------------------

def polyfit_center(x,y):
    r""" """
    coe =  numpy.polyfit(x,y, deg=2)
    return -0.5 * coe[1] / coe[0]
