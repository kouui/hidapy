#-----------------------------------------------------------------------------
# rebin array to produce a smaller or bigger array
# reference : https://scipy-cookbook.readthedocs.io/items/Rebinning.html
#-----------------------------------------------------------------------------


from ..ImportAll import *

import numpy as _numpy

#-----------------------------------------------------------------------------
# Here we deal with the simplest case where any desired new shape is valid and 
# no interpolation is done on the data to determine the new values. 
# * First, floating slices objects are created for each dimension. * Second, the 
# coordinates of the new bins are computed from the slices using mgrid. 
# * Then, coordinates are transformed to integer indices. * And, finally, 
# 'fancy indexing' is used to evaluate the original array at the desired indices.
#-----------------------------------------------------------------------------
from numpy.lib.index_tricks import mgrid as _mgrid

def rebin_naive_(a : T_ARRAY, newshape : Tuple[int,...]) -> T_ARRAY:
    """
    rebin a arra to a new shape
    """
    assert len(a.shape) == len(newshape)

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = _mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]

def rebin_naive_factor_(a : T_ARRAY, newshape : Tuple[int,...]) -> T_ARRAY:
    """
    rebin an array to a new shape,
    newshape must be a factor of a.shape
    """
    assert len(a.shape) == len(newshape)
    assert not _numpy.sometrue(_numpy.mod( a.shape, newshape ))

    slices = [ slice(None,None, old/new) for old,new in zip(a.shape,newshape) ]
    return a[slices]

#-----------------------------------------------------------------------------
# Here is an other way to deal with the reducing case for arrs. 
# This acts identically to IDL's rebin command where all values in the 
# original array are summed and divided amongst the entries in the new array. 
# As in IDL, the new shape must be a factor of the old one. 
# The ugly 'evList trick' builds and executes a python command of the form

# reference : https://gist.github.com/derricw/95eab740e1b08b78c03f
#-----------------------------------------------------------------------------

def rebin_(arr : T_ARRAY, newshape : Tuple[int,...], operation : str = "mean"):
    """
    Bins an arr in all axes based on the target shape, by summing or
        averaging.
    Number of output dimensions must match number of input dimensions.
    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = rebin_(m, newshape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if arr.ndim != len(newshape):
        raise ValueError("Shape mismatch: {} -> {}".format(arr.shape,
                                                           newshape))
    compression_pairs = [(d, c//d) for d, c in zip(newshape,
                                                   arr.shape)]
    flattened = [l for p in compression_pairs for l in p]
    arr = arr.reshape(flattened)
    for i in range(len(newshape)):
        if operation.lower() == "sum":
            arr = arr.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            arr = arr.mean(-1*(i+1))
    return arr

#-----------------------------------------------------------------------------
# A python version of congrid, used in IDL, 
# for resampling of data to arbitrary sizes, 
# using a variety of nearest-neighbour and interpolation routines.
#-----------------------------------------------------------------------------
from ._Congrid import congrid_

