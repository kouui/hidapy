
#-----------------------------------------------------------------------------
# plotting functions for particular usage
#-----------------------------------------------------------------------------

from ..ImportAll import *

import matplotlib.pyplot as _plt
import numpy as _numpy

def show_image_and_hist_(img : T_ARRAY, name : Optional[str] = None, 
        vmin : Union[None, float, int]=None, 
        vmax : Union[None, float, int]=None,
        yscale : str = "log"):
    r""" """
    fig, axs = _plt.subplots(1,2, figsize=(7,3), dpi=100)

    ax = axs[0]
    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    if name is not None:
        ax.set_title(name)

    ax = axs[1]
    if vmin is None or vmax is None:
        bins = 51
    else:
        bins = _numpy.linspace(vmin,vmax,51)
    ax.hist( img.reshape(-1), bins=bins )

    ax.set_yscale(yscale)

    _plt.show()