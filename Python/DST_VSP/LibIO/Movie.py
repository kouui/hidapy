
#----------------------------------------------------------------------------
# save/write animation
#----------------------------------------------------------------------------

from ..ImportAll import *

import matplotlib.pyplot as _plt
import os

#----------------------------------------------------------------------------
# gif
from moviepy.editor import ImageSequenceClip as _ImageSequenceClip
#----------------------------------------------------------------------------

def img3d_to_gif_(img3d : T_ARRAY, fname : str, fps : int = 10, scale : float = 1.0):

    clip = _ImageSequenceClip( list( img3d.reshape( img3d.shape + (1,) ) ), fps=fps).resize(scale)
    clip.write_gif(fname)

    print(f"Saved as {fname}")

def img3d_to_mp4_(img3d : T_ARRAY, file_path : str, fps : int = 10, vmin : int = 0, vmax : int = 2**16, cmap : str = "gray"):

    CHECK_NDIM_(img3d, "img3d", 3)

    fig = _plt.figure(figsize=(6,6), dpi=100)
    ax = fig.add_axes([0,0,1,1])
    fnames = []
    for k in range(img3d.shape[0]):
        ax.cla()
        ax.imshow( img3d[k,:,:], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax )
        fname = os.path.join( os.path.dirname(file_path), f"{k:04d}.png" )
        fig.savefig(fname, dpi=100)
        fnames.append(fname)
    _plt.close()

    
    os.system(f"ffmpeg -r {fps} -f image2 -i '{os.path.dirname(file_path)}/%04d.png' -qscale 0 {file_path}")

    for fname in fnames:
        os.remove(fname)

    

    

    

