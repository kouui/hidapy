
#----------------------------------------------------------------------------
# save/write animation
#----------------------------------------------------------------------------

from ..ImportAll import *


#----------------------------------------------------------------------------
# gif
from moviepy.editor import ImageSequenceClip as _ImageSequenceClip
#----------------------------------------------------------------------------

def img3d_to_gif_(img3d : T_ARRAY, fname : str, fps : int = 10, scale : float = 1.0):

    clip = _ImageSequenceClip( list( img3d.reshape( img3d.shape + (1,) ) ), fps=fps).resize(scale)
    clip.write_gif(fname)

    print(f"Saved as {fname}")

