

#-----------------------------------------------------------------------------
# figure and axe manipulation @ matploltib
#-----------------------------------------------------------------------------
from ..ImportAll import *

#-----------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------

def move_axe(ax : T_AXE, dx : T_IF, dy : T_IF):
    r""" """
    pos1 = ax.get_position() # get the original position
    pos2 = [pos1.x0 + dx, pos1.y0+dy,  pos1.width, pos1.height]
    ax.set_position(pos2)

def remove_spline(*args : str, pos : Tuple[str,...] =("left","right","top","bottom")):
    r"""
    turn off the spline of all axes(*args)
    """

    for p in pos:
        assert p in ("left","right","top","bottom"), "bad position argument"

        for ax in args:
            ax.spines[p].set_visible(False)

def remove_tick_ticklabel(*args : T_AXE, kind : str ="xy"):
    r"""
    turn off x/y ticks and ticklables.

    4 |                |
    3 |                |
    2 |          --->  |
    1 |                |
      +------+         +-------+
      0   1   2
    """
    if 'x' in kind:
        for ax in args:
            ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                labelbottom=False,)
    if 'y' in kind:
        for ax in args:
            ax.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,         # ticks along the left edge are off
                labelleft=False)

def axes_no_padding(fig_kw : Dict[str,Any]={"figsize":(8,4),"dpi":100}, axe_kw : Dict[str,List]={"ax1":[0,0,1,1]}):
    r""" """
    fig = _plt.figure(figsize=fig_kw["figsize"], dpi=fig_kw["dpi"])

    axe_dict = {}
    for key, val in axe_kw.items():
        ax_ = fig.add_axes(val)
        axe_dict[key] = ax_
        remove_tick_ticklabel(ax_, kind="xy")
        remove_spline(ax_, pos=("left","right","top","bottom"))

    return fig, axe_dict
