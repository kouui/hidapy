import numpy
#-----------------------------------------------------------------------------
# plot
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------

def show_image_and_hist(img, name=None, vmin=None, vmax=None):
    r""" """
    fig, axs = plt.subplots(1,2, figsize=(9,4), dpi=100)

    ax = axs[0]
    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    if name is not None:
        ax.set_title(name)

    ax = axs[1]
    if vmin is None or vmax is None:
        bins = 51
    else:
        bins = numpy.linspace(vmin,vmax,51)
    ax.hist( img.reshape(-1), bins=bins )

    plt.show()


def move_axe(ax, dx, dy):
    r""" """
    pos1 = ax.get_position() # get the original position
    pos2 = [pos1.x0 + dx, pos1.y0+dy,  pos1.width, pos1.height]
    ax.set_position(pos2)

def remove_spline(*args,pos=("left","right","top","bottom")):
    r"""
    turn off the spline of all axes(*args)
    """

    for p in pos:
        assert p in ("left","right","top","bottom"), "bad position argument"

        for ax in args:
            ax.spines[p].set_visible(False)

def remove_tick_ticklabel(*args, kind="xy"):
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

def axes_no_padding(fig_kw={"figsize":(8,4),"dpi":100}, axe_kw={"ax1":[0,0,1,1]}):
    r""" """
    fig = plt.figure(figsize=fig_kw["figsize"], dpi=fig_kw["dpi"])

    axe_dict = {}
    for key, val in axe_kw.items():
        ax_ = fig.add_axes(val)
        axe_dict[key] = ax_
        remove_tick_ticklabel(ax_, kind="xy")
        remove_spline(ax_, pos=("left","right","top","bottom"))

    return fig, axe_dict
