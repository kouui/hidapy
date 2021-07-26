#-----------------------------------------------------------------------------
# flat filed related functions
#-----------------------------------------------------------------------------

from astropy.utils import data
from numba.cuda.simulator.cudadrv.devicearray import check_array_compatibility
from ..ImportAll import *
import os
import numpy as _numpy

#-----------------------------------------------------------------------------
# stack flat for each angle of rotating waveplate
from astropy.io import fits as _fits
import warnings as _warnings
_warnings.simplefilter('ignore')
from ..LibIO.File import load_fits_ as _load_fits
from ..LibIO.File import dump_fits_ as _dump_fits_
#-----------------------------------------------------------------------------

def stack_flats_(fnames : List[str], n : int, prefix : str = ''):

    nF = len( fnames )
    _, img3d = _load_fits(fnames[0], verbose=False )
    nA, nrow, ncol = img3d.shape
    del img3d

    fnames_stack : List[str] = []

    #for kA in range(nA):
    for kA in (n,):
        flat_stack = _numpy.empty((nF, nrow, ncol), dtype="uint16")
        for kF in range(nF):
            fname = fnames[kF]
            with _fits.open(fname, lazy_load_hdus=True) as hdul:
                hdul.verify('fix')
                flat_stack[kF,:,:] = hdul[0].data[kA,:,:].astype("uint16")
        
        if len(prefix) > 0:
            save_path = os.path.join(prefix, f"flat_stack.{kA:03d}.fits")
            fnames_stack.append( save_path )
            _dump_fits_(flat_stack, "uint16", save_path, header=None)
            print(f"Angle No.{kA} : Saved as {save_path}")
            return fnames_stack, flat_stack
        else:
            return fnames_stack, flat_stack 

def flats_to_flat_field_(img3d : T_ARRAY, pad_x : int = 10, pad_y : int = 10):

    CHECK_NDIM_(img3d, "img3d", 3)

    flt_avg : T_ARRAY = img3d.mean(axis=0)
    flt_fld = _numpy.ones_like( flt_avg )
    flt_avg_sub = flt_avg[pad_y:-pad_y, pad_x:-pad_x]
    flt_fld[pad_y:-pad_y, pad_x:-pad_x] = flt_avg_sub / flt_avg_sub.mean(axis=1).reshape(-1,1)

    return flt_fld


        
        
#-----------------------------------------------------------------------------
# split and warp
# 1. split to left and right panel
# 2. a rough shift to align right panel to left panel
# 3. remove padding cols and rows in right panel (left panel also)
# 4. find hair line and absorption line poly2dfit for left and right simultaneously
# 5. find crossing points between hair line and absorption line
# 6. create rectangular coordinates as mapping target
# 7. perform warp on left and right panel based on coordinates mapping
from .Align import get_left_right_split_x_ as _get_left_right_split_x_
from .Align import remove_padding_ as _remove_padding_
from .Align import cross_to_rects_coords_map_ as _cross_to_rects_coords_map_
from .Align import calculate_warp_tform_spline_ as _calculate_warp_tform_spline_
from .Align import coords_warp_spline_ as _coords_warp_spline_

from .Fit import poly2dfit_curve_line_ as _poly2dfit_curve_line_
from .Fit import poly2d_cross_ as _poly2d_cross_ 

from ..LibImage.Measure import measure_shift2d_ as _measure_shift2d_
from ..LibImage.Shift import fft_shift_ as _fft_shift_

from dataclasses import dataclass
#-----------------------------------------------------------------------------

@dataclass
class Split_Warp_Params:

    # {left/right -> (x1,x2)}
    xr : Dict[str, Tuple[int,int]]
    dyx : T_ARRAY
    # {left/right ->  {h/v -> [ (xs, ys, coe), ... ]} }
    lines : Dict[str, Dict[str,List[Tuple[T_ARRAY,T_ARRAY,T_ARRAY]]]]
    # {left/right ->  {v1/v2 -> [ (y,x), ... ]} }
    cross : Dict[str, Dict[str,List[Tuple[int,int]]]]
    # {left/right ->  {L/R -> T_ARRAY(n_points,4)} } , 4 : (y_src,x_src,y_dst,x_dst)
    coords_map : Dict[str, Dict[str,T_ARRAY]]
    # {left/right ->  tform }
    tforms : Dict[str, List]

@dataclass
class Split_Warp_Panels:

    # original 2d image for calculating the parameters
    img : T_ARRAY
    # img after splitting left and right and shift
    panel : Dict[str,T_ARRAY]
    # panel after removing shift padding
    panel_pad1 : Dict[str,T_ARRAY]
    # panel_pad1 after warping to make 
    # hair line vertical and absorption lien horizontal
    panel_warp : Dict[str, T_ARRAY]


def split_and_warp_(img : T_ARRAY, sw_params : Split_Warp_Params):

    CHECK_NDIM_(img, "img", 2)

    # 1. split to left and right panel
    xr = sw_params.xr
    panel : Dict[str,T_ARRAY] = {}
    for name in ("left", "right"):
        xr0 = xr[name]
        panel[name]  = img[:,xr0[0]:xr0[1]]

    # 2. a rough shift to align right panel to left panel
    panel["right0"] = panel["right"].copy()
    dyx = sw_params.dyx
    panel["right"] = _fft_shift_(panel["right0"], dyx )

    # 3. remove padding cols and rows in right/left panel
    panel_pad1 : Dict[str,T_ARRAY] = {}
    panel_pad1["left"], panel_pad1["right"] = _remove_padding_(panel=panel["left"], panel_shift=panel["right"], shift=dyx)

    # 4. find hair line and absorption line poly2dfit for left and right simultaneously
    lines = sw_params.lines

    # 5. find cross coordinates
    cross = sw_params.cross

    # 6. create rectangular coordinates mapping
    coords_map = sw_params.coords_map

    # 7. perform warp on left and right panel based on coordinates mapping
    panel_warp : Dict[str, T_ARRAY]  = {}
    tforms = sw_params.tforms
    for name in ("left", "right"):
        #src = _numpy.append( coords_map[name]["L"][:,:2][:,::-1], coords_map[name]["R"][:,:2][:,::-1], axis=0 )
        #dst = _numpy.append( coords_map[name]["L"][:,2:][:,::-1], coords_map
        #nrow, ncol = panel_pad1[name].shape
        tform = tforms[name]
        img_warp : T_ARRAY = _coords_warp_spline_(panel_pad1[name], tform)
        panel_warp[name] = img_warp
        #tforms[name] = tform

    del name, tform

    sw_panels = Split_Warp_Panels(
        img = img,
        panel = panel,
        panel_pad1 = panel_pad1,
        panel_warp = panel_warp,
    )

    return sw_panels


def find_split_and_warp_params_(img3d : T_ARRAY, 
        xy_lines : Dict[str, List[int]],
        split_center : Tuple[int,int] = (1000,850), split_inner_offset : int = 40):

    CHECK_NDIM_(img3d, "img3d", 3)

    # 1. split to left and right panel
    xr = _get_left_right_split_x_(img3d[:,:,:], split_center, inner_offset=split_inner_offset)
    img = img3d.mean(axis=0)
    panel : Dict[str,T_ARRAY] = {}
    for name in ("left", "right"):
        xr0 = xr[name]
        panel[name]  = img[:,xr0[0]:xr0[1]]

    # 2. a rough shift to align right panel to left panel
    panel["right0"] = panel["right"].copy()
    dyx : T_ARRAY = _measure_shift2d_(panel["right0"], panel["left"])
    panel["right"] = _fft_shift_(panel["right0"], dyx )
    #del img

    # 3. remove padding cols and rows in right/left panel
    panel_pad1 : Dict[str,T_ARRAY] = {}
    panel_pad1["left"], panel_pad1["right"] = _remove_padding_(panel=panel["left"], panel_shift=panel["right"], shift=dyx)

    # 4. find hair line and absorption line poly2dfit for left and right simultaneously
    lines : Dict[str, Dict[str,List[Tuple[T_ARRAY,T_ARRAY,T_ARRAY]]]] = {
        "left"  : {"h" : [],"v" : [],},
       "right" : {"h" : [],"v" : [],},
    }
    #x1, x2 = 30, 770
    x1, x2 = xy_lines["x1_x2"]
    #for y0 in (2008-45, 1832-45, 1625-45, 1042-45, 296-45, 137-45):
    for y0 in xy_lines["y0s"]:
        for name in ("left", "right"):
            xs, ys, coe = _poly2dfit_curve_line_(panel_pad1[name], y0, x1, x2, step_size=20)
            lines[name]["h"].append( (xs, ys, coe) )
    del x1, x2, y0, xs, ys, coe, name

    #y1, y2 = 30, 2000
    y1, y2 = xy_lines["y1_y2"]
    #for x0 in (54,732):
    for x0 in xy_lines["x0s"]:
        for name in ("left", "right"):
            ys, xs, coe = _poly2dfit_curve_line_(panel_pad1[name], x0, y1, y2, step_size=50, kind=1)
            lines[name]["v"].append( (xs, ys, coe) )
    del y1, y2, x0, xs, ys, coe, name

    # 5. find cross coordinates

    cross : Dict[str, Dict[str,List[Tuple[int,int]]]] = {
       "left"  : { "v1" : [], "v2" : [],},
        "right" : { "v1" : [], "v2" : [],},
    }
    for name in ("left", "right"):
        count = 0
        for _, _, coe_v in lines[name]["v"]:
            count += 1
            for _, _, coe_h in lines[name]["h"]:
                cross[name][f"v{count}"].append( _poly2d_cross_(panel_pad1[name], coe_h, coe_v) )
    del count, name, coe_h, coe_v

    # 6. create rectangular coordinates mapping

    coords_map : Dict[str, Dict[str,T_ARRAY]] = {}
    for name in ("left", "right"):
        cross_v1 = cross[name]["v1"]
        cross_v2 = cross[name]["v2"]
        coe_v1 = lines[name]["v"][0][2]
        coe_v2 = lines[name]["v"][1][2]
        coords_map[name] = _cross_to_rects_coords_map_(cross_v1,cross_v2,coe_v1,coe_v2)
    del cross_v1, cross_v2, coe_v1, coe_v2, name

    # 7. perform warp on left and right panel based on coordinates mapping
    panel_warp : Dict[str, T_ARRAY]  = {}
    tforms : Dict[str, List] = {}
    for name in ("left", "right"):
        src = _numpy.append( coords_map[name]["L"][:,:2][:,::-1], coords_map[name]["R"][:,:2][:,::-1], axis=0 )
        dst = _numpy.append( coords_map[name]["L"][:,2:][:,::-1], coords_map[name]["R"][:,2:][:,::-1], axis=0 )
        #tform = LibPolAlign.calculate_warp_tform_piecewise_(src, dst)
        #img_warp = LibPolAlign.coords_warp_skimage_(panel_pad1[name], tform)
        nrow, ncol = panel_pad1[name].shape
        tform = _calculate_warp_tform_spline_(src[:,::-1], dst[:,::-1], output_region=[0,0,nrow-1,ncol-1])
        img_warp : T_ARRAY = _coords_warp_spline_(panel_pad1[name], tform)
        panel_warp[name] = img_warp
        tforms[name] = tform

    del name, src, dst, nrow, ncol, tform

    sw_params = Split_Warp_Params(
        xr = xr, 
        dyx = dyx, 
        lines= lines,
        cross = cross, 
        coords_map = coords_map, 
        tforms = tforms,
    )
    sw_panels = Split_Warp_Panels(
        img = img,
        panel = panel,
        panel_pad1 = panel_pad1,
        panel_warp = panel_warp,
    )

    return sw_params, sw_panels




