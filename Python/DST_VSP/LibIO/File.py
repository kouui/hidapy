
#----------------------------------------------------------------------------
# file io
#----------------------------------------------------------------------------
from ..ImportAll import *


#----------------------------------------------------------------------------
# fits io
from astropy.io import fits as _fits
import warnings as _warnings
_warnings.simplefilter('ignore')
#----------------------------------------------------------------------------

def load_fits_(fname : str, dtype : str ='uint16', verbose : bool =True, lazy : bool = True):

    with _fits.open(fname, lazy_load_hdus=lazy) as hdul:
        hdul.verify('fix')
        header = hdul[0].header
        data = hdul[0].data.astype(dtype)

        if verbose:
            print(f"data size  : {data.nbytes/1024/1024/1024:.2f} [GB]")
            print(f"data type  : {data.dtype}")
            print(f"data shape : {data.shape}")
            print(f'Exposure   : {int(float(header["EXP"])*1000)} [ms]')
            print(f"Camera     : {header['CAMERA']}")
            print(f"Date Time  : {header['DATE_OB2']}")

    return header, data


def dump_fits_(img : T_ARRAY, dtype : str, fname : str, header : Optional[_fits.Header] = None):
    
    hdu_list = _fits.HDUList([])

    hdu = _fits.ImageHDU(img, header=header)
    hdu.scale(dtype)

    hdu_list.append(hdu)

    hdu_list.writeto(fname)

    print(f"saved as {fname}")


#-----------------------------------------------------------------------------
# json io
import json as _json
#-----------------------------------------------------------------------------


def dump_json_(data : Any, fname : str, indent : int = 2, ensure_ascii : bool = False):

    with open(fname, 'w') as f:
        _json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

def load_json_(fname : str):

    with open(fname) as f:
        data = _json.load(f)

    return data

#----------------------------------------------------------------------------
# hdf5 io
#----------------------------------------------------------------------------