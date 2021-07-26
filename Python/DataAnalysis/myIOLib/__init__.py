#-----------------------------------------------------------------------------
# pickle io
import pickle
#-----------------------------------------------------------------------------
def _dump_pickle(_data, _fname):

    with open(_fname, 'wb') as _f:
        pickle.dump(_data, _f, protocol=pickle.HIGHEST_PROTOCOL)

def _load_pickle(_fname):

    with open( _fname , 'rb') as _f:
        _data= pickle.load(_f)

    return _data

#-----------------------------------------------------------------------------
# json io
import json
#-----------------------------------------------------------------------------


def _dump_json(_data, _fname, _indent=2, _ensure_ascii=False):

    with open(_fname, 'w') as _f:
        json.dump(_data, _f, indent=_indent, ensure_ascii=_ensure_ascii)

def _load_json(_fname):

    with open(_fname) as _f:
        _data = json.load(_f)

    return _data

#-----------------------------------------------------------------------------
# fits io
from astropy.io import fits
import warnings
warnings.simplefilter('ignore')
#-----------------------------------------------------------------------------

def _load_fits(fname, dtype='uint16', verbose=True):
    r""" """

    with fits.open(fname) as hdul:
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

    return data

def _dump_fits(img, fname, header):
    r""" """
    hdu_list = fits.HDUList([])

    hdu = fits.ImageHDU(img, header=header)
    hdu_list.append(hdu)

    hdu_list.writeto(fname)
    print(f"saved as {fname}")
