
#-----------------------------------------------------------------------------
# fits file I/O
from astropy.io import fits
import warnings
warnings.simplefilter('ignore')
#-----------------------------------------------------------------------------

def readfits(fname, dtype='uint16', verbose=True):
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
