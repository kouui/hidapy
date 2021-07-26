import numpy

#-----------------------------------------------------------------------------
# fit
from numpy import sin, cos, pi
from lmfit import Minimizer, Parameters
#print(f"lmfit version : {lmfit.__version__}")
#-----------------------------------------------------------------------------

def fitfunc_LinearPol(theta, p, thetaP, sign=1):

    theta_i = (theta[:] + p['alpha'])*pi/180. + pi*p['gamma']/p['T']
    d_thetaP = - ( thetaP - p["thetaP0"] )*pi/180.

    y  = p['C1I']
    y += p['C2I'] * cos( 2*d_thetaP ) * sign
    y += p['C5I'] * sin( 2*theta_i - 2*d_thetaP ) * sign
    #y -= p['Ac2'] * cos( 2*theta_i ) * sign
    #y += p['C6I'] * sin( 4*theta_i ) * sign
    y += p['C6I'] * cos( 4*theta_i - 2*d_thetaP ) * sign

    y += p['As1'] * sin( theta_i ) * sign
    y += p['Ac1'] * cos( theta_i ) * sign

    return y

def fitres_LinearPol(p, theta, y_data, thetaP, mask):
    r""" """
    residual = numpy.zeros(y_data.shape, dtype='float32')

    if y_data.ndim == 1:

        y_model = fitfunc_LinearPol(theta, p, thetaP)
        residual[:] = y_model[:] * mask - y_data[:] * mask

    elif y_data.ndim == 2:

        for k in range(y_data.shape[0]):
            y_model = fitfunc_LinearPol(theta, p, thetaP[k])
            residual[k,:] = y_model[:] * mask - y_data[k,:] * mask

    else:
        raise ValueError("y_data should have ndim equal to 1 or 2")

    return residual

def fit_LinearPol(theta, y_data, thetaP, mask=1, init_p=None,
                  gamma=0.2, T=4, C1I=2E4, C2I=50, C5I=5E2, C6I=1E4):
    r""" """

    if init_p is None:
        pfit = Parameters()
        pfit.add(name='gamma', value=gamma,  vary=False)
        pfit.add(name='T',     value=T,     vary=False)
        pfit.add(name='alpha', value=0,    vary=True, min=-45, max=45)
        pfit.add(name='As1',   value=0,  vary=True)
        pfit.add(name='Ac1',   value=0,  vary=True)
        pfit.add(name='C1I',   value=C1I,  vary=True, min=0)
        pfit.add(name='C2I',   value=C2I,    vary=True, min=0)
        pfit.add(name='C5I',   value=C5I,  vary=True, min=0)
        pfit.add(name='C6I',   value=C6I,  vary=True, min=0)
        pfit.add(name='thetaP0',   value=90,  vary=True, min=0, max=180)
        #pfit.add(name='Ac4',   expr='Ac2 * As4 / As2')
    else:
        pfit = init_p

    mini = Minimizer(fitres_LinearPol, pfit, fcn_args=(theta, y_data, thetaP, mask))
    out = mini.minimize(method='leastsq')
    best_fit = y_data + out.residual.reshape(y_data.shape)

    return out, best_fit

def fitfunc_LinearPol_v0(theta, p, sign=1):

    theta_i = (theta[:] + p['alpha'])*pi/180. + pi*p['gamma']/p['T']

    y  = p['C']
    y += p['As2'] * sin( 2*theta_i ) * sign
    y -= p['Ac2'] * cos( 2*theta_i ) * sign
    y += p['As4'] * sin( 4*theta_i ) * sign
    y += p['Ac4'] * cos( 4*theta_i ) * sign

    y += p['As1'] * sin( theta_i ) * sign
    y += p['Ac1'] * cos( theta_i ) * sign

    return y

def fitres_LinearPol_v0(p, theta, y_data, mask):
    r""" """
    residual = numpy.zeros(y_data.shape, dtype='float32')

    if y_data.ndim == 1:

        y_model = fitfunc_LinearPol_v0(theta, p)
        residual[:] = y_model[:] * mask - y_data[:] * mask

    elif y_data.ndim == 2:

        for k in range(y_data.shape[0]):
            y_model = fitfunc_LinearPol_v0(theta, p)
            residual[k,:] = y_model[:] * mask - y_data[k,:] * mask

    else:
        raise ValueError("y_data should have ndim equal to 1 or 2")

    return residual

def fit_LinearPol_v0(theta, y_data, mask=1, init_p=None,
                  gamma=0.2, T=4, C=1.5E4, As2=50, Ac2=50, As4=5E3, Ac4=1E4):
    r""" """

    if init_p is None:
        pfit = Parameters()
        pfit.add(name='gamma', value=gamma,  vary=False)
        pfit.add(name='T',     value=T,     vary=False)
        pfit.add(name='alpha', value=0,    vary=True, min=-45, max=45)
        pfit.add(name='As1',   value=0,  vary=True)
        pfit.add(name='Ac1',   value=0,  vary=True)
        pfit.add(name='C',     value=C,  vary=True)#, min=0)
        pfit.add(name='As2',   value=As2,    vary=True, min=0)
        pfit.add(name='Ac2',   value=Ac2,  vary=True)#, min=0)
        pfit.add(name='As4',   value=As4,  vary=True)#, min=0)
        pfit.add(name='Ac4',   value=Ac4,  vary=True, min=0)
        pfit.add(name='thetaP0',   value=45,  vary=True, min=0, max=180)
        #pfit.add(name='Ac4',   expr='Ac2 * As4 / As2')
    else:
        pfit = init_p

    mini = Minimizer(fitres_LinearPol_v0, pfit, fcn_args=(theta, y_data, mask))
    out = mini.minimize(method='leastsq')
    best_fit = y_data + out.residual.reshape(y_data.shape)

    return out, best_fit


#-----------------------------------------------------------------------------
# calculate waveplate
from numpy import arctan, arccos
#-----------------------------------------------------------------------------

def get_waveplate_param(theta, Ip, Im, alpha, T, gamma, xlim, n_period):
    r""" """
    parp, yp = fit_curve_fft(theta, Ip, alpha=alpha, gamma=gamma, T=T, xlim=xlim, n_period=n_period)
    parm, ym = fit_curve_fft(theta, Im, alpha=alpha, gamma=gamma, T=T, xlim=xlim, n_period=n_period)

    Cp = parp['C']
    Cm = parm['C']
    A4p = parp["Ac4"]
    A4m = parm["Ac4"]
    A2p = parp["As2"]
    A2m = parm["As2"]

    f1 = Cp*A4m + Cm*A4p
    f2 = 2*A4p*A4m
    f3 = T / (4*pi*gamma)

    delta = arccos( (f3 * f1 * sin(1/f3) - f2) / (f3 * f1 * sin(1/f3) + f2) )

    deltap = arctan( cos(0.5/f3) * (cos(delta) - 1.) / (2 * sin(delta)) * A2p / A4p )

    Rp_Rm = - A4p / A4m

    return delta, deltap, Rp_Rm

#-----------------------------------------------------------------------------
# waveplate parameters to Constants
#-----------------------------------------------------------------------------
def wp_param_To_const(wpc, T, gamma):
    r""" """
    fac = 2 * pi / T
    dlt = wpc["delta"] * pi / 180
    dltp = wpc["deltap"] * pi / 180

    C_ = numpy.zeros([7], dtype='float32')

    C_[1] = gamma
    C_[2] = gamma * cos(dltp) * 0.5 * (1+cos(dlt))
    C_[3] = - gamma * sin(dltp) * cos(dlt)
    C_[4] = - 1/fac * sin(fac*gamma) * cos(dltp) * sin(dlt)
    C_[5] = - 1/fac * sin(fac*gamma) * sin(dltp) * sin(dlt)
    C_[6] = 0.5/fac * sin(2*fac*gamma) * cos(dltp) * 0.5 * ( 1-cos(dlt) )

    return C_
#-----------------------------------------------------------------------------
# general dualfit
#-----------------------------------------------------------------------------

def dualfitfunc_Normal(theta, p, C_):

    theta_i = (theta[:] + p['alpha'])*pi/180. + pi*p['gamma']/p['T']

    y  = 0
    #y += C_[1]
    y += C_[2] * p['Q_I']
    y += C_[3] * p['V_I']
    y += (C_[4] * p['V_I'] + C_[5] * p['Q_I']) * sin( 2*theta_i )
    y -= C_[5] * p['U_I'] * cos( 2*theta_i )
    y += C_[6] * p['U_I'] * sin( 4*theta_i )
    y += C_[6] * p['Q_I'] * cos( 4*theta_i )

    return y

def dualfitres_Normal(p, theta, y_data, C_, mask):
    r""" """
    residual = numpy.zeros(y_data.shape, dtype='float32')

    if y_data.ndim == 1:

        y_model = dualfitfunc_Normal(theta, p, C_)
        residual[:] = y_model[:] * mask - y_data[:] * mask

    elif y_data.ndim == 2:

        for k in range(y_data.shape[0]):
            y_model = dualfitfunc_Normal(theta, p, C_)
            residual[k,:] = y_model[:] * mask - y_data[k,:] * mask

    else:
        raise ValueError("y_data should have ndim equal to 1 or 2")

    return residual

def dualfit_Normal(theta, y_data, C_, mask=1, init_p=None,
                  gamma=0.2, T=5, RpI=1.5E4, Q_I=0.5, U_I=0.5, V_I=0):
    r""" """

    if init_p is None:
        pfit = Parameters()
        pfit.add(name='gamma', value=gamma,  vary=False)
        pfit.add(name='T',     value=T,     vary=False)
        pfit.add(name='alpha', value=0,    vary=True, min=-45, max=45)
        #pfit.add(name='As1',   value=0,  vary=True)
        #pfit.add(name='Ac1',   value=0,  vary=True)
        pfit.add(name='RpI',     value=RpI,  vary=True, min=0)
        pfit.add(name='Q_I',   value=Q_I,    vary=True, min=-1, max=1)
        pfit.add(name='U_I',   value=U_I,  vary=True, min=-1, max=1)
        pfit.add(name='V_I',   value=V_I,  vary=True, min=-1, max=1)
    else:
        pfit = init_p

    mini = Minimizer(dualfitres_Normal, pfit, fcn_args=(theta, y_data, C_, mask))
    out = mini.minimize(method='leastsq')
    best_fit = y_data + out.residual.reshape(y_data.shape)

    return out, best_fit

def dualfit_fft(theta, y, alpha, T, gamma, xlim, n_period, C_):
    r""" """
    params, y_fft_ = fit_curve_fft(theta, y, alpha=alpha, gamma=gamma, T=T, xlim=xlim, n_period=n_period)
    Stokes = {
        "RpQ" : params['Ac4'] / C_[6],
        "RpU" : params["As4"] / C_[6],
    }
    Stokes["RpV"] = (params["As2"] - Stokes["RpQ"] * C_[5]) / C_[4]

    return params, y_fft_, Stokes

#-----------------------------------------------------------------------------
# process one .fits file
from . import fitsIO

import sys
_LIB_DIR_ = '/home/kouui/Libs/Python/DataAnalysis/'
if _LIB_DIR_ not in sys.path:
    sys.path.append(_LIB_DIR_)

from ImageLib import FFTbase
import myIOLib
#-----------------------------------------------------------------------------
def demodulate_single_fits(fname, dark, imc, wpc, C_, outFile, T, gamma, n_sample, xlim, binx=2, biny=2, hasFlat=True, imgL=None, imgR=None, isSave=True):
    r""" """

    if imgL is None and imgR is None:

        arr = fitsIO.readfits(fname,verbose=False).astype('float32') - dark[:,:]

        nT = arr.shape[0]
        nX = imc['xL'][1]-imc['xL'][0]
        nY = 2048
        imgL = numpy.zeros((2048,nX,nT), 'float32')
        imgR = numpy.zeros((2048,nX,nT), 'float32')
        if hasFlat:
            for k in range(nT):
                img = FFTbase.transform_img(arr[k,:,:], angle=imc["rot-angle"])
                imgL[:,:,k] = img[:,imc['xL'][0]:imc['xL'][1]] / imc["flatL"][:,:-1]
                imgR0 = img[:,imc['xR'][0]:imc['xR'][1]].copy() / imc["flatR"][:,:-1]
                imgR[:,:,k] = FFTbase.shift2d(imgR0, -imc["xoff"], -imc["yoff"])
        else:
            for k in range(nT):
                img = FFTbase.transform_img(arr[k,:,:], angle=imc["rot-angle"])
                imgL[:,:,k] = img[:,imc['xL'][0]:imc['xL'][1]]# / imc["flatL"][:,:-1]
                imgR0 = img[:,imc['xR'][0]:imc['xR'][1]].copy()# / imc["flatR"][:,:-1]
                imgR[:,:,k] = FFTbase.shift2d(imgR0, -imc["xoff"], -imc["yoff"])
    else:
        nT = imgL.shape[2]
        nX = imc['xL'][1]-imc['xL'][0]
        nY = 2048

    n_period = n_sample // (T//gamma)
    theta = numpy.linspace(0, n_period*360, n_sample, endpoint=False)
    #C_ = wp_param_To_const(wpc, T, gamma)

    xs = 50, nX-50
    ys = int(-imc["yoff"]) + 50, nY-50
    nx = xs[1] - xs[0]
    ny = ys[1] - ys[0]

    if binx > 1 and biny > 1:
        nx_real = nx // binx
        ny_real = ny // biny
        outArray = numpy.zeros((4,ny_real,nx_real),dtype='float32')
        for i in range(ny_real):
            for j in range(nx_real):
                Im = imgL[ys[0]+i*biny:ys[0]+i*biny+biny,xs[0]+j*binx:xs[0]+j*binx+binx,:].mean(axis=(0,1))
                Ip = imgR[ys[0]+i*biny:ys[0]+i*biny+biny,xs[0]+j*binx:xs[0]+j*binx+binx,:].mean(axis=(0,1))

                y_data = Ip-Im*wpc["R+_R-"]
                _, y_fft, stok = dualfit_fft(theta, y_data, wpc['alpha'], T, gamma, xlim=xlim, n_period=n_period, C_=C_)

                RpI = (Ip+Im*wpc["R+_R-"]).mean() / C_[1]
                stok_vec = numpy.ones(4, dtype='float32')
                stok_vec[1] = stok["RpQ"] / RpI
                stok_vec[2] = stok["RpU"] / RpI
                stok_vec[3] = stok["RpV"] / RpI

                outArray[:,i,j] = stok_vec[:]


    else:
        raise ValueError("binx > 1 and biny > 1")

    #---
    if isSave:
        header = myIOLib.fits.Header()
        header["xrLeft"] = f"{imc['xL'][0]+xs[0]},{imc['xL'][0]+xs[0]+nx-(nx%binx)}"
        header["xrRight"] = f"{imc['xR'][0]+xs[0]},{imc['xR'][0]+xs[0]+nx-(nx%binx)}"
        header["yr"] = f"{ys[0]},{ys[1]-(ny%biny)}"
        header["binx"] = str(binx)
        header["biny"] = str(biny)

        myIOLib._dump_fits(outArray, outFile, header)
    #---
    return outArray

#-----------------------------------------------------------------------------
# fft
from scipy.integrate import simps
#-----------------------------------------------------------------------------
def fft_amplitude(_theta, _y, _n, _kind):
    r"""

    Parameters
    -----------
    _theta : float, 1darray, [:math:`rad`]
        rotation angle

    _y : float, 1darray,
        data

    _n : int
        number of period in array _theta

    _kind : str,
        specify which component you want to compute
        'c', 's1', 'c1', 's2', 'c2', 's4', 'c4'

    """

    if _kind == 'c':
        _f = lambda _x : 1
    elif _kind == 's1':
        _f = lambda _x : sin(_x)
    elif _kind == 'c1':
        _f = lambda _x : cos(_x)
    elif _kind == 's2':
        _f = lambda _x : sin(2*_x)
    elif _kind == 'c2':
        _f = lambda _x : cos(2*_x)
    elif _kind == 's4':
        _f = lambda _x : sin(4*_x)
    elif _kind == 'c4':
        _f = lambda _x : cos(4*_x)
    else:
        raise ValueError("_kind should be in ('c', 's1', 'c1', 's2', 'c2', 's4', 'c4')")

    _integral = simps(_y[:]*_f(_theta), x=_theta)

    if _kind == 'c':
        _integral *= 0.5

    return _integral / (_n*pi)

def fit_curve_fft(th, y, alpha=18, gamma=0.2, T=4, xlim=240, n_period=1):
    r""" """

    params = {"alpha":alpha, "gamma":gamma, "T":T}
    x_fft_ = (th[:]+params['alpha'])*pi/180. + pi*params['gamma']/params['T']

    params["C"]   = fft_amplitude(x_fft_[:xlim], y[:xlim], n_period, 'c')
    params["As1"] = fft_amplitude(x_fft_[:xlim], y[:xlim], n_period, 's1')
    params["Ac1"] = fft_amplitude(x_fft_[:xlim], y[:xlim], n_period, 'c1')
    params["As2"] = fft_amplitude(x_fft_[:xlim], y[:xlim], n_period, 's2')
    params["Ac2"] = fft_amplitude(x_fft_[:xlim], y[:xlim], n_period, 'c2')
    params["As4"] = fft_amplitude(x_fft_[:xlim], y[:xlim], n_period, 's4')
    params["Ac4"] = fft_amplitude(x_fft_[:xlim], y[:xlim], n_period, 'c4')

    theta_i = x_fft_
    y_fft_  = params['C']
    y_fft_ += params['As2'] * sin( 2*theta_i )
    y_fft_ += params['Ac2'] * cos( 2*theta_i )
    y_fft_ += params['As4'] * sin( 4*theta_i )
    y_fft_ += params['Ac4'] * cos( 4*theta_i )
    y_fft_ += params['As1'] * sin( theta_i )
    y_fft_ += params['Ac1'] * cos( theta_i )

    return params, y_fft_
