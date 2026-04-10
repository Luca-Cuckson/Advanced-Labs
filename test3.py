import scipy
import numpy as np
import functions as func

nchannels = 16384
channelE = 0.533053
rho = 11.34 # g / cm^3
LLD = 30
cutoff = 8192
binwidth = 10
dpi = 300

r = 29.6
d = 68.564

channels_raw = np.linspace(0, nchannels, 16384)
channels_raw = channels_raw[LLD:]

Na22Counts = func.load_maestro_spe(r"DataFiles\Readings\Na22 Raised 95 mins x100 10th March.Spe")
Na22Counts = Na22Counts[LLD:]

def gauss(x, A, mu, sigma, C):
    return A*np.exp(-(x-mu)**2/(2*sigma**2)) + C

def Gaussing(fitfunc, x, y, nopeak, low, high, n):
    xdata = x[low:high]
    ydata = y[low:high] - nopeak

    p0 = [np.max(ydata), xdata[np.argmax(ydata)],  20, 0]
    
    params, cov = scipy.optimize.curve_fit(fitfunc, xdata, ydata, p0=p0) # Fit for Gaussian parameters
    A, mu, sigma, c = params # Extract Gaussian parameters
    perr = np.sqrt(np.diag(cov))
    A_err, mu_err, sigma_err, c_err = perr

    fit = gauss(xdata, A, mu, sigma, c)

    return A, mu, sigma, c, mu_err, sigma_err

def no_peaking(x, y, minE, low, high, maxE):  
    shwoopx = np.append(x[minE:low], x[high:maxE])
    shwoopy = np.append(y[minE:low], y[high:maxE])

    coeffs = np.polyfit(shwoopx, np.log10(shwoopy), deg=1)
    no_peak = 10 ** np.polyval(coeffs, x[low:high])

    #interp = scipy.interpolate.make_smoothing_spline(shwoopx, shwoopy)
    #no_peak = interp(x[low:high])

    return no_peak

def efficienting(A0, halflife, T, Y, r, d, Ncounts, Ncounts_err, t): # Need A0 in Bq (counts / second), halflife and T (age of source) can just be same unit, time of reading t in seconds
    r_err = 0.05
    d_err = 0.224
    
    N_emit = func.decay(A0, halflife, T) * Y * t

    G = (r ** 2) / (4 * d ** 2) # factors of pi cancel

    N_on = G * N_emit
    N_on_rerr = abs(((r + r_err) ** 2) / (4 * d ** 2) * N_emit - N_on)
    N_on_derr = abs(((r) ** 2) / (4 * (d + d_err) ** 2) * N_emit - N_on)
    N_on_err = np.sqrt(N_on_rerr**2 + N_on_derr**2)

    Efficiency = (Ncounts[0] / N_on) * 100

    N_err = ((Ncounts[0] + Ncounts_err) / N_on) * 100 - ((Ncounts[0] / N_on) * 100)
    G_err = ((Ncounts[0] / N_on) * 100) - ((Ncounts[0] / (N_on + N_on_err)) * 100)
    Error = np.sqrt(N_err**2 + G_err**2)
    return Efficiency, Error

def gauss_area(func, A, mu, sigma, C, botlim, uplim):
    area = scipy.integrate.quad(func, botlim, uplim, args=(A, mu, sigma, C))
    return area

#################################################################################################################

# Na22 First peak - 511 keV

minE, low, high, maxE = 795, 800, 1170, 1175

no_peak = no_peaking(channels_raw, Na22Counts, minE, low, high, maxE)

A, mu, sigma, c, mu_err, sigma_err = Gaussing(gauss, channels_raw, Na22Counts, no_peak, low, high, 102)
Ech = 511 / mu
FWHM = 2.355 * sigma
E_FWHM = FWHM * Ech
Ech_err = Ech * (mu_err / mu)
FWHM_err = 2.355 * sigma_err
E_FWHM_err = E_FWHM * np.sqrt((Ech_err/Ech)**2 + (FWHM_err/FWHM)**2)

print('Na511:', [mu, Ech, FWHM, E_FWHM])
print('Na511:', [mu_err, Ech_err, FWHM_err, E_FWHM_err])

#############################################################

area = gauss_area(gauss, A, mu, sigma, c, low, high)

area_error = gauss_area(gauss, A, mu+mu_err, sigma+sigma_err, c, low, high)
area_err = area_error[0] - area[0]
print('area:', area, '+/-', area_err)

Na511_Efficiency = efficienting(420000, 2.6, 9.6, 1.807, r, d, area, area_err, 5589)

print(Na511_Efficiency)