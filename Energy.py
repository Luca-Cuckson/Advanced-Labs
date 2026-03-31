import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import functions as func


nchannels = 16384
channelE = 0.533053
rho = 11.34 # g / cm^3
LLD = 30
cutoff = 8192
binwidth = 10

channels = np.linspace(0, nchannels*channelE, 16384)
channels = channels[LLD:]

channels_raw = np.linspace(0, nchannels, 16384)
channels_raw = channels_raw[LLD:]

Na22Counts = func.load_maestro_spe(r"DataFiles\Readings\Na22 Raised 95 mins x100 10th March.Spe")
Na22Counts = Na22Counts[LLD:]

Co60Counts = func.load_maestro_spe(r"DataFiles\Readings\Co60 30 mins x100 19th Feb.Spe")
Co60Counts = Co60Counts[LLD:]

Cs137Counts = func.load_maestro_spe(r"DataFiles\Readings\Cs137 Raised 80 mins x100 12th March.Spe")
Cs137Counts = Cs137Counts[LLD:]

######################################################################################################################################################
# Trying scipy's peak search

print(scipy.signal.find_peaks(Na22Counts, width=(100,600)))

plt.figure(1).add_axes((0, 0, 1.2, 0.68))
plt.step(channels_raw, np.log10(Na22Counts + 1), linewidth=0.2)
plt.axvline(343, color='r', lw=0.5)
plt.axvline(975, color='r', lw=0.5)
plt.axvline(2403, color='r', lw=0.5)
plt.savefig('Res&Eff/check.svg', bbox_inches = 'tight')


# Well that clearly ain't working superbly 

######################################################################################################################################################
# Functions

def gauss(x, A, mu, sigma, C):
    return A*np.exp(-(x-mu)**2/(2*sigma**2)) + C

def double_gauss(x, A1, mu1, sigma1, A2, mu2, sigma2, C):
    return A1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + C

# The interpolation function that finds the continued curve under the peak (assuming a straight line in log-scale)
def no_peaking(x, y, minE, low, high, maxE):  
    shwoopx = np.append(x[minE:low], x[high:maxE])
    shwoopy = np.append(y[minE:low], y[high:maxE])

    coeffs = np.polyfit(shwoopx, np.log10(shwoopy), deg=1)
    no_peak = 10 ** np.polyval(coeffs, x[low:high])

    #interp = scipy.interpolate.make_smoothing_spline(shwoopx, shwoopy)
    #no_peak = interp(x[low:high])

    return no_peak

# Function to generate a plot to visually check if interpolation function is cutting under the peak correctly
def check_plot(counts, x, no_peak, minE, low, high, maxE, n):
    plt.figure(n).add_axes((0.05,0.05,1.2,0.68))
    plt.step(x, np.log10(counts+1), linewidth=0.2, alpha=0.6)
    plt.step(x[low:high], np.log10(no_peak+1), linewidth=0.2, alpha=0.6)
    plt.axvline(x[low], color='r', lw=0.2)
    plt.axvline(x[high], color='r', lw=0.2)
    plt.axvline(x[minE], color='b', lw=0.2)
    plt.axvline(x[maxE], color='b', lw=0.2)
    plt.axhline(0, color='k', lw=0.3)

def Gaussing(fitfunc, x, y, nopeak, low, high, n):
    xdata = x[low:high]
    ydata = y[low:high] - nopeak

    p0 = [np.max(ydata), xdata[np.argmax(ydata)],  20, 0]
    
    params, cov = scipy.optimize.curve_fit(fitfunc, xdata, ydata, p0=p0) # Fit for Gaussian parameters
    A, mu, sigma, c = params # Extract Gaussian parameters

    fit = gauss(xdata, A, mu, sigma, c)

    plt.figure(n).add_axes((0, 0, 1.2, 0.68))
    plt.bar(xdata, ydata, width=1, alpha=0.3)
    plt.plot(xdata, fit)
    return A, mu, sigma, c

######################################################################################################################################################
# Chop of some peaks!
# Gonna do this in raw channels so then can also get a look at linearity of energy stuff
# Na22 First peak - 511 keV

minE, low, high, maxE = 795, 800, 1170, 1175

no_peak = no_peaking(channels_raw, Na22Counts, minE, low, high, maxE)

check_plot(Na22Counts, channels_raw, no_peak, minE, low, high, maxE, 2)
plt.savefig('Res&Eff/Na22_FirstPeak.svg', bbox_inches = 'tight')

A, mu, sigma, c = Gaussing(gauss, channels_raw, Na22Counts, no_peak, low, high, 102)
plt.savefig('Res&Eff/Na22_Peak.svg', bbox_inches = 'tight')

Na511Ech = 511 / mu
Na511FWHM = 2.355 * sigma
Na511E_FWHM = Na511Ech * 0.530# * Na511FWHM

print([Na511Ech, Na511FWHM])

######################################################################################################################################################
# Na22 Second peak - 1275 keV

minE, low, high, maxE = 2000, 2130, 2700, 2701

no_peak = no_peaking(channels_raw, Na22Counts, minE, low, high, maxE)

check_plot(Na22Counts, channels_raw, no_peak, minE, low, high, maxE, 3)
plt.savefig('Res&Eff/Na22_SecondPeak.svg', bbox_inches = 'tight')

A, mu, sigma, c = Gaussing(gauss, channels_raw, Na22Counts, no_peak, low, high, 103)
plt.savefig('Res&Eff/Na22_Peak2.svg', bbox_inches = 'tight')

Na1275Ech = 1275 / mu
Na1275FWHM = 2.355 * sigma
Na1275E_FWHM = Na1275Ech * 0.530# * Na1275FWHM

print([Na1275Ech, Na1275FWHM])

######################################################################################################################################################
# Co60 First peak - 1173 keV

minE, low, high, maxE = 1904, 1905, 2750, 2751

no_peak = no_peaking(channels_raw, Co60Counts, minE, low, high, maxE)

check_plot(Co60Counts, channels_raw, no_peak, minE, low, high, maxE, 4)
plt.savefig('Res&Eff/Co60_FirstPeak.svg', bbox_inches = 'tight')

xdata = channels_raw[low:high]
ydata = Co60Counts[low:high] - no_peak

p0 = [9000, 2255, 120, 8000, 2560, 120, 0]
    
params, cov = scipy.optimize.curve_fit(double_gauss, xdata, ydata, p0=p0) # Fit for Gaussian parameters
A1, mu1, sigma1, A2, mu2, sigma2, C = params # Extract Gaussian parameters

fit1 = gauss(xdata, A1, mu1, sigma1, 0)
fit2 = gauss(xdata, A2, mu2, sigma2, 0)

plt.figure(104).add_axes((0, 0, 1.2, 0.68))
plt.bar(xdata, ydata, width=1, alpha=0.3)
plt.plot(xdata, fit1)
plt.plot(xdata, fit2)
plt.plot(xdata, fit1 + fit2)
plt.savefig('Res&Eff/Co60_Peak.svg', bbox_inches = 'tight')

Co1173Ech = 1173 / mu1
Co1173FWHM = 2.355 * sigma1
Co1173E_FWHM = Co1173Ech * 0.530# * Co1173FWHM

Co1332Ech = 1332 / mu2
Co1332FWHM = 2.355 * sigma2
Co1332E_FWHM = Co1332Ech * 0.530# * Co1332FWHM

print([Co1173Ech, Co1173FWHM])
print([Co1332Ech, Co1332FWHM])

######################################################################################################################################################
# Resolutions

Energies = np.array([511, 1173, 1275, 1332])
E_FWHMs = np.array([Na511E_FWHM, Co1173E_FWHM, Na1275E_FWHM, Co1332E_FWHM])

Resolutions = E_FWHMs / Energies * 100

plt.figure(200)
plt.plot(Energies, Resolutions)
plt.plot((Energies[0], Energies[-1]), (Resolutions[0], Resolutions[-1]), linestyle='dashed')
plt.savefig('Res&Eff/Resolution', bbox_inches='tight')