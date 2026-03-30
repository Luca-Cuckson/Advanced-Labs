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

######################################################################################################################################################
# Trying scipy's peak search

print(scipy.signal.find_peaks(Na22Counts, width=(100,1000)))

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

# The interpolation function that finds the continued curve under the peak (assuming a straight line in log-scale)
def no_peaking(x, y, minE, low, high, maxE):  
    shwoopx = np.append(x[minE:low], x[high:maxE])
    shwoopy = np.append(y[minE:low], y[high:maxE])

    coeffs = np.polyfit(shwoopx, np.log10(shwoopy), deg=1)
    no_peak = 10 ** np.polyval(coeffs, x[low:high])
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

######################################################################################################################################################
# Chop of some peaks!
# Gonna do this in raw channels so then can also get a look at linearity of energy stuff

minE = 775
low = 780 
high = 1140 
maxE = 1181 

no_peak = no_peaking(channels_raw, Na22Counts, minE, low, high, maxE)

check_plot(Na22Counts, channels_raw, no_peak, minE, low, high, maxE, 2)
plt.savefig('Res&Eff/Na22_FirstPeak.svg', bbox_inches = 'tight')


# From Geeks for geeks for now:

xdata = channels_raw[low:high]
ydata = Na22Counts[low:high] - no_peak

p0 = [
    np.max(ydata),      # A
    xdata[np.argmax(ydata)],  # mu
    20,                 # sigma (guess)
    0                   # C
]

params, cov = scipy.optimize.curve_fit(gauss, xdata, ydata, p0=p0)

print(params)

A, mu, sigma, c = params

#parameters, _ = scipy.optimize.curve_fit(gauss, channels_raw[low:high], Na22Counts[low:high] - no_peak)
#params, cov = parameters
fit_y = gauss(channels_raw[low:high], A, mu, sigma, c)




plt.figure(100).add_axes((0, 0, 1.2, 0.68))
plt.bar(channels_raw[low:high], Na22Counts[low:high] - no_peak, width=1, alpha=0.3)
plt.plot(channels_raw[low:high], fit_y)
plt.savefig('Res&Eff/Na22_Peak.svg', bbox_inches = 'tight')
