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

background = func.load_maestro_spe(r"DataFiles\Backgrounds\Sr90 Gamma Raised 24 hours x100 26th Feb.Spe")
background = background[LLD:]

Pbcounts40 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.4mm Pb 21 hours x100 27th Feb.Spe")
Pbcounts40 = Pbcounts40[LLD:] * (84745 / 75043)
Pbcounts64 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.64mm Pb 44 hours x100 12th March.Spe")
Pbcounts64 = Pbcounts64[LLD:] * (84745 / 159519)
Pbcounts144 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 1.2mm Pb 69 hours x100 2nd March.Spe")
Pbcounts144 = Pbcounts144[LLD:] * (84745 / 244860)
Pbcounts282 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 2.82mm Pb 92 hours x100 10th March.Spe")
Pbcounts282 = Pbcounts282[LLD:] * (84745 / 332296)

Cucounts64 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.6mm Cu 25 hours x100 3rd March.Spe")
Cucounts64 = Cucounts64[LLD:] * (84745 / 89451)

Alcounts64 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.64mm Al 18 hours x100 4th March.Spe")
Alcounts64 = Alcounts64[LLD:] * (84745 / 64248)

Agcounts64 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.6mm Ag 27 hours x100 5th March.Spe")
Agcounts64 = Agcounts64[LLD:] * (84745 / 94912)

######################################################################################################################################################

file = r"DataFiles\AttenCoeffs\Pbattenuation_coeff.txt"
E, mu_rho = np.loadtxt(file, usecols=(0,1), unpack=True)
E = E*1e3

mu = mu_rho * rho

logE = np.log(E)
logmu = np.log(mu)

log_interp = scipy.interpolate.interp1d(logE, logmu, kind='linear', fill_value='extrapolate')
mu_vals = np.exp(log_interp(np.log(channels)))  # convert keV → MeV



x = func.binmean(channels, binwidth) # calculating the binned energy values for all measurements
#xish = func.binmean(channels, binwidth*2)

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
    plt.bar(x, np.log10(func.binsum(counts, binwidth)), width=5, alpha=0.6)
    plt.bar(x[low:high], np.log10(no_peak), width=5, alpha=0.6)
    plt.axvline(75, lw=0.8, color='r')
    plt.axvline(1461, lw=0.8, color='g')
    plt.axvline(x[low], lw=0.8, color='r', linewidth=0.1)
    plt.axvline(x[high], lw=0.8, color='r', linewidth=0.1)
    plt.axvline(x[minE], lw=0.8, color='b', linewidth=0.1)
    plt.axvline(x[maxE], lw=0.8, color='b', linewidth=0.1)
    plt.axhline(0, color='k', linewidth=0.3)

######################################################################################################################################################
# Pb 1.44 mm

minE = 100
low = 232 
high = 286 
maxE = 300 

y = func.binsum(Pbcounts144, binwidth)

no_peak = no_peaking(x, y, minE, low, high, maxE)

######################################################################################################################################################
# Ag 0.64 mm

y2 = func.binsum(Agcounts64, binwidth)

no_peak2 = no_peaking(x, y2, minE, low, high, maxE)

######################################################################################################################################################
# Al 0.59 mm

y3 = func.binsum(Alcounts64, binwidth)

no_peak3 = no_peaking(x, y3, minE, low, high, maxE)

######################################################################################################################################################
# Pb 0.64 mm

minE4 = 100
low4 = 240 
high4 = 300 
maxE4 = 312 

y4 = func.binsum(Pbcounts64, binwidth)

no_peak4 = no_peaking(x, y4, minE4, low4, high4, maxE4)

######################################################################################################################################################
# Background

y5 = func.binsum(background, binwidth)

no_peak5 = no_peaking(x, y5, minE, low, high, maxE)

######################################################################################################################################################

check_plot(Agcounts64, x, no_peak2, minE, low, high, maxE, 2)
plt.savefig('PeakGraphs/K40__Ag64.svg', bbox_inches = 'tight')

check_plot(Pbcounts144, x, no_peak, minE, low, high, maxE, 1)
plt.savefig('PeakGraphs/K40__Pb144.svg', bbox_inches = 'tight')

check_plot(Alcounts64, x, no_peak3, minE, low, high, maxE, 3)
plt.savefig('PeakGraphs/K40__Al59.svg', bbox_inches = 'tight')

check_plot(Pbcounts64, x, no_peak4, minE4, low4, high4, maxE4, 4)
plt.savefig('PeakGraphs/K40__Pb64.svg', bbox_inches = 'tight')

check_plot(background, x, no_peak5, minE, low, high, maxE, 5)
plt.savefig('PeakGraphs/K40__background.svg', bbox_inches = 'tight')



#labels = ['Pb 1.44 mm', 'Ag 0.64 mm', 'Al 0.59 mm', 'Pb 0.64 mm', 'Background']

plt.figure(100).add_axes((0.05,0.05,1.2,0.68))
plt.bar(x[low:high], y[low:high] - no_peak, width=5, alpha=0.3)
#plt.bar(x[low:high], y2[low:high] - no_peak2, width=5, alpha=0.3)
#plt.bar(x[low:high], y3[low:high] - no_peak3, width=5, alpha=0.3)
#plt.bar(x[low4:high4], y4[low4:high4] - no_peak4, width=5, alpha=0.3)
plt.bar(x[low:high], y5[low:high] - no_peak5, width=5, alpha=0.3)
plt.axvline(1398, lw=1)
plt.axvline(1402, lw=1)
#plt.legend(labels)
plt.savefig('PeakGraphs/K40__peaks.svg', bbox_inches = 'tight')



a = func.binsum(Pbcounts144, binwidth)
b = func.binsum(func.atten(background, mu_vals, 0.144), binwidth)
x2 = x * (1402/1398)

plt.figure(99).add_axes((0.05,0.05,1.2,0.68))
plt.bar(x2, np.log10(a+1), width=5, alpha=0.4)
plt.bar(x, np.log10(b+1), width=5, alpha=0.4)
plt.savefig('PeakGraphs/overlapped.svg', bbox_inches = 'tight')