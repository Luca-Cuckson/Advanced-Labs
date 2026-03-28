import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import functions as func


nchannels = 16384
channelE = 0.533053
#thick = 0.04 # cm
rho = 11.34 # g / cm^3
LLD = 30
cutoff = 8192
binwidth = 10

channels = np.linspace(0, nchannels*channelE, 16384)
channels = channels[LLD:]

Pbcounts40 = func.load_maestro_spe("Sr90 Brem 0.4mm Pb 21 hours x100 27th Feb.Spe")
Pbcounts40 = Pbcounts40[LLD:] * (84745 / 75043)
Pbcounts144 = func.load_maestro_spe("Sr90 Brem 1.2mm Pb 69 hours x100 2nd March.Spe")
Pbcounts144 = Pbcounts144[LLD:] * (84745 / 244860)
Pbcounts282 = func.load_maestro_spe("Sr90 Brem 2.82mm Pb 92 hours x100 10th March.Spe")
Pbcounts282 = Pbcounts282[LLD:] * (84745 / 332296)

Cucounts64 = func.load_maestro_spe("Sr90 Brem 0.6mm Cu 25 hours x100 3rd March.Spe")
Cucounts64 = Cucounts64[LLD:] * (84745 / 89451)

Alcounts64 = func.load_maestro_spe("Sr90 Brem 0.64mm Al 18 hours x100 4th March.Spe")
Alcounts64 = Alcounts64[LLD:] * (84745 / 64248)

Agcounts64 = func.load_maestro_spe("Sr90 Brem 0.6mm Ag 27 hours x100 5th March.Spe")
Agcounts64 = Agcounts64[LLD:] * (84745 / 94912)

######################################################################################################################################################

file = 'Pbattenuation_coeff.txt'
E, mu_rho = np.loadtxt(file, usecols=(0,1), unpack=True)
E = E*1e3

mu = mu_rho * rho

logE = np.log(E)
logmu = np.log(mu)

log_interp = scipy.interpolate.interp1d(logE, logmu, kind='linear', fill_value='extrapolate')
mu_vals = np.exp(log_interp(np.log(channels)))  # convert keV → MeV



x = func.binmean(channels, binwidth)
#xish = func.binmean(channels, binwidth*2)

def gauss(x, A, mu, sigma, C):
    return A*np.exp(-(x-mu)**2/(2*sigma**2)) + C

def no_peaking(x, y, minE, low, high, maxE):
    shwoopx = np.append(x[minE:low], x[high:maxE])
    shwoopy = np.append(y[minE:low], y[high:maxE])

    coeffs = np.polyfit(shwoopx, np.log10(shwoopy), deg=1)
    no_peak = 10 ** np.polyval(coeffs, x[low:high])
    return no_peak

def check_plot(counts, x, no_peak, minE, low, high, maxE):
    plt.figure(2).add_axes((0.05,0.05,1.2,0.68))
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
#yish = func.binsum(Pbcounts144, binwidth)


shwoopx = np.append(x[minE:low], x[high:maxE])
shwoopy = np.append(y[minE:low], y[high:maxE])


#interp = scipy.interpolate.make_interp_spline(shwoopx, shwoopy, k=3)
#no_peak = interp(func.binmean(channels, binwidth)[low:high])

interp = scipy.interpolate.PchipInterpolator(shwoopx, np.log(shwoopy))
no_peak = interp(x[low:high])

######################################################################################################################################################
# Ag 0.64 mm

y2 = func.binsum(Agcounts64, binwidth)

no_peak2 = no_peaking(x, y2, minE, low, high, maxE)

#yish2 = func.binsum(Agcounts64, binwidth)


#shwoopx2 = np.append(x[minE:low], x[high:maxE])
#shwoopy2 = np.append(y[minE:low], y[high:maxE])


#shwoopx2 = np.append(x[minE:low], x[high:maxE])
#shwoopy2 = np.append(y2[minE:low], y2[high:maxE])

#from scipy.interpolate import UnivariateSpline

#spline2 = UnivariateSpline(shwoopx2, shwoopy2, s=0.5)  # adjust s
#no_peak2 = spline2(x[low:high])

#interp2 = scipy.interpolate.make_interp_spline(shwoopx2, np.log10(shwoopy2), k=1)
#no_peak2 = 10**interp2(x[low:high])

#coeffs2 = np.polyfit(shwoopx2, np.log10(shwoopy2), deg=1)
#no_peak2 = 10 ** np.polyval(coeffs2, x[low:high])


#interp2 = scipy.interpolate.PchipInterpolator(shwoopx2, shwoopy2)
#no_peak2 = interp2(x[low:high])

######################################################################################################################################################
# Al 0.59 mm

y3 = func.binsum(Alcounts64, binwidth)

#yish3 = func.binsum(Alcounts64, binwidth)


shwoopx3 = np.append(x[minE:low], x[high:maxE])
shwoopy3 = np.append(y3[minE:low], y3[high:maxE])


#shwoopx3 = np.append(x[minE:low], x[high:maxE])
#shwoopy3 = np.append(y3[minE:low], y3[high:maxE])


interp3 = scipy.interpolate.PchipInterpolator(shwoopx3, shwoopy3)
no_peak3 = interp3(x[low:high])

######################################################################################################################################################

check_plot(Agcounts64, x, no_peak2, minE, low, high, maxE)
plt.savefig('K40__Ag64.svg', bbox_inches = 'tight')



#plt.figure(6).add_axes((0.05,0.05,1.2,0.68))
#plt.bar(x, np.log10(func.binsum(Pbcounts144, binwidth)), width=5, alpha=0.6)
#plt.bar(x[low:high], np.log10(no_peak), width=5, alpha=0.6)
#plt.axvline(75, lw=0.8, color='r')
#plt.axvline(1461, lw=0.8, color='g')
#plt.axvline(x[low], lw=0.8, color='r', linewidth=0.1)
#plt.axvline(x[high], lw=0.8, color='r', linewidth=0.1)
#plt.axvline(x[minE], lw=0.8, color='b', linewidth=0.1)
#plt.axvline(x[maxE], lw=0.8, color='b', linewidth=0.1)
#plt.plot(shwoopx, np.log10(shwoopy), linewidth=0.3)
#plt.axhline(0, color='k', linewidth=0.3)
#plt.savefig('K40__Pb144.svg', bbox_inches = 'tight')


#plt.figure(2).add_axes((0.05,0.05,1.2,0.68))
#plt.bar(x, np.log10(func.binsum(Agcounts64, binwidth)), width=5, alpha=0.6)
#plt.bar(x[low:high], np.log10(no_peak2), width=5, alpha=0.6)
#plt.axvline(75, lw=0.8, color='r')
#plt.axvline(1461, lw=0.8, color='g')
#plt.axvline(x[low], lw=0.8, color='r', linewidth=0.1)
#plt.axvline(x[high], lw=0.8, color='r', linewidth=0.1)
#plt.axvline(x[minE], lw=0.8, color='b', linewidth=0.1)
#plt.axvline(x[maxE], lw=0.8, color='b', linewidth=0.1)
#plt.plot(shwoopx2, np.log10(shwoopy2), linewidth=0.3)
#plt.axhline(0, color='k', linewidth=0.3)
#plt.savefig('K40__Ag64.svg', bbox_inches = 'tight')


#plt.figure(3).add_axes((0.05,0.05,1.2,0.68))
#plt.bar(x, np.log10(func.binsum(Alcounts64, binwidth)), width=5, alpha=0.6)
#plt.bar(x[low:high], np.log10(no_peak3), width=5, alpha=0.6)
#plt.axvline(75, lw=0.8, color='r')
#plt.axvline(1461, lw=0.8, color='g')
#plt.axvline(x[low], lw=0.8, color='r', linewidth=0.1)
#plt.axvline(x[high], lw=0.8, color='r', linewidth=0.1)
#plt.axvline(x[minE], lw=0.8, color='b', linewidth=0.1)
#plt.axvline(x[maxE], lw=0.8, color='b', linewidth=0.1)
#plt.plot(shwoopx3, np.log10(shwoopy3), linewidth=0.3)
#plt.axhline(0, color='k', linewidth=0.3)
#plt.savefig('K40__Al59.svg', bbox_inches = 'tight')


plt.figure(1).add_axes((0.05,0.05,1.2,0.68))
plt.bar(x[low:high], y[low:high] - no_peak, width=5, alpha=0.4)
plt.bar(x[low:high], y2[low:high] - no_peak2, width=5, alpha=0.4)
plt.bar(x[low:high], y3[low:high] - no_peak3, width=5, alpha=0.4)
plt.savefig('K40__peaks.svg', bbox_inches = 'tight')