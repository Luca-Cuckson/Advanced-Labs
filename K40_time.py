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

######################################################################################################################################################
# Pb 1.44 mm

minE = 60 *2
low = 116  *2
high = 143 *2
maxE = 150 *2
print(func.binmean(channels, binwidth)[low])
print(func.binmean(channels, binwidth)[high])

shwoopx = np.append(func.binmean(channels, binwidth)[minE:low], func.binmean(channels, binwidth)[high:maxE])
shwoopy = np.append(func.binsum(Pbcounts144, binwidth)[minE:low], func.binsum(Pbcounts144, binwidth)[high:maxE])


interp = scipy.interpolate.make_interp_spline(shwoopx, np.log10(shwoopy), k=3)
no_peak = interp(func.binmean(channels, binwidth)[low:high])

######################################################################################################################################################
# Pb 2.82 mm

minE2 = 60 * 2
low2 = 116 * 2
high2 = 143 * 2
maxE2 = 150 * 2
print(func.binmean(channels, binwidth)[low2])
print(func.binmean(channels, binwidth)[high2])

shwoopx2 = np.append(func.binmean(channels, binwidth)[minE2:low2], func.binmean(channels, binwidth)[high2:maxE2])
shwoopy2 = np.append(func.binsum(Pbcounts282, binwidth)[minE2:low2], func.binsum(Pbcounts282, binwidth)[high2:maxE2])


interp2 = scipy.interpolate.make_interp_spline(shwoopx2, np.log10(shwoopy2), k=3)
no_peak2 = interp2(func.binmean(channels, binwidth)[low2:high2])

######################################################################################################################################################
# Pb 0.40 mm

minE3 = 60 * 2
low3 = 116 * 2
high3 = 143 * 2
maxE3 = 150 * 2
print(func.binmean(channels, binwidth)[low3])
print(func.binmean(channels, binwidth)[high3])

shwoopx3 = np.append(func.binmean(channels, binwidth)[minE3:low3], func.binmean(channels, binwidth)[high3:maxE3])
shwoopy3 = np.append(func.binsum(Pbcounts40, binwidth)[minE3:low3], func.binsum(Pbcounts40, binwidth)[high3:maxE3])


interp3 = scipy.interpolate.make_interp_spline(shwoopx3, np.log10(shwoopy3), k=1)
no_peak3 = interp3(func.binmean(channels, binwidth)[low3:high3])

######################################################################################################################################################
# Cu 0.59 mm

minE4 = 60 * 2
low4 = 116 * 2
high4 = 143 * 2
maxE4 = 150 * 2
print(func.binmean(channels, binwidth)[low4])
print(func.binmean(channels, binwidth)[high4])

shwoopx4 = np.append(func.binmean(channels, binwidth)[minE4:low4], func.binmean(channels, binwidth)[high4:maxE4])
shwoopy4 = np.append(func.binsum(Cucounts64, binwidth)[minE4:low4], func.binsum(Cucounts64, binwidth)[high4:maxE4])


interp4 = scipy.interpolate.make_interp_spline(shwoopx4, np.log10(shwoopy4), k=1)
no_peak4 = interp4(func.binmean(channels, binwidth)[low4:high4])

######################################################################################################################################################
# Ag 0.64 mm

minE5 = 60 * 2
low5 = 116 * 2
high5 = 143 * 2
maxE5 = 150 * 2
print(func.binmean(channels, binwidth)[low5])
print(func.binmean(channels, binwidth)[high5])

shwoopx5 = np.append(func.binmean(channels, binwidth)[minE5:low5], func.binmean(channels, binwidth)[high5:maxE5])
shwoopy5 = np.append(func.binsum(Agcounts64, binwidth)[minE5:low5], func.binsum(Agcounts64, binwidth)[high5:maxE5])


interp5 = scipy.interpolate.make_interp_spline(shwoopx5, np.log10(shwoopy5), k=1)
no_peak5 = interp5(func.binmean(channels, binwidth)[low5:high5])

######################################################################################################################################################
# Al 0.64 mm

minE6 = 60 * 2
low6 = 116 * 2
high6 = 143 * 2
maxE6 = 150 * 2
print(func.binmean(channels, binwidth)[low6])
print(func.binmean(channels, binwidth)[high6])

shwoopx6 = np.append(func.binmean(channels, binwidth)[minE6:low6], func.binmean(channels, binwidth)[high6:maxE6])
shwoopy6 = np.append(func.binsum(Alcounts64, binwidth)[minE6:low6], func.binsum(Alcounts64, binwidth)[high6:maxE6])


interp6 = scipy.interpolate.make_interp_spline(shwoopx6, np.log10(shwoopy6), k=1)
no_peak6 = interp6(func.binmean(channels, binwidth)[low6:high6])

######################################################################################################################################################

plt.figure(6).add_axes((0.05,0.05,1.2,0.68))
plt.bar(func.binmean(channels, binwidth), func.logging(func.binsum(Pbcounts144, binwidth)), width=5, alpha=0.6)
plt.bar(func.binmean(channels, binwidth)[low:high], no_peak, width=5, alpha=0.6)
#plt.bar(binmean(channels, binwidth), logging(binsum(Pbcounts74 - atten(background, mu_vals, 0.074), binwidth)), width=5, alpha=0.6)
plt.axvline(75, lw=0.8, color='r')
plt.axvline(1461, lw=0.8, color='g')
plt.axvline(func.binmean(channels, binwidth)[low], lw=0.8, color='r', linewidth=0.1)
plt.axvline(func.binmean(channels, binwidth)[high], lw=0.8, color='r', linewidth=0.1)
plt.axvline(func.binmean(channels, binwidth)[minE], lw=0.8, color='b', linewidth=0.1)
plt.axvline(func.binmean(channels, binwidth)[maxE], lw=0.8, color='b', linewidth=0.1)
plt.plot(shwoopx, func.logging(shwoopy), linewidth=0.3)
plt.axhline(0, color='k', linewidth=0.3)
plt.savefig('K40_peak1.svg', bbox_inches = 'tight')


plt.figure(2).add_axes((0.05,0.05,1.2,0.68))
plt.bar(func.binmean(channels, binwidth), func.logging(func.binsum(Pbcounts282, binwidth)), width=5, alpha=0.6)
plt.bar(func.binmean(channels, binwidth)[low2:high2], no_peak2, width=5, alpha=0.6)
#plt.bar(binmean(channels, binwidth), logging(binsum(Pbcounts74 - atten(background, mu_vals, 0.074), binwidth)), width=5, alpha=0.6)
plt.axvline(75, lw=0.8, color='r')
plt.axvline(1461, lw=0.8, color='g')
plt.axvline(func.binmean(channels, binwidth)[low2], lw=0.8, color='r', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[high2], lw=0.8, color='r', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[minE2], lw=0.8, color='b', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[maxE2], lw=0.8, color='b', linewidth=0.3)
plt.plot(shwoopx2, func.logging(shwoopy2), linewidth=0.3)
plt.axhline(0, color='k', linewidth=0.6)
plt.savefig('K40_peak3.svg', bbox_inches = 'tight')


plt.figure(3).add_axes((0.05,0.05,1.2,0.68))
plt.bar(func.binmean(channels, binwidth), func.logging(func.binsum(Pbcounts40, binwidth)), width=5, alpha=0.6)
plt.bar(func.binmean(channels, binwidth)[low3:high3], no_peak3, width=5, alpha=0.6)
#plt.bar(binmean(channels, binwidth), logging(binsum(Pbcounts74 - atten(background, mu_vals, 0.074), binwidth)), width=5, alpha=0.6)
plt.axvline(75, lw=0.8, color='r')
plt.axvline(1461, lw=0.8, color='g')
plt.axvline(func.binmean(channels, binwidth)[low3], lw=0.8, color='r', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[high3], lw=0.8, color='r', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[minE3], lw=0.8, color='b', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[maxE3], lw=0.8, color='b', linewidth=0.3)
plt.plot(shwoopx3, func.logging(shwoopy3), linewidth=0.3)
plt.axhline(0, color='k', linewidth=0.6)
plt.savefig('K40_peak4.svg', bbox_inches = 'tight')


plt.figure(4).add_axes((0.05,0.05,1.2,0.68))
plt.bar(func.binmean(channels, binwidth), func.logging(func.binsum(Cucounts64, binwidth)), width=5, alpha=0.6)
plt.bar(func.binmean(channels, binwidth)[low4:high4], no_peak4, width=5, alpha=0.6)
#plt.bar(binmean(channels, binwidth), logging(binsum(Pbcounts74 - atten(background, mu_vals, 0.074), binwidth)), width=5, alpha=0.6)
plt.axvline(75, lw=0.8, color='r')
plt.axvline(1461, lw=0.8, color='g')
plt.axvline(func.binmean(channels, binwidth)[low4], lw=0.8, color='r', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[high4], lw=0.8, color='r', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[minE4], lw=0.8, color='b', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[maxE4], lw=0.8, color='b', linewidth=0.3)
plt.plot(shwoopx4, func.logging(shwoopy4), linewidth=0.3)
plt.axhline(0, color='k', linewidth=0.6)
plt.savefig('K40_peak5.svg', bbox_inches = 'tight')


plt.figure(5).add_axes((0.05,0.05,1.2,0.68))
plt.bar(func.binmean(channels, binwidth), func.logging(func.binsum(Agcounts64, binwidth)), width=5, alpha=0.6)
plt.bar(func.binmean(channels, binwidth)[low5:high5], no_peak5, width=5, alpha=0.6)
#plt.bar(binmean(channels, binwidth), logging(binsum(Pbcounts74 - atten(background, mu_vals, 0.074), binwidth)), width=5, alpha=0.6)
plt.axvline(75, lw=0.8, color='r')
plt.axvline(1461, lw=0.8, color='g')
plt.axvline(func.binmean(channels, binwidth)[low5], lw=0.8, color='r', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[high5], lw=0.8, color='r', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[minE5], lw=0.8, color='b', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[maxE5], lw=0.8, color='b', linewidth=0.3)
plt.plot(shwoopx5, func.logging(shwoopy5), linewidth=0.3)
plt.axhline(0, color='k', linewidth=0.6)
plt.savefig('K40_peak6.svg', bbox_inches = 'tight')


plt.figure(7).add_axes((0.05,0.05,1.2,0.68))
plt.bar(func.binmean(channels, binwidth), func.logging(func.binsum(Alcounts64, binwidth)), width=5, alpha=0.6)
plt.bar(func.binmean(channels, binwidth)[low6:high6], no_peak6, width=5, alpha=0.6)
#plt.bar(binmean(channels, binwidth), logging(binsum(Pbcounts74 - atten(background, mu_vals, 0.074), binwidth)), width=5, alpha=0.6)
plt.axvline(75, lw=0.8, color='r')
plt.axvline(1461, lw=0.8, color='g')
plt.axvline(func.binmean(channels, binwidth)[low6], lw=0.8, color='r', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[high6], lw=0.8, color='r', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[minE6], lw=0.8, color='b', linewidth=0.3)
plt.axvline(func.binmean(channels, binwidth)[maxE6], lw=0.8, color='b', linewidth=0.3)
plt.plot(shwoopx6, func.logging(shwoopy6), linewidth=0.3)
plt.axhline(0, color='k', linewidth=0.6)
plt.savefig('K40_peak7.svg', bbox_inches = 'tight')


plt.figure(1).add_axes((0.05,0.05,1.2,0.68))
plt.bar(func.binmean(channels, binwidth)[low:high], (func.logging(func.binsum(Pbcounts144, binwidth)[low:high] - 10**no_peak)), width=5, alpha=0.3)
plt.bar(func.binmean(channels, binwidth)[low2:high2], (func.logging(func.binsum(Pbcounts282, binwidth)[low2:high2] - 10**no_peak2)), width=5, alpha=0.3)
plt.bar(func.binmean(channels, binwidth)[low3:high3], (func.logging(func.binsum(Pbcounts40, binwidth)[low3:high3] - 10**no_peak3)), width=5, alpha=0.3)
plt.bar(func.binmean(channels, binwidth)[low4:high4], (func.logging(func.binsum(Cucounts64, binwidth)[low4:high4] - 10**no_peak4)), width=5, alpha=0.3)
plt.bar(func.binmean(channels, binwidth)[low5:high5], (func.logging(func.binsum(Agcounts64, binwidth)[low5:high5] - 10**no_peak5)), width=5, alpha=0.3)
plt.bar(func.binmean(channels, binwidth)[low6:high6], (func.logging(func.binsum(Alcounts64, binwidth)[low6:high6] - 10**no_peak6)), width=5, alpha=0.3)
plt.xlim(0, 4000)
plt.savefig('K40_peak2.svg', bbox_inches = 'tight')