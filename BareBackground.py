import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import functions as func


nchannels = 16384
channelE = 0.533053
rho = 11.35 # g / cm^3
LLD = 30
cutoff = 8192
binwidth = 10

channels = np.linspace(0, nchannels*channelE, 16384)
channels = channels[LLD:]

background = func.load_maestro_spe(r"DataFiles\Backgrounds\Sr90 Gamma Raised 24 hours x100 26th Feb.Spe")
background = background[LLD:] * (72000 / 84745)

background0 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY000.Spe")
background0 = background0[LLD:]
background1 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY001.Spe")
background1 = background1[LLD:]
background2 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY002.Spe")
background2 = background2[LLD:]
background3 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY003.Spe")
background3 = background3[LLD:]
background4 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY004.Spe")
background4 = background4[LLD:]
background5 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY005.Spe")
background5 = background5[LLD:]
background6 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY006.Spe")
background6 = background6[LLD:]
background7 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY007.Spe")
background7 = background7[LLD:]
background8 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY008.Spe")
background8 = background8[LLD:]
background9 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY009.Spe")
background9 = background9[LLD:]
background10 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY010.Spe")
background10 = background10[LLD:]
background11 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY011.Spe")
background11 = background11[LLD:]
background12 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY012.Spe")
background12 = background12[LLD:]
background13 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY013.Spe")
background13 = background13[LLD:]
background14 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY014.Spe")
background14 = background14[LLD:]
background15 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY015.Spe")
background15 = background15[LLD:]
background16 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY016.Spe")
background16 = background16[LLD:]
background17 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY017.Spe")
background17 = background17[LLD:]
background18 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY018.Spe")
background18 = background18[LLD:]
background19 = func.load_maestro_spe(r"DataFiles\Backgrounds\DECAY019.Spe")
background19 = background19[LLD:]

Pbcounts144 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 1.2mm Pb 69 hours x100 2nd March.Spe")
Pbcounts144 = Pbcounts144[LLD:] * (72000 / 244860)

BareBackground = func.load_maestro_spe(r"DataFiles\Backgrounds\Gamma Background 21 hours x100 13th March.Spe")
BareBackground = BareBackground[LLD:] * (72000 / 76344)


bare_background = background0 + background1 + background2 + background3 + background4 + background5 + background6 + background7 + background8 + background9 + background10 + background11 + background12 + background13 + background14 + background15 + background16 + background17 + background18 + background19

######################################################################################################################################################
# Calculating attenuation coefficients

file = r"DataFiles\AttenCoeffs\Pbattenuation_coeff.txt"
E, mu_rho = np.loadtxt(file, usecols=(0,1), unpack=True)
E = E*1e3

mu = mu_rho * rho

logE = np.log(E)
logmu = np.log(mu)

log_interp = scipy.interpolate.interp1d(logE, logmu, kind='linear', fill_value='extrapolate')
mu_vals = np.exp(log_interp(np.log(channels * (1461/1402))))  # convert keV → MeV


# Cheeky little x values setting

x = func.binmean(channels, binwidth) # calculating the binned energy values for all measurements
xbare = x * (1402/1432)
xBare = x * (1432 / 1502)


######################################################################################################################################################
# Compare backgrounds

a = func.binsum(bare_background, binwidth)
b = func.binsum(background, binwidth)
b2 = func.binsum(BareBackground, binwidth)

plt.figure(1).add_axes((0.05,0.05,1.2,0.68))
plt.bar(x, func.logging(a), width=5, alpha=0.6)
plt.bar(xBare, func.logging(b2), width=5, alpha=0.6)
plt.axvline(1500, lw=0.2)
plt.axvline(1432, lw=0.2)
plt.axhline(0, color='k', lw=0.6)
plt.savefig('BackgroundStuff/Comparison.svg', bbox_inches = 'tight')

######################################################################################################################################################
# Background interpolator 

interp_bare = scipy.interpolate.make_interp_spline(channels * (1402/1432), bare_background, k=3)

interp_Bare = scipy.interpolate.make_interp_spline(channels * (1402/1502), BareBackground, k=3)

aligned_bareback = interp_bare(channels)
aligned_Bareback = interp_bare(channels)

c = func.binsum(aligned_bareback, binwidth)
c2 = func.binsum(aligned_Bareback, binwidth)

plt.figure(2).add_axes((0,0,1.2,0.68))
#plt.bar(x, func.logging(b), width=5, alpha=0.6)
plt.bar(xbare, func.logging(c2), width=5, alpha=0.6)
plt.bar(xBare, func.logging(a), width=5, alpha=0.6)
plt.axvline(1405, lw=0.2)
plt.axhline(0, color='k', lw=0.6)
plt.savefig('BackgroundStuff/Aligned_Comparison.svg', bbox_inches = 'tight')

######################################################################################################################################################
# Source background

#d = b - c
d = func.binsum(background - aligned_bareback, binwidth)
d2 = func.binsum(background - aligned_Bareback, binwidth)

plt.figure(3).add_axes((0,0,1.2,0.68))
plt.bar(x, func.logging(d2), width=5)
plt.axhline(0, color='k', lw=0.6)
plt.savefig('BackgroundStuff/Aligned_SrBack.svg', bbox_inches = 'tight')

######################################################################################################################################################
# Pb144 Bremsstrahlung

Pbinterp = scipy.interpolate.make_interp_spline(channels * (1402/1398), Pbcounts144, k=3)
aligned_Pb = Pbinterp(channels)


Pb144Brem = aligned_Pb - aligned_Bareback - func.atten(background - aligned_Bareback, mu_vals, 0.144)

e = func.binsum(Pb144Brem, binwidth)

plt.figure(4).add_axes((0,0,1.2,0.68))
plt.bar(x, func.logging(e), width=5)
plt.axhline(0, color='k', lw=0.6)
plt.axhline(np.log10(25), color='r', lw=0.6)
plt.axhline(-np.log10(25), color='r', lw=0.6)
plt.axvline(1402, lw=0.2)
plt.axvline(1802, lw=0.2)
plt.savefig('BackgroundStuff/Pb144_Brem.svg', bbox_inches = 'tight')

plt.figure(5).add_axes((0,0,1.2,0.68))
plt.bar(x, func.logging(func.binsum(aligned_Pb, binwidth)), width=5)
plt.bar(x, func.logging(func.binsum(aligned_Bareback, binwidth)), width=5)
plt.bar(x, func.logging(func.binsum(func.atten(background - aligned_Bareback, mu_vals, 0.144), binwidth)), width=5)
plt.axhline(0, color='k', lw=0.6)
plt.axvline(1402, lw=0.2, color='r')
plt.axvline(75 * (1402/1461), lw=0.2, color='r')
plt.savefig('BackgroundStuff/Pb144_Brem_Check.svg', bbox_inches = 'tight')
