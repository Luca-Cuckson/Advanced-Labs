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

background = func.load_maestro_spe("Sr90 Gamma Raised 24 hours x100 26th Feb.Spe")
background = background[LLD:]

backbackground = func.load_maestro_spe("Sr90 Gamma Raised 24 hours x100 26th Feb.Spe")
backbackground = backbackground[LLD:] * (2/3)

Pbcounts40 = func.load_maestro_spe("Sr90 Brem 0.4mm Pb 21 hours x100 27th Feb.Spe")
Pbcounts40 = Pbcounts40[LLD:] * (84745 / 75043)
Pbcounts64 = func.load_maestro_spe("Sr90 Brem 0.64mm Pb 44 hours x100 12th March.Spe")
Pbcounts64 = Pbcounts64[LLD:] * (84745 / 159519)
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
# Attenuation coefficients

def get_mus(filename, rho, channels):
    E, mu_rho = np.loadtxt(filename, usecols=(0,1), unpack=True)
    E = E*1e3

    mu = mu_rho * rho

    logE = np.log(E)
    logmu = np.log(mu)

    log_interp = scipy.interpolate.interp1d(logE, logmu, kind='linear', fill_value='extrapolate')
    mu_vals = np.exp(log_interp(np.log(channels)))  # convert keV → MeV

    return mu_vals

mu_vals_Pb = get_mus('Pbattenuation_coeff.txt', rho, channels)

######################################################################################################################################################
# "Proper" background stuff

source = background - backbackground

# visual check
plt.figure(1).add_axes((0.05,0.05,1.2,0.68))
plt.bar(func.binmean(channels, binwidth), func.logging(func.binsum(source, binwidth)), width=5)
plt.axvline(75, lw=0.8, color='r')
plt.axhline(0, color='k', linewidth=0.5)
plt.savefig('saucey.svg', bbox_inches = 'tight')





def get_proper_background(pureback, Sr90back, mus, thick):
    sourceback = Sr90back - pureback
    attenuated = func.atten(sourceback, mus, thick)
    fullback = attenuated + pureback
    return fullback

def get_stripped(wannastrip, background):
    return wannastrip - background

def get_stripped_SNR(wannastrip, background):
    stripped = wannastrip - background
    sigma = np.sqrt(wannastrip + background) # uses assumption of Poisson statistics where variance = value
    SNR = stripped / sigma
    return SNR


x = func.binmean(channels, binwidth)


y_Pb_144 = func.binsum(get_stripped(Pbcounts144, get_proper_background(backbackground, background, mu_vals_Pb, 0.144)), binwidth)

plt.figure(2).add_axes((0.05,0.05,1.2,0.68))
plt.bar(x, func.logging(y_Pb_144), width=5)
plt.savefig('Brem_Pb_144.svg', bbox_inches='tight')