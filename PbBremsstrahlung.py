import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats

######################################################################################################################################################
# Extracting Data

nchannels = 16384
channelE = 0.533053
#thick = 0.04 # cm
rho = 11.34 # g / cm^3
LLD = 0
cutoff = 8192
binwidth = 20

channels = np.linspace(0, nchannels*channelE, 16384)
channels = channels[LLD:]

def load_maestro_spe(path):
    with open(path, "r") as f:
        lines = f.readlines()

    # Find the $DATA: section
    for i, line in enumerate(lines):
        if line.strip().startswith("$DATA"):
            data_start = i
            break

    # Next line: "first_channel last_channel"
    first, last = map(int, lines[data_start + 1].split())
    n = last - first + 1

    # Next n lines: counts
    data = np.array([int(x) for x in lines[data_start + 2 : data_start + 2 + n]])
    return data

background = load_maestro_spe(r"DataFiles\Backgrounds\Sr90 Gamma Raised 24 hours x100 26th Feb.Spe")
background = background[LLD:]
Pbcounts40 = load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.4mm Pb 21 hours x100 27th Feb.Spe")
Pbcounts40 = Pbcounts40[LLD:] * (84745 / 75043)
Pbcounts74 = load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.7mm Pb 25 hours x100 6th March.Spe")
Pbcounts74 = Pbcounts74[LLD:] * (84745 / 88875)
Pbcounts144 = load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 1.2mm Pb 69 hours x100 2nd March.Spe")
Pbcounts144 = Pbcounts144[LLD:] * (84745 / 244860)

######################################################################################################################################################
# Functions

def atten(counts, mu, x): # thickness x in cm
    return counts * np.exp(-mu * x)

def logging(counts):  # VERY IMPORTANT: FOR VISUALISING PLOTS ONLY, NOT SOUNDS FOR DATA ANALYSIS
    logged = np.empty(len(counts))
    for i in range(len(counts)):
        if -1 <= counts[i] <= 1:
            logged[i] = 0.0
        if 1 < counts[i]:
            logged[i] = np.log10(counts[i])
        if counts[i] < -1:
            logged[i] = -np.log10(-counts[i])
    return logged

def binsum(array, width):
    floor = len(array) // width
    multiple = floor * width
    binned = array[:multiple].reshape(-1, width).sum(axis=1)
    if multiple < len(array):
        binned = np.append(binned, np.sum(array[multiple:]))
    return binned

def binmean(array, width):
    floor = len(array) // width
    multiple = floor * width
    binned = array[:multiple].reshape(-1, width).mean(axis=1)
    if multiple < len(array):
        binned = np.append(binned, np.mean(array[multiple:]))
    return binned

def get_stripped_SNR(wannastrip, background, binwidth):
    stripped = binsum(wannastrip - background, binwidth)
    sigma = np.sqrt(binsum(wannastrip + background, binwidth)) # uses assumption of Poisson statistics where variance = value
    for i in range(len(sigma)):
        if sigma[i] == 0:
            sigma[i] = 1   
    SNR = stripped / sigma
    return SNR

######################################################################################################################################################
# Calculating attenuated background - so far assuming all background attenuated by target

file = r"DataFiles\AttenCoeffs\Pbattenuation_coeff.txt"
E, mu_rho = np.loadtxt(file, usecols=(0,1), unpack=True)
E = E*1e3

mu = mu_rho * rho

logE = np.log(E)
logmu = np.log(mu)

log_interp = scipy.interpolate.interp1d(logE, logmu, kind='linear', fill_value='extrapolate')
mu_vals = np.exp(log_interp(np.log(channels)))  # convert keV → MeV

print(E)
print(mu_vals)

# A good visual check
plt.figure(1).add_axes((0.05,0.05,1.2,0.68))
plt.plot(np.log10(channels), np.log10(mu_vals), alpha=0.6, linewidth=0.8)
plt.plot(np.log10(E), np.log10(mu), alpha=0.6, linewidth=0.8)
plt.savefig('PbGraphs/Brem_mus.svg', bbox_inches = 'tight')

print('mu check')


# Another check
plt.figure(4).add_axes((0.05,0.05,1.2,0.68))
plt.step(channels, logging(background), where='pre', alpha=0.6, color='r', linewidth=0.1)
plt.step(channels, logging(atten(background, mu_vals, 0.04)), where='pre', alpha=0.6, color='b', linewidth=0.1)
plt.step(channels, logging(Pbcounts40), where='pre', alpha=0.6, color='g', linewidth=0.1)
plt.savefig('PbGraphs/Brem_back_atten.svg', bbox_inches = 'tight')

print('background attenuation check')

######################################################################################################################################################
# Pb 0.40 mm 

# Binned
plt.figure(5).add_axes((0.05,0.05,1.2,0.68))
plt.bar(binmean(channels, binwidth), logging(binsum(Pbcounts40 - atten(background, mu_vals, 0.04), binwidth)), width=5)
plt.axvline(75, lw=0.8, color='r')
plt.axhline(0, color='k', linewidth=0.6)
plt.savefig('PbGraphs/Pb40_attenBin.svg', bbox_inches = 'tight')

print('Binned 0.4')


# Not binned 
#plt.figure(2).add_axes((0.05,0.05,1.2,0.68))
###plt.step(channels, logging(Brem2), where='pre', alpha=0.6, color='r', linewidth=0.1)
#plt.bar(channels, logging(Pbcounts40 - atten(background, mu_vals, 0.04)), width=1)
##plt.step(channels, logging(Brem), where='pre', alpha=0.5, color='b')
#plt.axvline(75, lw=0.8, color='r')
#plt.axhline(0, color='k', linewidth=0.6)
#plt.savefig('Pb40_atten.svg', bbox_inches = 'tight')

print('0.40')


######################################################################################################################################################
# Pb 0.74 mm 

# Binned
plt.figure(7).add_axes((0.05,0.05,1.2,0.68))
plt.bar(binmean(channels, binwidth), logging(binsum(Pbcounts74 - atten(background, mu_vals, 0.074), binwidth)), width=5)
plt.axvline(75, lw=0.8, color='r')
plt.axhline(0, color='k', linewidth=0.6)
plt.savefig('PbGraphs/Pb74_attenBin.svg', bbox_inches = 'tight')

print('Binned 0.74')


######################################################################################################################################################
# Pb 1.44 mm 

# Binned
plt.figure(6).add_axes((0.05,0.05,1.2,0.68))
plt.bar(binmean(channels, binwidth), logging(binsum(Pbcounts144 - atten(background, mu_vals, 0.144), binwidth)), width=5, alpha=1)
#plt.bar(binmean(channels, binwidth), logging(binsum(Pbcounts74 - atten(background, mu_vals, 0.074), binwidth)), width=5, alpha=0.6)
plt.axvline(75, lw=0.8, color='r')
plt.axvline(1461, lw=0.8, color='g')
plt.axhline(0, color='k', linewidth=0.6)
plt.savefig('PbGraphs/Pb144_attenBin.svg', bbox_inches = 'tight')

print('Binned 1.44')

x = binmean(channels, binwidth)
#y = binsum(Pbcounts144 - atten(background, mu_vals, 0.144), binwidth)
y = np.log10(get_stripped_SNR(Pbcounts144, atten(background, mu_vals, 0.144), binwidth))

plt.figure(8).add_axes((0.05,0.05,1.2,0.68))
plt.bar(x, y, width=5)
plt.axhline(np.log10(5), lw=0.4, color='r')
plt.axhline(0, lw=0.4, color='k')
plt.axhline(-np.log10(5), lw=0.4, color='r')
plt.savefig('PbGraphs/SNR_Pb144_attenBin.svg', bbox_inches = 'tight')

# Non binned
#plt.figure(3).add_axes((0.05,0.05,1.2,0.68))
#plt.bar(channels, logging(Pbcounts144 - atten(background, mu_vals, 0.144)), width=1)
#plt.axvline(75, lw=0.8, color='r')
#plt.axvline(1460, lw=0.8, color='g')
#plt.axhline(0, color='k', linewidth=0.6)
#plt.savefig('Pb144_atten.svg', bbox_inches = 'tight')

print('1.44')