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
binwidth = 20

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

y = func.binsum(Na22Counts, binwidth)
x = func.binmean(channels_raw, binwidth)

print(scipy.signal.find_peaks(y, width=(5,80)))

plt.figure(1).add_axes((0, 0, 1.2, 0.68))
plt.step(x / binwidth, np.log10(y + 1), linewidth=0.2)
plt.axvline(34/2, color='r', lw=0.5)
plt.axvline(97/2, color='r', lw=0.5)
plt.axvline(100/2, color='g', lw=0.5)
plt.axvline(177/2, color='r', lw=0.5)
plt.axvline(240/2, color='r', lw=0.5)
plt.axvline(333/2, color='r', lw=0.5)
plt.savefig('Res&Eff/check2.svg', bbox_inches = 'tight')