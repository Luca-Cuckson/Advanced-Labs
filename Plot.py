import numpy as np
import matplotlib.pyplot as plt
import scipy
import functions as func

binwidth = 10
ECh = 0.554661447159516
dpi = 300
LLD =  30

end = 2278.7 / ECh
binned_end = int(end / binwidth)

def spec_plot(x, y, yerror, n):
    SNR = y / yerror
    binned_SNR = func.binsum(SNR, binwidth)
    binned_SNR = binned_SNR[:binned_end]

    x, y, yerror = func.binmean(x, binwidth), func.binsum(y, binwidth), func.binsum(yerror, binwidth)
    x, y, yerror = x[:binned_end], y[:binned_end], yerror[:binned_end]

    plt.figure(n).add_axes((0,0,1.2,0.68))
    plt.bar(x, func.logging(y), width=5)
    plt.plot(x, func.logging(yerror), color='r', linewidth=0.5)

    plt.figure(n).add_axes((0,-0.3,1.2,0.25))
    plt.bar(x, func.logging(binned_SNR), width=5)
    plt.axhline(np.log10(5), xmin=0, color='r', lw=0.4)


channels, brem, brem_err = np.loadtxt('Report/Pb64_results')
channels, brem, brem_err = channels[30:], brem[30:], brem_err[30:]
print(len(channels), len(brem), len(brem_err))

spec_plot(channels, brem, brem_err, 1)
plt.savefig('Report/Pb64_spectrum', bbox_inches='tight', dpi=dpi)

