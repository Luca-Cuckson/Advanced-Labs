import numpy as np
import matplotlib.pyplot as plt
import scipy
import functions as func
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

binwidth = 10
ECh = 0.554661447159516
dpi = 300
LLD =  30
mult = 1.2

end = 2278.7 / ECh
binned_end = int(end / binwidth)

end2 = 600
end3 = end2 * binwidth

raw_channels = np.linspace(1, 16384, 16384)

plt.rcParams["font.size"] = 22
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['mathtext.fontset'] = 'cm'

###############################################################################################################
def gauss(x, A, mu, sigma, C):
    return A*np.exp(-(x-mu)**2/(2*sigma**2)) + C

def double_gauss(x, A1, mu1, sigma1, A2, mu2, sigma2, C):
    return A1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + C

def no_peaking(x, y, minE, low, high, maxE):  
    shwoopx = np.append(x[minE:low], x[high:maxE])
    shwoopy = np.append(y[minE:low], y[high:maxE])

    coeffs = np.polyfit(shwoopx, np.log10(shwoopy), deg=1)
    no_peak = 10 ** np.polyval(coeffs, x[low:high])

    #interp = scipy.interpolate.make_smoothing_spline(shwoopx, shwoopy)
    #no_peak = interp(x[low:high])

    return no_peak

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

def K40_plot1(counts, x, no_peak, minE, low, high, maxE, n):
    x, counts = x[:end2], counts[:end2]

    plt.rcParams["font.size"] = 19

    plt.figure(n).add_axes((0, 0, 1, 0.8))
    plt.step(x, counts, linewidth=1.2, color='mediumorchid')
    plt.step(x[low:high], no_peak, linewidth=0.2, alpha=0.6, color='blueviolet')
    plt.fill_between(x[low:high], no_peak, step='mid', alpha=0.3, color='blueviolet')
    #plt.axvline(x[low], color='r', lw=0.2)
    #plt.axvline(x[high], color='r', lw=0.2)
    plt.axhline(0, color='k', lw=0.3)
    plt.yscale('log')
    plt.xlim([0, max(x)])
    plt.ylim([0, max(y)*mult])
    plt.ylabel(r"Counts")
    plt.xlabel(r"Channels")

    xdata = x[low:high]
    ydata = y[low:high] - no_peak

    p0 = [np.max(ydata), xdata[np.argmax(ydata)],  20, 0]
    
    params, cov = scipy.optimize.curve_fit(gauss, xdata, ydata, p0=p0) # Fit for Gaussian parameters
    A, mu, sigma, c = params # Extract Gaussian parameters

    fit = gauss(xdata, A, mu, sigma, c)

    plt.rcParams["font.size"] = 16

    plt.figure(n).add_axes((0.57, 0.37, 0.42, 0.42))
    plt.bar(xdata, ydata, width=10, alpha=0.3, color='mediumorchid')
    plt.plot(xdata, fit, color='blueviolet')
    #plt.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    plt.ylim([0, max(ydata) * 1.1])
    plt.xticks([2600,2750,2900])

def K40_plot2(counts, x, no_peak, minE, low, high, maxE, n):
    x, counts = x[:end3], counts[:end3]

    plt.rcParams["font.size"] = 19

    plt.figure(n).add_axes((0, 0, 1, 0.8))
    plt.step(x, counts, linewidth=1.2, color='mediumorchid')
    plt.step(x[low:high], no_peak, linewidth=0.2, alpha=0.6, color='blueviolet')
    plt.fill_between(x[low:high], no_peak, step='mid', alpha=0.3, color='blueviolet')
    #plt.axvline(x[low], color='r', lw=0.2)
    #plt.axvline(x[high], color='r', lw=0.2)
    plt.axhline(0, color='k', lw=0.3)
    plt.yscale('log')
    plt.xlim([0, max(x)])
    plt.ylim([0, max(counts)*1.1])
    plt.ylabel(r"Counts")
    plt.xlabel(r"Channel")

    xdata = x[low:high]
    ydata = counts[low:high] - no_peak

    p0 = [9000, 2255, 120, 8000, 2560, 120, 0]
    
    params, cov = scipy.optimize.curve_fit(double_gauss, xdata, ydata, p0=p0) # Fit for Gaussian parameters
    A1, mu1, sigma1, A2, mu2, sigma2, C = params # Extract Gaussian parameters
    A1_err, mu1_err, sigma1_err, A2_err, mu2_err, sigma2_err, C_err = np.sqrt(np.diag(cov))

    fit1 = gauss(xdata, A1, mu1, sigma1, 0)
    fit2 = gauss(xdata, A2, mu2, sigma2, 0)

    plt.rcParams["font.size"] = 16

    plt.figure(n).add_axes((0.63, 0.46, 0.36, 0.33))
    plt.bar(xdata, ydata, width=10, alpha=0.08, color='mediumorchid')
    plt.plot(xdata, fit1, color='darkviolet')
    plt.plot(xdata, fit2, color='magenta')
    plt.plot(xdata, fit1 + fit2, color='indigo')
    plt.ylim([0, max(ydata)*1.05])
    #plt.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    plt.xticks([2000,2300,2600])
    plt.yticks([3000,6000,9000])

def K40_plot(counts, x, no_peak, minE, low, high, maxE, n):
    x, counts = x[:end2], counts[:end2]

    fig, ax = plt.subplots(); ax.set_box_aspect(0.5)

    ax.step(x, counts, linewidth=1.2, alpha=0.6)
    ax.step(x[low:high], no_peak, linewidth=0.2, alpha=0.6)
    ax.axvline(x[low], color='r', lw=0.2)
    ax.axvline(x[high], color='r', lw=0.2)
    ax.axhline(0, color='k', lw=0.3)
    ax.yscale('log')
    ax.xlim([0, max(x)])
    ax.ylim([0, max(y)*1.1])

    #xdata = x[low:high]
    #ydata = y[low:high] - no_peak

    #p0 = [np.max(ydata), xdata[np.argmax(ydata)],  20, 0]
    
    #params, cov = scipy.optimize.curve_fit(gauss, xdata, ydata, p0=p0) # Fit for Gaussian parameters
    #A, mu, sigma, c = params # Extract Gaussian parameters

    #fit = gauss(xdata, A, mu, sigma, c)

    #plt.figure(n).add_axes((0, 0, 1.2, 0.68))
    #plt.bar(xdata, ydata, width=1, alpha=0.3)
    #plt.plot(xdata, fit)

def no_negative(counts):
    for i in range(len(counts)):
        if counts[i] < 1:
            counts[i] = 1
    return counts

###############################################################################################################

def spec_plot(x, y, yerror, fin, n):
    SNR = y / yerror
    binned_SNR = func.binsum(SNR, binwidth)
    binned_SNR = binned_SNR[:binned_end]

    x, y, yerror = func.binmean(x, binwidth), func.binsum(y, binwidth), func.binsum(yerror, binwidth)
    x, y, yerror = x[:binned_end], y[:binned_end], yerror[:binned_end]

    peak_begin, peak_end = np.abs(x - 72).argmin(), np.abs(x - 85).argmin()

    peakx, peaky = x[peak_begin:peak_end], y[peak_begin:peak_end]

    finbin = np.abs(x - fin).argmin()


    plt.figure(n).add_axes((0,0,1.2,0.8))
    plt.step(x[:finbin+1], y[:finbin+1], where='mid', linewidth=1.2, color='mediumorchid')
    plt.fill_between(x[:finbin+1], y[:finbin+1], step='mid', alpha=0.6, color='mediumorchid')
    plt.step(x[finbin:], y[finbin:], where='mid', linewidth=1.2, color='mediumorchid', alpha = 0.6)
    plt.fill_between(x[finbin:], y[finbin:], step='mid', alpha=0.3, color='mediumorchid')
    ##plt.fill_between(x, y+yerror, y-yerror, step='mid', alpha=0.4, color='gray')
    #plt.fill_between(peakx, peaky, step='mid', alpha=0.3, color='r')
    plt.axvline(fin, color='k', lw=1, linestyle='dotted')
    #plt.axvline(85, color='g', lw=0.4)
    plt.xlim([0, max(x)])
    plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    plt.yscale('log')
    plt.ylabel(r"Counts")

    plt.figure(n).add_axes((0,-0.3,1.2,0.28))
    plt.step(x[:finbin+1], binned_SNR[:finbin+1], where='mid', linewidth=1.2, color='mediumorchid')
    plt.fill_between(x[:finbin+1], binned_SNR[:finbin+1], step='mid', alpha=0.6, color='mediumorchid')
    plt.step(x[finbin:], binned_SNR[finbin:], where='mid', linewidth=1.2, color='mediumorchid', alpha = 0.6)
    plt.fill_between(x[finbin:], binned_SNR[finbin:], step='mid', alpha=0.3, color='mediumorchid')
    plt.axhline(5, xmin=0, color='k', lw=1, linestyle='dashed')
    plt.axvline(fin, color='k', lw=1, linestyle='dotted')
    plt.xlim([0,max(x)])
    plt.yscale('log')
    plt.ylabel(r"SNR")
    plt.xlabel(r"Energy (keV)")
    plt.xticks(np.linspace(0, 2100, 8))
    plt.yticks([1,100, 10000])

###############################################################################################################
# Pb 0.64 mm target raw spectrum

Pbcounts64 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.64mm Pb 44 hours x100 12th March.Spe")
Pbcounts64 = Pbcounts64 * (84745 / 159519)

low, high = int(2540 / binwidth), int(2960 / binwidth)
data, assigned_no, target_thick = Pbcounts64, 20, 0.064
#####
x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)


no_peak = no_peaking(x, y, minE, low, high, maxE)

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))

K40_plot1(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Report/Pb64_raw.png', bbox_inches = 'tight', dpi=dpi)
plt.close()



###############################################################################################################
# Pb 0.64 mm target Bremsstrahlung spectrum

plt.rcParams["font.size"] = 22

channels, brem, brem_err = np.loadtxt('Report/Pb64_results')
channels, brem, brem_err = channels[30:], brem[30:], brem_err[30:]
i = 1146
print(len(channels), len(brem), len(brem_err))

spec_plot(channels, brem, brem_err, i, 1)
plt.savefig('Report/Pb64_spectrum', bbox_inches='tight', dpi=dpi)

###############################################################################################################
# Cu 0.61 mm target Bremsstrahlung spectrum

plt.rcParams["font.size"] = 22

brem, brem_err = np.loadtxt('Report/Cu61_results')
brem, brem_err = brem[30:], brem_err[30:]
i = 519
print(len(channels), len(brem), len(brem_err))

spec_plot(channels, brem, brem_err, i, 2)
plt.savefig('Report/Cu61_spectrum', bbox_inches='tight', dpi=dpi)

###############################################################################################################
# Co60
Co60Counts = func.load_maestro_spe(r"DataFiles\Readings\Co60 30 mins x100 19th Feb.Spe")
Co60Counts = Co60Counts[LLD:]

minE, low, high, maxE = 1914, 1915, 2740, 2741
no_peak = no_peaking(raw_channels, Co60Counts, minE, low, high, maxE)

K40_plot2(Co60Counts, raw_channels, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Report/Co60_spectrum.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

###############################################################################################################
# Bare background spectrum

BB, BB_err = np.loadtxt('Report/BB_results')
BB, BB_err = BB[30:], BB_err[30:]

spec_plot(channels, BB, BB_err, i, 3)
plt.savefig('Report/BB_spectrum', bbox_inches='tight', dpi=dpi)
###############################################################

BareBackground = func.load_maestro_spe(r"DataFiles\Backgrounds\Gamma Background 21 hours x100 13th March.Spe")
BareBackground = BareBackground * (84745 / 76344)

low, high = int(2550 / binwidth), int(3080 / binwidth)
data, assigned_no = BareBackground, 1001

x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

K40_plot1(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Report/Bare_back.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

###############################################################################################################
# Source background spectrum

plt.rcParams["font.size"] = 18

SB, SB_err = np.loadtxt('Report/SB_results')
x, SB = x[LLD:], SB[LLD:]
x, SB = func.binmean(channels, binwidth), func.binsum(SB, binwidth)
SB, x = SB[:binned_end], x[:binned_end]

#spec_plot(channels, SB, SB_err, i, 4)
#plt.savefig('Report/SB_spectrum2', bbox_inches='tight', dpi=dpi)

plt.figure(48).add_axes((0,0,1.0,0.8))
plt.step(x, no_negative(SB), where='mid', linewidth=1.2, color='mediumorchid')
plt.ylim([1, max(SB)*mult])
plt.xlim([0, max(x)])
plt.yscale('log')
plt.ylabel(r"Counts")
plt.xlabel(r"Energy (keV)")
plt.savefig('Report/SB_spectrum', bbox_inches='tight', dpi=dpi)

###############################################################################################################
# Thicknesses

plt.rcParams["font.size"] = 20

Thicknesses, Nets, Nets_err = np.loadtxt('Report/Thickness_nets')

plt.figure(4001)
plt.bar(Thicknesses, Nets, width=0.1, color='blueviolet', alpha=0.6)
plt.bar(Thicknesses, height= 2*Nets_err, bottom= Nets-Nets_err, width=0.1, alpha=0.6, color='mediumorchid')
plt.ylabel(r"Net Counts")
plt.xlabel(r"Target Thickness (mm)")
plt.yscale('log')
plt.savefig('Report/ThickComparison2.png', bbox_inches='tight', dpi=dpi)
plt.close()

Ref = Nets[1]
Nets, Nets_err = Nets / Ref, Nets_err / Ref

plt.figure(4000).add_axes((0,0,1,0.6))
plt.yticks([0.6,0.7,0.8,0.9,1.0,1.1])
plt.errorbar(Thicknesses, Nets, Nets_err, linestyle='none', marker='v', color='blueviolet', ecolor='mediumorchid', markersize=8)
plt.ylabel(r"Relative Yield")
plt.xlabel(r"Target Thickness (mm)")
plt.savefig('Report/ThickComparison.png', bbox_inches='tight', dpi=dpi)
plt.close()

###############################################################################################################
# Zs

Zs, Nets, Nets_err = np.loadtxt('Report/Z_nets')
Nets, Nets_err = Nets / Ref, Nets_err / Ref

plt.figure(4008).add_axes((0,0,1,0.6))
#plt.yticks([0.6,0.7,0.8,0.9,1.0,1.1])
plt.errorbar(Zs, Nets, Nets_err, linestyle='none', marker='v', color='blueviolet', ecolor='mediumorchid', markersize=8)
plt.ylabel(r"Relative Yield")
plt.xlabel(r"Target Atomic Number Z")
plt.savefig('Report/Z_Comparison.png', bbox_inches='tight', dpi=dpi)
plt.close()

###############################################################################################################
# Actual Pb 0.74 mm

actualx, Actual, Actual_err = np.loadtxt('Report/Pb74_actual')
actualx, Actual = func.binmean(actualx, binwidth), func.binsum(Actual, binwidth)
Actual = Actual[:binned_end]

plt.figure(53).add_axes((0,0,1.0,0.8))
plt.step(actualx, no_negative(Actual), where='mid', linewidth=1.2, color='darkorchid')
plt.ylim([0, max(Actual)*1.05])
plt.xlim([500, 1100])
plt.axvline(actualx.min(), 0, Actual[0]/(max(Actual*1.05)), linewidth=1.2, color='darkorchid')
plt.axvline(actualx[-1], 0, Actual[-1]/(max(Actual*1.05)), linewidth=1.2, color='darkorchid')
plt.ylabel(r"Counts")
plt.xlabel(r"Energy (keV)")
plt.savefig('Report/Actual_spectrum', bbox_inches='tight', dpi=dpi)
