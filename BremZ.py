import numpy as np
import matplotlib.pyplot as plt
import scipy
import functions as func

nchannels = 16384
#channelE = 0.533053
rho = 11.34 # g / cm^3
LLD = 30
cutoff = 8192
binwidth = 10
dpi = 600
K40Energy = 1461
interp_err_factor = 1
SNR_lim = 5

raw_channels = np.linspace(1, 16384, 16384)

BothBackground = func.load_maestro_spe(r"DataFiles\Backgrounds\Sr90 Gamma Raised 24 hours x100 26th Feb.Spe")
BareBackground = func.load_maestro_spe(r"DataFiles\Backgrounds\Gamma Background 21 hours x100 13th March.Spe")
BareBackground = BareBackground * (84745 / 76344)

Pbcounts64 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.64mm Pb 44 hours x100 12th March.Spe")
Pbcounts64 = Pbcounts64 * (84745 / 159519)

Cucounts64 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.6mm Cu 25 hours x100 3rd March.Spe")
Cucounts64 = Cucounts64 * (84745 / 89451)

Alcounts64 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.64mm Al 18 hours x100 4th March.Spe")
Alcounts64 = Alcounts64 * (84745 / 64248)

Agcounts64 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.6mm Ag 27 hours x100 5th March.Spe")
Agcounts64 = Agcounts64 * (84745 / 94912)

######################################################################################################################################################
# Functions

def gauss(x, A, mu, sigma, C):
    return A*np.exp(-(x-mu)**2/(2*sigma**2)) + C

def double_gauss(x, A1, mu1, sigma1, A2, mu2, sigma2, C):
    return A1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2*sigma2**2)) + C

# The interpolation function that finds the continued curve under the peak (assuming a straight line in log-scale)
def no_peaking(x, y, minE, low, high, maxE):  
    shwoopx = np.append(x[minE:low], x[high:maxE])
    shwoopy = np.append(y[minE:low], y[high:maxE])

    coeffs = np.polyfit(shwoopx, np.log10(shwoopy), deg=1)
    no_peak = 10 ** np.polyval(coeffs, x[low:high])

    #interp = scipy.interpolate.make_smoothing_spline(shwoopx, shwoopy)
    #no_peak = interp(x[low:high])

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

def efficienting(A0, halflife, T, Y, r, d, Ncounts, t): # Need A0 in Bq (counts / second), halflife and T (age of source) can just be same unit, time of reading t in seconds
    r_err = 0.05
    d_err = 0.224
    
    N_emit = func.decay(A0, halflife, T) * Y * t

    G = (r ** 2) / (4 * d ** 2) # factors of pi cancel

    N_on = G * N_emit
    N_on_rerr = abs(((r + r_err) ** 2) / (4 * d ** 2) * N_emit - N_on)
    N_on_derr = abs(((r) ** 2) / (4 * (d + d_err) ** 2) * N_emit - N_on)
    N_on_err = np.sqrt(N_on_rerr**2 + N_on_derr**2)

    Efficiency = (Ncounts[0] / N_on) * 100

    N_err = ((Ncounts[0] + np.sqrt(Ncounts[0])) / N_on) * 100 - ((Ncounts[0] / N_on) * 100)
    G_err = ((Ncounts[0] / N_on) * 100) - ((Ncounts[0] / (N_on + N_on_err)) * 100)
    Error = np.sqrt(N_err**2 + G_err**2)
    return Efficiency, Error

def gauss_area(func, A, mu, sigma, C, botlim, uplim):
    area = scipy.integrate.quad(func, botlim, uplim, args=(A, mu, sigma, C))
    return area

def spec_plot(x, y, n):
    plt.figure(n).add_axes((0,0,1.2,0.68))
    plt.bar(x, func.logging(y), width=5)
    plt.axhline(0, color='k', lw=0.3)
    plt.axhline(np.log10(25), color='r', lw=0.6)
    plt.axhline(-np.log10(25), color='r', lw=0.6)

def find_end(x, y, limit, n):
    yish = y[20:]
    for i in range(len(yish)):
        if yish[i]<limit and yish[i+1]<limit and yish[i+2]<limit and yish[i+3]<limit and yish[i+4]<limit:
            print([yish[i], yish[i+1], yish[i+2], yish[i+3], yish[i+4]])
            end = i + 20
            break
    plt.figure((n + 300)).add_axes((0.05,0.05,1.2,0.68))
    plt.bar(x, func.logging(y), width=5)
    plt.axvline(x[end], color='r', lw=0.3)
    plt.axhline(np.log10(SNR_lim), color='r', lw=0.3)
    plt.axhline(-np.log10(SNR_lim), color='r', lw=0.3)
    return end

######################################################################################################################################################
# Backgrounds
######################################################################################################################################################
# Combined background

low, high = int(2436 / binwidth), int(2830 / binwidth)
x, data, assigned_no = func.binmean(raw_channels, binwidth), func.binsum(BothBackground, binwidth), 1000
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, data, minE, low, high, maxE)

check_plot(data, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Z_Time!/Combined_background_check1.png', bbox_inches = 'tight', dpi=dpi)

A, mu, sigma, c = Gaussing(gauss, x, data, no_peak, low, high, (assigned_no + 100))
plt.savefig('Z_Time!/Combined_background_peak1.png', bbox_inches = 'tight', dpi=dpi)

Both_mu = mu

E_Ch = K40Energy / mu

print(E_Ch)


channels = raw_channels * E_Ch
channels_binned = func.binmean(channels, binwidth)

######################################################################################################################################################
# Plain background

low, high = int(2550 / binwidth), int(3080 / binwidth)
data, assigned_no = BareBackground, 1001

x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

check_plot(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Z_Time!/Plain_background_check.png', bbox_inches = 'tight', dpi=dpi)

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Z_Time!/Plain_background_peak.png', bbox_inches = 'tight', dpi=dpi)

interp = scipy.interpolate.make_interp_spline(channels * (Both_mu / mu), data, k=3)

bare_aligned = interp(channels) * (mu / Both_mu)
bare_err = np.sqrt(abs(bare_aligned)) * interp_err_factor

bare_aligned_binned = func.binsum(bare_aligned, binwidth)

######################################################################################################################################################
# Source background

SourceBackground = BothBackground - bare_aligned
Source_err = np.sqrt(BothBackground + bare_err ** 2)

SourceBackBin = func.binsum(SourceBackground, binwidth)

plt.figure(2000)
plt.bar(channels_binned, func.logging(SourceBackBin+1), width=5)
plt.savefig('Z_Time!/Source_Background', bbox_inches = 'tight', dpi=dpi)

######################################################################################################################################################
# Calculating attenuation coefficients
# Pb
file = r"DataFiles\AttenCoeffs\Pbattenuation_coeff.txt"
E, mu_rho = np.loadtxt(file, usecols=(0,1), unpack=True)
E = E*1e3

mu = mu_rho * rho

logE = np.log(E)
logmu = np.log(mu)

log_interp = scipy.interpolate.interp1d(logE, logmu, kind='linear', fill_value='extrapolate')
Pbmu_vals = np.exp(log_interp(np.log(channels)))  # convert keV → MeV

plt.figure(3001).add_axes((0.05,0.05,1.2,0.68))
plt.plot(np.log10(channels), np.log10(Pbmu_vals), alpha=0.6, linewidth=0.8)
plt.plot(np.log10(E), np.log10(mu), alpha=0.6, linewidth=0.8)
plt.savefig('Z_Time!/Pb_mus.svg', bbox_inches = 'tight')

###### Al
rho = 2.7

file = r"DataFiles\AttenCoeffs\Alattenuation_coeff.txt"
E, mu_rho = np.loadtxt(file, usecols=(0,1), unpack=True)
E = E*1e3

mu = mu_rho * rho

logE = np.log(E)
logmu = np.log(mu)

log_interp = scipy.interpolate.interp1d(logE, logmu, kind='linear', fill_value='extrapolate')
Almu_vals = np.exp(log_interp(np.log(channels)))  # convert keV → MeV

###### Ag
rho = 10.49

file = r"DataFiles\AttenCoeffs\Agattenuation_coeff.txt"
E, mu_rho = np.loadtxt(file, usecols=(0,1), unpack=True)
E = E*1e3

mu = mu_rho * rho

logE = np.log(E)
logmu = np.log(mu)

log_interp = scipy.interpolate.interp1d(logE, logmu, kind='linear', fill_value='extrapolate')
Agmu_vals = np.exp(log_interp(np.log(channels)))  # convert keV → MeV

##### Cu
rho = 8.96

file = r"DataFiles\AttenCoeffs\Cuattenuation_coeff.txt"
E, mu_rho = np.loadtxt(file, usecols=(0,1), unpack=True)
E = E*1e3

mu = mu_rho * rho

logE = np.log(E)
logmu = np.log(mu)

log_interp = scipy.interpolate.interp1d(logE, logmu, kind='linear', fill_value='extrapolate')
Cumu_vals = np.exp(log_interp(np.log(channels)))  # convert keV → MeV

plt.figure(3004).add_axes((0.05,0.05,1.2,0.68))
plt.plot(np.log10(channels), np.log10(Cumu_vals), alpha=0.6, linewidth=0.8)
plt.plot(np.log10(E), np.log10(mu), alpha=0.6, linewidth=0.8)
plt.savefig('Z_Time!/Cu_mus.svg', bbox_inches = 'tight')

######################################################################################################################################################
# Lead Time!!
######################################################################################################################################################
# Al 0.64 mm
#####
low, high = int(2435 / binwidth), int(2820 / binwidth)
data, assigned_no, target_thick, mu_vals = Alcounts64, 1, 0.064, Almu_vals
#####
x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

check_plot(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Z_Time!/Al64_check.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Z_Time!/Al64_peak.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

interp = scipy.interpolate.make_interp_spline(channels * (Both_mu / mu), data, k=3)
aligned = interp(channels) * (mu / Both_mu)
aligned_err = np.sqrt(abs(aligned)) * interp_err_factor
aligned_binned = func.binsum(aligned, binwidth)

Bremsstrahlung = aligned - func.atten(SourceBackground, mu_vals, target_thick) - bare_aligned
Brem_err = np.sqrt(aligned_err ** 2 + Source_err ** 2 + bare_err ** 2)
binned_Brem = func.binsum(Bremsstrahlung, binwidth)
spec_plot(channels_binned, binned_Brem, (assigned_no + 200))
plt.axvline(75, color='g', lw = 0.3)
plt.savefig('Z_Time!/Al64_spectrum.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

SNR = Bremsstrahlung / Brem_err
SNR_binned = func.binsum(SNR, binwidth)
i = find_end(channels_binned, SNR_binned, SNR_lim, assigned_no)
plt.savefig('Z_Time!/Al64_SNR.png', bbox_inches='tight', dpi=dpi)
plt.close()

print(channels_binned[i])

NetCounts = np.sum(binned_Brem[int(30/binwidth) : i])
NetAl = NetCounts
print('Al 0.64 mm counts:', NetCounts)

######################################################################################################################################################
# Cu 0.61 mm
#####
low, high = int(2440 / binwidth), int(2840 / binwidth)
data, assigned_no, target_thick, mu_vals = Cucounts64, 2, 0.061, Cumu_vals
#####
x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

check_plot(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Z_Time!/Cu61_check.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Z_Time!/Cu61_peak.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

interp = scipy.interpolate.make_interp_spline(channels * (Both_mu / mu), data, k=3)
aligned = interp(channels) * (mu / Both_mu)
aligned_err = np.sqrt(abs(aligned)) * interp_err_factor
aligned_binned = func.binsum(aligned, binwidth)

print(np.sum(data))
print(np.sum(aligned))

Bremsstrahlung = aligned - func.atten(SourceBackground, mu_vals, target_thick) - bare_aligned
Brem_err = np.sqrt(aligned_err ** 2 + Source_err ** 2 + bare_err ** 2)
binned_Brem = func.binsum(Bremsstrahlung, binwidth)
spec_plot(channels_binned, binned_Brem, (assigned_no + 200))
plt.axvline(75, color='g', lw = 0.3)
plt.savefig('Z_Time!/Cu61_spectrum.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

SNR = Bremsstrahlung / Brem_err
SNR_binned = func.binsum(SNR, binwidth)
i = find_end(channels_binned, SNR_binned, SNR_lim, assigned_no)
plt.savefig('Z_Time!/Cu61_SNR.png', bbox_inches='tight', dpi=dpi)
plt.close()

print(channels_binned[i])

NetCounts = np.sum(binned_Brem[int(30/binwidth) : i])
NetCu = NetCounts
print('Cu 0.61 mm counts:', NetCounts)

thing = func.binsum(func.atten(SourceBackground, mu_vals, target_thick) + bare_aligned, binwidth)

plt.figure(5000).add_axes((0,0,1.2,0.68))
plt.bar(channels_binned, func.logging(aligned_binned), width=5, alpha=0.6)
plt.bar(channels_binned, func.logging(thing), width=5, alpha=0.6)
#plt.bar(channels_binned, func.logging(bare_aligned_binned), width=5, alpha=0.6)
#plt.bar(channels_binned, func.logging(func.binsum(func.atten(SourceBackground, mu_vals, target_thick), binwidth)), width=5, alpha=0.6)
plt.axvline(1461, lw=0.2, color='g')
plt.axhline(0, color='k', lw=0.6)
plt.savefig('Z_Time!/Cu_Check_Alignment.png', bbox_inches = 'tight', dpi=dpi)

######################################################################################################################################################
# Ag 0.64 mm
#####
low, high = int(2460 / binwidth), int(2840 / binwidth)
data, assigned_no, target_thick, mu_vals = Agcounts64, 3, 0.064, Agmu_vals
#####
x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

check_plot(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Z_Time!/Ag64_check.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Z_Time!/Ag64_peak.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

interp = scipy.interpolate.make_interp_spline(channels * (Both_mu / mu), data, k=3)
aligned = interp(channels) * (mu / Both_mu)
aligned_err = np.sqrt(abs(aligned)) * interp_err_factor
aligned_binned = func.binsum(aligned, binwidth)

Bremsstrahlung = aligned - func.atten(SourceBackground, mu_vals, target_thick) - bare_aligned
Brem_err = np.sqrt(aligned_err ** 2 + Source_err ** 2 + bare_err ** 2)
binned_Brem = func.binsum(Bremsstrahlung, binwidth)
spec_plot(channels_binned, binned_Brem, (assigned_no + 200))
plt.axvline(75, color='g', lw = 0.3)
plt.savefig('Z_Time!/Ag64_spectrum.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

SNR = Bremsstrahlung / Brem_err
SNR_binned = func.binsum(SNR, binwidth)
i = find_end(channels_binned, SNR_binned, SNR_lim, assigned_no)
plt.savefig('Z_Time!/Ag64_SNR.png', bbox_inches='tight', dpi=dpi)
plt.close()

print(channels_binned[i])

NetCounts = np.sum(binned_Brem[int(30/binwidth) : i])
NetAg = NetCounts
print('Ag 0.64 mm counts:', NetCounts)

######################################################################################################################################################
# Pb 0.64 mm
#####
low, high = int(2540 / binwidth), int(2960 / binwidth)
data, assigned_no, target_thick, mu_vals = Pbcounts64, 4, 0.064, Pbmu_vals
#####
x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

check_plot(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Z_Time!/Pb64_check.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Z_Time!/Pb64_peak.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

interp = scipy.interpolate.make_interp_spline(channels * (Both_mu / mu), data, k=3)
aligned = interp(channels) * (mu / Both_mu)
aligned_err = np.sqrt(abs(aligned)) * interp_err_factor
aligned_binned = func.binsum(aligned, binwidth)

Bremsstrahlung = aligned - func.atten(SourceBackground, mu_vals, target_thick) - bare_aligned
Brem_err = np.sqrt(aligned_err ** 2 + Source_err ** 2 + bare_err ** 2)
binned_Brem = func.binsum(Bremsstrahlung, binwidth)
spec_plot(channels_binned, binned_Brem, (assigned_no + 200))
plt.axvline(75, color='g', lw = 0.3)
plt.savefig('Z_Time!/Pb64_spectrum.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

SNR = Bremsstrahlung / Brem_err
SNR_binned = func.binsum(SNR, binwidth)
i = find_end(channels_binned, SNR_binned, SNR_lim, assigned_no)
plt.savefig('Z_Time!/Pb64_SNR.png', bbox_inches='tight', dpi=dpi)
plt.close()

print(channels_binned[i])

NetCounts = np.sum(binned_Brem[int(30/binwidth) : i])
NetPb = NetCounts
print('Pb 0.64 mm counts:', NetCounts)

######################################################################################################################################################
# Comparison
######################################################################################################################################################

Zs = [13, 29, 79, 82]
Nets = [NetAl, NetCu, NetAg, NetPb]

plt.figure(4000)
plt.plot(Zs, Nets)
plt.savefig('Z_Time!/Comparison.png', bbox_inches='tight', dpi=dpi)
plt.close()