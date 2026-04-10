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
interp_err_factor = 1.5
SNR_lim = 5

raw_channels = np.linspace(1, 16384, 16384)

BothBackground = func.load_maestro_spe(r"DataFiles\Backgrounds\Sr90 Gamma Raised 24 hours x100 26th Feb.Spe")
BareBackground = func.load_maestro_spe(r"DataFiles\Backgrounds\Gamma Background 21 hours x100 13th March.Spe")
BareBackground = BareBackground * (84745 / 76344)

Pbcounts40 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.4mm Pb 21 hours x100 27th Feb.Spe")
Pbcounts40 = Pbcounts40 * (84745 / 75043)
Pbcounts64 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.64mm Pb 44 hours x100 12th March.Spe")
Pbcounts64 = Pbcounts64 * (84745 / 159519)
Pbcounts74 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 0.7mm Pb 25 hours x100 6th March.Spe")
Pbcounts74 = Pbcounts74 * (84745 / 88875)
Pbcounts100 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 1.00mm Pb 64 hours x100 16th March.Spe")
Pbcounts100 = Pbcounts100 * (84745 / 231052)
PbCounts141 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 1.2mm Pb 69 hours x100 2nd March.Spe")
PbCounts141 = PbCounts141 * (84745 / 244860)
Pbcounts282 = func.load_maestro_spe(r"DataFiles\Readings\Sr90 Brem 2.82mm Pb 92 hours x100 10th March.Spe")
Pbcounts282 = Pbcounts282 * (84745 / 332296)

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

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

######################################################################################################################################################
# Backgrounds
######################################################################################################################################################
# Combined background

low, high = int(2436 / binwidth), int(2830 / binwidth)
x, data, assigned_no = func.binmean(raw_channels, binwidth), func.binsum(BothBackground, binwidth), 1000
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, data, minE, low, high, maxE)

check_plot(data, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Thicknesses/Combined_background_check1.png', bbox_inches = 'tight', dpi=dpi)

A, mu, sigma, c = Gaussing(gauss, x, data, no_peak, low, high, (assigned_no + 100))
plt.savefig('Thicknesses/Combined_background_peak1.png', bbox_inches = 'tight', dpi=dpi)

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
plt.savefig('Thicknesses/Plain_background_check.png', bbox_inches = 'tight', dpi=dpi)

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Thicknesses/Plain_background_peak.png', bbox_inches = 'tight', dpi=dpi)

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
plt.savefig('Thicknesses/Source_Background', bbox_inches = 'tight', dpi=dpi)

######################################################################################################################################################
# Calculating attenuation coefficients

file = r"DataFiles\AttenCoeffs\Pbattenuation_coeff.txt"
E, mu_rho = np.loadtxt(file, usecols=(0,1), unpack=True)
E = E*1e3

mu = mu_rho * rho

logE = np.log(E)
logmu = np.log(mu)

log_interp = scipy.interpolate.interp1d(logE, logmu, kind='linear', fill_value='extrapolate')
mu_vals = np.exp(log_interp(np.log(channels)))  # convert keV → MeV

######################################################################################################################################################
# Lead Time!!
######################################################################################################################################################
# Pb 0.40 mm
#####
low, high = int(2435 / binwidth), int(2810 / binwidth)
data, assigned_no, target_thick = Pbcounts40, 1, 0.04
#####
print(np.sum(data))

x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

check_plot(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Thicknesses/Pb40_check.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Thicknesses/Pb40_peak.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

interp = scipy.interpolate.make_interp_spline(channels * (Both_mu / mu), data, k=3)
aligned = interp(channels) * (mu / Both_mu)
aligned_err = np.sqrt(abs(aligned)) * interp_err_factor
aligned_binned = func.binsum(aligned, binwidth)

Bremsstrahlung = aligned - func.atten(SourceBackground, mu_vals, target_thick) - bare_aligned
Brem_err = np.sqrt(aligned_err ** 2 + Source_err ** 2 + bare_err ** 2)
binned_Brem = func.binsum(Bremsstrahlung, binwidth)
spec_plot(channels_binned, binned_Brem, (assigned_no + 200))
plt.savefig('Thicknesses/Pb40_spectrum.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

SNR = Bremsstrahlung / Brem_err
SNR_binned = func.binsum(SNR, binwidth)
i = find_end(channels_binned, SNR_binned, SNR_lim, assigned_no)
plt.savefig('Thicknesses/Pb40_SNR.png', bbox_inches='tight', dpi=dpi)
plt.close()

cut40 = channels_binned[i]
print(channels_binned[i])

NetCounts = np.sum(binned_Brem[int(30/binwidth) : i])
Net40 = NetCounts
Net_Err = np.sqrt(np.sum(Brem_err**2))
Net40_err = Net_Err

print(np.sum(Bremsstrahlung))
print('Pb 0.40 mm counts:', NetCounts, '+/-', Net_Err)

######################################################################################################################################################
# Pb 0.64 mm
#####
low, high = int(2540 / binwidth), int(2960 / binwidth)
data, assigned_no, target_thick = Pbcounts64, 2, 0.064
#####
print(np.sum(data))

x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

check_plot(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Thicknesses/Pb64_check.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Thicknesses/Pb64_peak.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

interp = scipy.interpolate.make_interp_spline(channels * (Both_mu / mu), data, k=3)
aligned = interp(channels) * (mu / Both_mu)
aligned_err = np.sqrt(abs(aligned)) * interp_err_factor
aligned_binned = func.binsum(aligned, binwidth)

Bremsstrahlung = aligned - func.atten(SourceBackground, mu_vals, target_thick) - bare_aligned
Brem_err = np.sqrt(aligned_err ** 2 + Source_err ** 2 + bare_err ** 2)
binned_Brem = func.binsum(Bremsstrahlung, binwidth)
spec_plot(channels_binned, binned_Brem, (assigned_no + 200))
plt.savefig('Thicknesses/Pb64_spectrum.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

SNR = Bremsstrahlung / Brem_err
SNR_binned = func.binsum(SNR, binwidth)
i = find_end(channels_binned, SNR_binned, SNR_lim, assigned_no)
plt.savefig('Thicknesses/Pb64_SNR.png', bbox_inches='tight', dpi=dpi)
plt.close()

np.savetxt('Pb64_results', [channels, Bremsstrahlung, Brem_err])

cut64 = channels_binned[i]
print(channels_binned[i])

NetCounts = np.sum(binned_Brem[int(30/binwidth) : i])
Net64 = NetCounts
Net_Err = np.sqrt(np.sum(Brem_err**2))
Net64_err = Net_Err
print(np.sum(Bremsstrahlung))
print('Pb 0.64 mm counts:', NetCounts, '+/-', Net_Err)

######################################################################################################################################################
# Pb 0.74 mm
#####
low, high = int(2430 / binwidth), int(2820 / binwidth)
data, assigned_no, target_thick = Pbcounts74, 3, 0.074
#####
print(np.sum(data))

x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

check_plot(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Thicknesses/Pb74_check.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Thicknesses/Pb74_peak.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

interp = scipy.interpolate.make_interp_spline(channels * (Both_mu / mu), data, k=3)
aligned = interp(channels) * (mu / Both_mu)
aligned_err = np.sqrt(abs(aligned)) * interp_err_factor
aligned_binned = func.binsum(aligned, binwidth)

Bremsstrahlung = aligned - func.atten(SourceBackground, mu_vals, target_thick) - bare_aligned
Brem_err = np.sqrt(aligned_err ** 2 + Source_err ** 2 + bare_err ** 2)
binned_Brem = func.binsum(Bremsstrahlung, binwidth)
spec_plot(channels_binned, binned_Brem, (assigned_no + 200))
plt.savefig('Thicknesses/Pb74_spectrum.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

SNR = Bremsstrahlung / Brem_err
SNR_binned = func.binsum(SNR, binwidth)
i = find_end(channels_binned, SNR_binned, SNR_lim, assigned_no)
plt.savefig('Thicknesses/Pb74_SNR.png', bbox_inches='tight', dpi=dpi)
plt.close()

cut74 = channels_binned[i]
print(channels_binned[i])

NetCounts = np.sum(binned_Brem[int(30/binwidth) : i])
Net74 = NetCounts
Net_Err = np.sqrt(np.sum(Brem_err**2))
Net74_err = Net_Err

print(np.sum(Bremsstrahlung))
print('Pb 0.74 mm counts:', NetCounts, '+/-', Net_Err)

########################################################

a, b, c = 2.41629589e-06, -1.69925760e-02, 2.31120387e+01

print(channels[i*binwidth])
begin = int(511 / E_Ch)
print(channels[begin])

efficiency_x = channels[begin:i*binwidth]
count_y = Bremsstrahlung[begin:i*binwidth]
efficiency = quadratic(efficiency_x, a, b, c)

print(efficiency)

actual_brem = count_y * 100 / efficiency
actual_binned = func.binsum(actual_brem, binwidth)
actual_x = func.binmean(efficiency_x, binwidth)

plt.figure(5000)
plt.bar(actual_x, actual_binned, width=5)
plt.savefig('Thicknesses/Pb74_actual.png', bbox_inches='tight', dpi=dpi)

######################################################################################################################################################
# Pb 1.00 mm
#####
low, high = int(2580 / binwidth), int(3010 / binwidth)
data, assigned_no, target_thick = Pbcounts100, 4, 0.1
#####
print(np.sum(data))

x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

check_plot(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Thicknesses/Pb100_check.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Thicknesses/Pb100_peak.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

interp = scipy.interpolate.make_interp_spline(channels * (Both_mu / mu), data, k=3)
aligned = interp(channels) * (mu / Both_mu)
aligned_err = np.sqrt(abs(aligned)) * interp_err_factor
aligned_binned = func.binsum(aligned, binwidth)

Bremsstrahlung = aligned - func.atten(SourceBackground, mu_vals, target_thick) - bare_aligned
Brem_err = np.sqrt(aligned_err ** 2 + Source_err ** 2 + bare_err ** 2)
binned_Brem = func.binsum(Bremsstrahlung, binwidth)
spec_plot(channels_binned, binned_Brem, (assigned_no + 200))
plt.savefig('Thicknesses/Pb100_spectrum.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

SNR = Bremsstrahlung / Brem_err
SNR_binned = func.binsum(SNR, binwidth)
i = find_end(channels_binned, SNR_binned, SNR_lim, assigned_no)
plt.savefig('Thicknesses/Pb100_SNR.png', bbox_inches='tight', dpi=dpi)
plt.close()

cut100 = channels_binned[i]
print(channels_binned[i])

NetCounts = np.sum(binned_Brem[int(30/binwidth) : i])
Net100 = NetCounts
Net_Err = np.sqrt(np.sum(Brem_err**2))
Net100_err = Net_Err

print(np.sum(Bremsstrahlung))
print('Pb 1.00 mm counts:', NetCounts, '+/-', Net_Err)

######################################################################################################################################################
# Pb 1.41 mm
#####
low, high = int(2430 / binwidth), int(2810 / binwidth)
data, assigned_no, target_thick = PbCounts141, 5, 0.141
#####
print(np.sum(data))

x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

check_plot(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Thicknesses/Pb141_check.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Thicknesses/Pb141_peak.png', bbox_inches = 'tight', dpi=dpi)
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
plt.savefig('Thicknesses/Pb141_spectrum.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

SNR = Bremsstrahlung / Brem_err
SNR_binned = func.binsum(SNR, binwidth)
i = find_end(channels_binned, SNR_binned, SNR_lim, assigned_no)
plt.savefig('Thicknesses/Pb141_SNR.png', bbox_inches='tight', dpi=dpi)
plt.close()

cut141 = channels_binned[i]
print(channels_binned[i])

NetCounts = np.sum(binned_Brem[int(30/binwidth) : i])
Net141 = NetCounts
Net_Err = np.sqrt(np.sum(Brem_err**2))
Net141_err = Net_Err

print(np.sum(Bremsstrahlung))
print('Pb 1.41 mm counts:', NetCounts, '+/-', Net_Err)

######################################################################################################################################################
# Pb 2.82 mm
#####
low, high = int(2430 / binwidth), int(2810 / binwidth)
data, assigned_no, target_thick = Pbcounts282, 6, 0.282
#####
print(np.sum(data))

x, y = func.binmean(raw_channels, binwidth), func.binsum(data, binwidth)
minE, maxE = (low - 1), (high + 1)

no_peak = no_peaking(x, y, minE, low, high, maxE)

check_plot(y, x, no_peak, minE, low, high, maxE, assigned_no)
plt.savefig('Thicknesses/Pb282_check.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

A, mu, sigma, c = Gaussing(gauss, x, y, no_peak, low, high, (assigned_no + 100))
plt.savefig('Thicknesses/Pb282_peak.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

interp = scipy.interpolate.make_interp_spline(channels * (Both_mu / mu), data, k=3)
aligned = interp(channels) * (mu / Both_mu)
aligned_err = np.sqrt(abs(aligned)) * interp_err_factor
aligned_binned = func.binsum(aligned, binwidth)

Bremsstrahlung = aligned - func.atten(SourceBackground, mu_vals, target_thick) - bare_aligned
Brem_err = np.sqrt(aligned_err ** 2 + Source_err ** 2 + bare_err ** 2)
binned_Brem = func.binsum(Bremsstrahlung, binwidth)
spec_plot(channels_binned, binned_Brem, (assigned_no + 200))
plt.savefig('Thicknesses/Pb282_spectrum.png', bbox_inches = 'tight', dpi=dpi)
plt.close()

SNR = Bremsstrahlung / Brem_err
SNR_binned = func.binsum(SNR, binwidth)
i = find_end(channels_binned, SNR_binned, SNR_lim, assigned_no)
plt.savefig('Thicknesses/Pb282_SNR.png', bbox_inches='tight', dpi=dpi)
plt.close()

cut282 = channels_binned[i]
print(channels_binned[i])

NetCounts = np.sum(binned_Brem[int(30/binwidth) : i])
Net282 = NetCounts
Net_Err = np.sqrt(np.sum(Brem_err**2))
Net282_err = Net_Err

print(np.sum(Bremsstrahlung))
print('Pb 2.82 mm counts:', NetCounts, '+/-', Net_Err)

######################################################################################################################################################
# Comparison
######################################################################################################################################################

Thicknesses = [0.4, 0.64, 0.74, 1, 1.41, 2.82]
Nets = np.array([Net40, Net64, Net74, Net100, Net141, Net282])
Nets_err = np.array([Net40_err, Net64_err, Net74_err, Net100_err, Net141_err, Net282_err])

plt.figure(4000)
#plt.plot(Thicknesses, Nets)
plt.errorbar(Thicknesses, Nets, Nets_err)
plt.savefig('Thicknesses/Comparison.png', bbox_inches='tight', dpi=dpi)
plt.close()

plt.figure(4001)
plt.errorbar(Thicknesses, Nets, np.sqrt(Nets))
plt.savefig('Thicknesses/Comparison2.png', bbox_inches='tight', dpi=dpi)
plt.close()

Times = [75043, 159519, 88875, 231052, 244860, 332296]
Cuts = [cut40, cut64, cut74, cut100, cut141, cut282]

plt.figure(4002)
plt.plot(Times, Cuts)
plt.savefig('Thicknesses/Comparison3.png', bbox_inches='tight', dpi=dpi)
plt.close()