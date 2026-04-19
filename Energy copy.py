import numpy as np
import matplotlib.pyplot as plt
#import scipy.optimize
#import scipy.stats
import scipy
import functions as func


nchannels = 16384
channelE = 0.533053
rho = 11.34 # g / cm^3
LLD = 30
cutoff = 8192
binwidth = 10
dpi = 300

r = 29.6
d = 68.564

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

#print(scipy.signal.find_peaks(Na22Counts, width=(100,600)))

plt.figure(1).add_axes((0, 0, 1.2, 0.68))
plt.step(channels_raw, np.log10(Na22Counts + 1), linewidth=0.2)
plt.axvline(343, color='r', lw=0.5)
plt.axvline(975, color='r', lw=0.5)
plt.axvline(2403, color='r', lw=0.5)
plt.savefig('Res&Eff/check.svg', bbox_inches = 'tight')


# Well that clearly ain't working superbly 

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
    perr = np.sqrt(np.diag(cov))
    A_err, mu_err, sigma_err, c_err = perr

    fit = gauss(xdata, A, mu, sigma, c)

    plt.figure(n).add_axes((0, 0, 1.2, 0.68))
    plt.bar(xdata, ydata, width=1, alpha=0.3)
    plt.plot(xdata, fit)
    return A, mu, sigma, c, (mu_err+0.5), sigma_err

def efficienting(A0, halflife, T, Y, r, d, Ncounts, Ncounts_err, t): # Need A0 in Bq (counts / second), halflife and T (age of source) can just be same unit, time of reading t in seconds
    r_err = 0.05
    d_err = 0.224
    
    N_emit = func.decay(A0, halflife, T) * Y * t

    G = (r ** 2) / (4 * d ** 2) # factors of pi cancel

    N_on = G * N_emit
    N_on_rerr = abs(((r + r_err) ** 2) / (4 * d ** 2) * N_emit - N_on)
    N_on_derr = abs(((r) ** 2) / (4 * (d + d_err) ** 2) * N_emit - N_on)
    N_on_err = np.sqrt(N_on_rerr**2 + N_on_derr**2)

    Efficiency = (Ncounts[0] / N_on) * 100

    N_err = ((Ncounts[0] + Ncounts_err) / N_on) * 100 - ((Ncounts[0] / N_on) * 100)
    G_err = ((Ncounts[0] / N_on) * 100) - ((Ncounts[0] / (N_on + N_on_err)) * 100)
    Error = np.sqrt(N_err**2 + G_err**2)
    return Efficiency, Error

def gauss_area(func, A, mu, sigma, C, botlim, uplim):
    area = scipy.integrate.quad(func, botlim, uplim, args=(A, mu, sigma, C))
    return area

# chi-squared time!!!!!
def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2) # Note the `*model_params' here!

def line(x, m, c):
    return m*x + c

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def power_law(x, coeff, power, c):
    return coeff * x ** (power) + c

######################################################################################################################################################
# Chop of some peaks!
# Gonna do this in raw channels so then can also get a look at linearity of energy stuff
# Na22 First peak - 511 keV

minE, low, high, maxE = 795, 800, 1170, 1175

no_peak = no_peaking(channels_raw, Na22Counts, minE, low, high, maxE)

check_plot(Na22Counts, channels_raw, no_peak, minE, low, high, maxE, 2)
plt.savefig('Res&Eff/Na22_FirstPeak.svg', bbox_inches = 'tight')

A, mu, sigma, c, mu_err, sigma_err = Gaussing(gauss, channels_raw, Na22Counts, no_peak, low, high, 102)
plt.savefig('Res&Eff/Na22_Peak.svg', bbox_inches = 'tight')

Ech = 511 / mu
FWHM = 2.355 * sigma
E_FWHM = FWHM * Ech
Ech_err = Ech * (mu_err / mu)
FWHM_err = 2.355 * sigma_err
E_FWHM_err = E_FWHM * np.sqrt((Ech_err/Ech)**2 + (FWHM_err/FWHM)**2)

Na511E_FWHM = E_FWHM
Na511E_FWHM_err = E_FWHM_err

print('Na511:', [mu, Ech, FWHM, Ech_err, E_FWHM_err])

#############################################################

area = gauss_area(gauss, A, mu, sigma, c, low, high)

area_error = gauss_area(gauss, A, mu+mu_err, sigma+sigma_err, c, low, high)
area_err = area_error[0] - area[0]
print('area:', area, '+/-', area_err)

Na511_Efficiency = efficienting(420000, 2.6, 9.6, 1.807, r, d, area, area_err, 5589)

######################################################################################################################################################
# Na22 Second peak - 1275 keV

minE, low, high, maxE = 2000, 2130, 2700, 2701

no_peak = no_peaking(channels_raw, Na22Counts, minE, low, high, maxE)

check_plot(Na22Counts, channels_raw, no_peak, minE, low, high, maxE, 3)
plt.savefig('Res&Eff/Na22_SecondPeak.svg', bbox_inches = 'tight')

A, mu, sigma, c, mu_err, sigma_err = Gaussing(gauss, channels_raw, Na22Counts, no_peak, low, high, 103)
plt.savefig('Res&Eff/Na22_Peak2.svg', bbox_inches = 'tight')

Ech = 1275 / mu
FWHM = 2.355 * sigma
E_FWHM = FWHM * Ech
Ech_err = Ech * (mu_err / mu)
FWHM_err = 2.355 * sigma_err
E_FWHM_err = E_FWHM * np.sqrt((Ech_err/Ech)**2 + (FWHM_err/FWHM)**2)

Na1275E_FWHM = E_FWHM
Na1275E_FWHM_err = E_FWHM_err

print('Na1275:', [mu, Ech, FWHM, Ech_err, E_FWHM_err])

#############################################################

area = gauss_area(gauss, A, mu, sigma, c, low, high)

area_error = gauss_area(gauss, A, mu+mu_err, sigma+sigma_err, c, low, high)
area_err = area_error[0] - area[0]
print('area:', area, '+/-', area_err)

Na1275_Efficiency = efficienting(420000, 2.6, 9.6, 0.9994, r, d, area, area_err, 5589)

######################################################################################################################################################
# Co60 First peak - 1173 keV

minE, low, high, maxE = 1914, 1915, 2740, 2741
no_peak = no_peaking(channels_raw, Co60Counts, minE, low, high, maxE)

check_plot(Co60Counts, channels_raw, no_peak, minE, low, high, maxE, 4)
plt.savefig('Res&Eff/Co60_FirstPeak.png', bbox_inches = 'tight', dpi=dpi)

xdata = channels_raw[low:high]
ydata = Co60Counts[low:high] - no_peak

p0 = [9000, 2255, 120, 8000, 2560, 120, 0]
    
params, cov = scipy.optimize.curve_fit(double_gauss, xdata, ydata, p0=p0) # Fit for Gaussian parameters
A1, mu1, sigma1, A2, mu2, sigma2, C = params # Extract Gaussian parameters
A1_err, mu1_err, sigma1_err, A2_err, mu2_err, sigma2_err, C_err = np.sqrt(np.diag(cov))

fit1 = gauss(xdata, A1, mu1, sigma1, 0)
fit2 = gauss(xdata, A2, mu2, sigma2, 0)

plt.figure(104).add_axes((0, 0, 1.2, 0.68))
plt.bar(xdata, ydata, width=1, alpha=0.3)
plt.plot(xdata, fit1)
plt.plot(xdata, fit2)
plt.plot(xdata, fit1 + fit2)
plt.savefig('Res&Eff/Co60_Peak.png', bbox_inches = 'tight', dpi=dpi)

Co1173Ech = 1173 / mu1
Co1173FWHM = 2.355 * sigma1
Co1173E_FWHM = Co1173FWHM * Co1173Ech
Co1173Ech_err = Co1173Ech * (mu1_err / mu1)
Co1173FWHM_err = 2.355 * sigma1_err
Co1173E_FWHM_err = Co1173E_FWHM * np.sqrt((Co1173Ech_err/Co1173Ech)**2 + (Co1173FWHM_err/Co1173FWHM)**2)

Co1332Ech = 1332 / mu2
Co1332FWHM = 2.355 * sigma2
Co1332E_FWHM = Co1332FWHM * Co1332Ech
Co1332Ech_err = Co1332Ech * (mu2_err / mu2)
Co1332FWHM_err = 2.355 * sigma2_err
Co1332E_FWHM_err = Co1332E_FWHM * np.sqrt((Co1332Ech_err/Co1332Ech)**2 + (Co1332FWHM_err/Co1332FWHM)**2)

print('Co1173:', [Co1173Ech, Co1173FWHM, Co1173Ech_err, Co1173E_FWHM_err])
print('Co1332:', [Co1332Ech, Co1332FWHM, Co1332Ech_err, Co1332E_FWHM_err])

#############################################################

area1 = gauss_area(gauss, A1, mu1, sigma1, C, low, high)
area2 = gauss_area(gauss, A2, mu2, sigma2, C, low, high)

area_error1 = gauss_area(gauss, A1, mu1+mu1_err, sigma1+sigma1_err, C, low, high)
area_err1 = area_error1[0] - area1[0]
print('area:', area1, '+/-', area_err1)

area2_error = gauss_area(gauss, A2, mu2+mu2_err, sigma2+sigma2_err, C, low, high)
area2_err = area2_error[0] - area2[0]
print('area:', area2, '+/-', area2_err)

d2 = d + 27.93

Co1173_Efficiency = efficienting(3600000, 5.27, 9.64, 0.9985, r, d2, area1, area_err1, 1446)
Co1332_Efficiency = efficienting(3600000, 5.27, 9.64, 0.999826, r, d2, area2, area2_err, 1446)

######################################################################################################################################################
# Cs137 Peak - 661.7 keV

minE, low, high, maxE = 1065, 1070, 1540, 1545

no_peak = no_peaking(channels_raw, Cs137Counts, minE, low, high, maxE)

check_plot(Cs137Counts, channels_raw, no_peak, minE, low, high, maxE, 5)
plt.savefig('Res&Eff/Cs137_OnlyPeak.svg', bbox_inches = 'tight')

A, mu, sigma, c, mu_err, sigma_err = Gaussing(gauss, channels_raw, Cs137Counts, no_peak, low, high, 105)
plt.savefig('Res&Eff/Cs137_Peak.svg', bbox_inches = 'tight')

Ech = 661.7 / mu
FWHM = 2.355 * sigma
E_FWHM = FWHM * Ech
Ech_err = Ech * (mu_err / mu)
FWHM_err = 2.355 * sigma_err
E_FWHM_err = E_FWHM * np.sqrt((Ech_err/Ech)**2 + (FWHM_err/FWHM)**2)

CsE_FWHM = E_FWHM
CsE_FWHM_err = E_FWHM_err

print('Cs662:', [mu, Ech, FWHM, Ech_err, E_FWHM_err])

#############################################################

area = gauss_area(gauss, A, mu, sigma, c, low, high)

area_error = gauss_area(gauss, A, mu+mu_err, sigma+sigma_err, c, low, high)
area_err = area_error[0] - area[0]
print('area:', area, '+/-', area_err)

Cs_Efficiency = efficienting(370000, 30, 47.4, 0.8499, r, d, area, area_err, 4651)

######################################################################################################################################################
# Resolutions

plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['mathtext.fontset'] = 'cm'

print(type(Na511E_FWHM_err), Na511E_FWHM_err)
print(type(CsE_FWHM_err), CsE_FWHM_err)
print(type(Co1173E_FWHM_err), Co1173E_FWHM_err)
print(type(Na1275E_FWHM_err), Na1275E_FWHM_err)
print(type(Co1332E_FWHM_err), Co1332E_FWHM_err)

Energies = np.array([511, 661.7, 1173, 1275, 1332])
E_FWHMs = np.array([Na511E_FWHM, CsE_FWHM, Co1173E_FWHM, Na1275E_FWHM, Co1332E_FWHM])#, 214.56079073121322])
E_FWHM_errs = np.array([Na511E_FWHM_err, CsE_FWHM_err, Co1173E_FWHM_err, Na1275E_FWHM_err, Co1332E_FWHM_err])

Resolutions = E_FWHMs / Energies * 100
Resolution_errs = E_FWHM_errs / Energies * 100

print('Resolutions:', Resolutions)
print('Resolution errors:', Resolution_errs)

p0 = [-0.01, 20]
params, cov = scipy.optimize.curve_fit(line, Energies, Resolutions, p0)
print(params)
m, c = params
x_grid = np.linspace(500, 1350, 1000)
y_line = line(x_grid, m, c)
line_chi = chi_squared((m, c), line, Energies, Resolutions, Resolution_errs) / 3
print('line chi ssquared =', line_chi)

plt.figure(200)
#plt.plot(Energies, Resolutions)
#plt.plot((Energies[0], Energies[-1]), (Resolutions[0], Resolutions[-1]), linestyle='dashed')
plt.errorbar(Energies, Resolutions, yerr=Resolution_errs, linestyle='none', marker='v', color='blueviolet', ecolor='mediumorchid', markersize=8)
plt.plot(x_grid, y_line, linestyle='dashed', color='mediumorchid', linewidth=2)
plt.ylabel(r"% Resolution")
plt.xlabel(r"Energy (keV)")
plt.savefig('Res&Eff/Resolution', bbox_inches='tight', dpi=dpi)

######################################################################################################################################################
# Efficiencies

Efficiencies = [Na511_Efficiency[0], Cs_Efficiency[0], Co1173_Efficiency[0], Na1275_Efficiency[0], Co1332_Efficiency[0]]
Eff_err = np.array([Na511_Efficiency[1], Cs_Efficiency[1], Co1173_Efficiency[1], Na1275_Efficiency[1], Co1332_Efficiency[1]])

p0_1 = [-0.01, 20]
params, cov = scipy.optimize.curve_fit(line, Energies, Efficiencies, p0_1)
m, c1 = params
print(params)

p0_2 = [0.01, -0.01, 0]
params, cov = scipy.optimize.curve_fit(quadratic, Energies, Efficiencies, p0_2)
a, b, c2 = params
print(params)

#p0_3 = [8000, -1.12, 0]
#params, cov = scipy.optimize.curve_fit(power_law, Energies, Efficiencies, p0_3)
#coeff, power, c3 = params

x_grid = np.linspace(500, 1350, 1000)
y_line = line(x_grid, m, c1)
y_quad = quadratic(x_grid, a, b, c2)
#y_power = power_law(x_grid, coeff, power, c3)

line_chi = chi_squared((m, c1), line, Energies, Efficiencies, Eff_err) / 3
quad_chi = chi_squared((a, b, c2), quadratic, Energies, Efficiencies, Eff_err) / 2

print('line chi ssquared =', line_chi)
print('quadratic chi ssquared =', quad_chi)

print('Efficiencies:', Efficiencies)
print('Efficiencies error:', Eff_err)


plt.figure(300)
plt.errorbar(Energies, Efficiencies, yerr=(Eff_err), linestyle='none', marker='v', color='blueviolet', ecolor='mediumorchid', markersize=8)
#plt.plot(x_grid, y_line, linestyle='dashed')
plt.plot(x_grid, y_quad, linestyle='dashed', color='mediumorchid', linewidth=2)
#plt.plot(x_grid, y_power, linestyle='dashed')
plt.ylabel(r"% Efficiency")
plt.xlabel(r"Energy (keV)")
plt.savefig('Res&Eff/Efficiency', bbox_inches='tight', dpi=300)