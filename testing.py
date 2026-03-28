import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


nchannels = 16384
channelE = 0.533053
thick = 0.04 # cm
rho = 11.34 # g / cm^3

channels = np.linspace(0, nchannels*channelE, 16384)
channels = channels[2:]

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

background = load_maestro_spe("Sr90 Gamma Raised 24 hours x100 26th Feb.Spe")
background = background[2:] * 75043.72 / 84745.14
PbBrem = load_maestro_spe("Sr90 Brem 0.4mm Pb 21 hours x100 27th Feb.Spe")
PbBrem = PbBrem[2:]

def plot(counts, colour):
    a = colour
    return plt.step(channels, np.log10(counts), where='pre', alpha=0.6, color='a')



#plt.figure(1).add_axes((0.05,0.05,1.2,0.68))
#plt.plot(channels, np.log10(background))
#plt.plot(channels, np.log10(PbBrem))
#plt.step(channels, np.log10(background), where='pre', alpha=0.6)
#plt.step(channels, np.log10(PbBrem), where='pre', alpha=0.6, color='r')
#plt.fill_between(channels, np.log10(background), step='mid', alpha=0.3)
#plt.savefig('plot.svg', bbox_inches = 'tight')
#plt.show()


Brem = PbBrem - background

for i in range(len(Brem)):
    if Brem[i] < 1:
        Brem[i] = 1

plt.figure(2).add_axes((0.05,0.05,1.2,0.68))
plt.step(channels, np.log10(Brem), where='pre', alpha=0.6, linewidth=0.2)
#plt.fill_between(channels, np.log10(Brem), step='mid', alpha=0.3)
#plt.plot(channels, np.log10(Brem))
plt.savefig('Brem.svg', bbox_inches = 'tight')


def atten(counts, mu, x):
    return counts * np.exp(-mu * x)

file = 'attenuation_coeff.txt'
E, mu_rho = np.loadtxt(file, usecols=(0,1), unpack=True)

mu = mu_rho * rho

logE = np.log(E)
logmu = np.log(mu)

log_interp = scipy.interpolate.interp1d(logE, logmu, kind='linear', fill_value='extrapolate')
mu_vals = np.exp(log_interp(np.log(channels/1e3)))  # convert keV → MeV


print(mu_vals)


plt.figure(3).add_axes((0.05,0.05,1.2,0.68))
plt.plot(np.log10(channels), np.log10(mu_vals), alpha=0.6, linewidth=0.8)
plt.plot(np.log10(E*1000), np.log10(mu), alpha=0.6, linewidth=0.8)
plt.savefig('mus.svg', bbox_inches = 'tight')
plt.show()



attenuated = atten(background, mu_vals, thick)


#plt.figure(4).add_axes((0.05,0.05,1.2,0.68))
#plt.step(channels, np.log10(background), where='pre', alpha=0.6)
#plt.step(channels, np.log10(attenuated+1), where='pre', alpha=0.6, color='r')
#plt.savefig('attenuated.svg', bbox_inches = 'tight')



print(min(attenuated))



Brem2 = PbBrem - attenuated
Brem2_inv = attenuated - PbBrem

def logging(counts):
    logged = np.empty(len(counts))
    for i in range(len(counts)):
        if -1 <= counts[i] <= 1:
            logged[i] = 0.0
        if 1 < counts[i]:
            logged[i] = np.log10(counts[i])
        if counts[i] < -1:
            logged[i] = -np.log10(-counts[i])
    return logged

plt.figure(5).add_axes((0.05,0.05,1.2,0.68))
#plt.step(channels, np.log10(background), where='pre', alpha=0.6)
###plt.step(channels, logging(Brem2), where='pre', alpha=0.6, color='r', linewidth=0.1)
plt.bar(channels, logging(Brem2), width=1)
#plt.step(channels, np.log10(Brem2), where='pre', alpha=0.6, color='b', linewidth=0.01)
#plt.step(channels, -np.log10(Brem2_inv), where='pre', alpha=0.5, color='r')
#plt.step(channels, logging(Brem), where='pre', alpha=0.5, color='b')
plt.axvline(75, lw=0.8, color='r')
plt.axhline(0, color='k')
plt.savefig('attenuated_Brem.svg', bbox_inches = 'tight')

print(np.min(Brem2))

Brem3 = np.empty(len(Brem2))
for i in range(len(Brem2)):
    if Brem2[i] < 1:
        Brem3[i] = 1
    else:
        Brem3[i]=Brem2[i]

Brem4  = attenuated - PbBrem
Brem5 = np.empty(len(Brem4))
for i in range(len(Brem4)):
    if Brem4[i] < 1:
        Brem5[i] = 1
    else:
        Brem5[i]=Brem4[i]

plt.figure(6).add_axes((0.05,0.05,1.2,0.68))
#plt.step(channels, np.log10(background), where='pre', alpha=0.6)
plt.step(channels, np.log10(Brem3), where='pre', alpha=0.6, color='b', linewidth=0.2)
plt.step(channels, np.log10(Brem5), where='pre', alpha=0.6, color='r', linewidth=0.2)
#plt.step(channels, logging(Brem2), where='pre', alpha=0.6, color='r', linewidth=0.01)
#plt.step(channels, -np.log10(Brem2_inv), where='pre', alpha=0.5, color='r')
#plt.step(channels, logging(Brem), where='pre', alpha=0.5, color='b')
plt.axvline(75, lw=0.8)
plt.savefig('attenuated_Brem2.svg', bbox_inches = 'tight')


plt.figure(7).add_axes((0.05,0.05,1.2,0.68))
#plt.step(channels, logging(PbBrem-background), where='pre', alpha=0.6, color='r', linewidth=0.2)
plt.bar(channels, logging(PbBrem-background), width=1)
#plt.step(channels, np.log10(Brem3), where='pre', alpha=0.6, color='b', linewidth=0.2)
plt.axhline(0, color='k', linewidth=0.5)
plt.savefig('Brem2.svg', bbox_inches = 'tight')


#plt.show()