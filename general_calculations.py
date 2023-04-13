import numpy as np
import math
import matplotlib.pyplot as plt

# Physical
r0 = 2.8179403267e-13  # en cm
NA = 6.022e23  # en /mol
mc2 = 0.511  # MeV
c = 299792458  # m/s


def energy_threshold(ref_index):
    e_th = mc2 * ((1 / (np.sqrt(1 - (1 / ref_index ** 2)))) - 1)
    return e_th


def cerenkov_emission_angle(ref_index, energy):
    theta = math.degrees(math.acos(1 / (ref_index * np.sqrt(1 - (1 / (energy / mc2 + 1)) ** 2))))
    return theta


def gamma(energy):
    gamma = energy / mc2 + 1
    return gamma


def beta(energy):
    beta = np.sqrt(1 - 1 / gamma(energy) ** 2)
    return (beta)


def gyration_radius(energy, magnetic_field):
    rg = (gamma(energy)*beta(energy)*mc2*1e9) / (magnetic_field*c)
    return rg


def energy_gyration_radius(magnetic_field, rg):
    e_gr = (np.sqrt((rg / (1e9 *mc2/(magnetic_field * c)))**2 + 1 ) - 1)*mc2
    return e_gr


def fxy(x, y, energy, magnetic_field):
    fxy = (x-(gyration_radius(energy, magnetic_field)) )**2 + (y)**2 - gyration_radius(energy, magnetic_field)**2
    return fxy


# Cervenkov threshold energies

n = 1.333  # water
E_th = energy_threshold(n)

#Figures

T = np.arange(0, 6, 0.1)
plt.plot(T, gyration_radius(T, 1.5), label='1.5 T')
plt.plot(T, gyration_radius(T, 1), label='1 T')
plt.plot(T, gyration_radius(T, 0.5), label='0.5 T')
plt.plot(T, gyration_radius(T, 0.2), label='0.2 T')
plt.axvline(x = E_th)
plt.xscale("log")
plt.xlabel('Energy [MeV]')
plt.ylabel('Gyration radius [mm]')
plt.legend(framealpha=1, frameon=True)
plt.show()

T = np.arange(0, 6, 0.1)
plt.plot(T, np.pi*gyration_radius(T, 1.5) -1, label='1.5 T')
plt.plot(T, np.pi*gyration_radius(T, 1)-1, label='1 T')
plt.plot(T, np.pi*gyration_radius(T, 0.5)-1, label='0.5 T')
plt.plot(T, np.pi*gyration_radius(T, 0.2)-1, label='0.2 T')
plt.axvline(x = E_th, label = 'Cerenkov threshold', color='black')
#plt.axvline(x = energy_gyration_radius(1.5, 1), color='red')
plt.ylim([0, 50])
plt.xscale("log")
plt.xlabel('Energy [MeV]')
plt.ylabel('Pathlenght correction [mm]')
plt.legend(framealpha=1, frameon=True)
plt.show()


# Cervenkov threshold energies

n = 1.333  # water
E_th = energy_threshold(n)
theta = cerenkov_emission_angle(n, 6)
print(6, theta)
theta = cerenkov_emission_angle(n, E_th + 0.001)
print(E_th, theta)

# gyration radius
print(beta(E_th))


#Considering the cavity geometry

d = 1 # cavity diameter
'''
T = np.arange(0, energy_gyration_radius(0.5, 1), 0.01)
plt.plot(T, np.pi*gyration_radius(T, 0.5)-1, label='0.2 T')
plt.axvline(x = E_th, label = 'Cerenkov threshold')
#plt.axvline(x = energy_gyration_radius(1.5, 1), color='red')
#plt.ylim([0, 100])
plt.xscale("log")
plt.xlabel('Energy [MeV]')
plt.ylabel('Pathlenght correction [mm]')
plt.legend(framealpha=1, frameon=True)

plt.subplot(122)
'''
T = np.arange(0, energy_gyration_radius(1.5, 1), 0.01)
plt.plot(T, np.pi*gyration_radius(T, 1.5)-1, label='1.5 T')
T = np.arange(0, energy_gyration_radius(1, 1), 0.01)
plt.plot(T, np.pi*gyration_radius(T, 1)-1, label='1 T')
T = np.arange(0, energy_gyration_radius(0.5, 1), 0.01)
plt.plot(T, np.pi*gyration_radius(T, 0.5)-1, label='0.5 T')
T = np.arange(0, energy_gyration_radius(0.2, 1), 0.01)
plt.plot(T, np.pi*gyration_radius(T, 0.2)-1, label='0.2 T')
plt.axvline(x = E_th, label = 'Cerenkov threshold', color='black')
#plt.axvline(x = energy_gyration_radius(1.5, 1), color='red')
#plt.ylim([0, 100])
plt.xscale("log")
plt.xlabel('Energy [MeV]')
plt.ylabel('Pathlenght correction [mm]')
plt.legend(framealpha=1, frameon=True)
plt.show()
plt.show()
