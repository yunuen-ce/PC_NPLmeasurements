# Substraction method: removing the Cerenkov for  0 T, 0.5 T and 1.5 T, perpendicular orientation reverse
import os
import matplotlib.pyplot as plt
from xarray import open_dataarray
import numpy as np
from functions import *

folder = os.path.expanduser('/Users/yun/Library/CloudStorage/OneDrive-Medscintinc/SpectraOutput/')
files = os.listdir('/Users/yun/Library/CloudStorage/OneDrive-Medscintinc/SpectraOutput/')

path = '/Users/yun/Library/CloudStorage/OneDrive-Medscintinc/SpectraOutput/'
colors = ['tomato', 'deepskyblue', 'mediumseagreen', 'purple', 'grey', 'darkorange',  'plum', 'black',]

# Detector data, order
scintillators = ['bcf10', 'bcf60', 'ej204', 'bcf60-Lee', 'Medscint']
filenames_0T_reverse = [['20220921-145005.spectra', '20220921-145038.spectra', '20220921-145115.spectra'],
                        ['20220927-101247.spectra', '20220927-101316.spectra', '20220927-101344.spectra'],
                        ['20220922-141408.spectra', '20220922-141442.spectra', '20220922-141517.spectra'],
                        ['20220927-141540.spectra', '20220927-141644.spectra', '20220927-142548.spectra'],
                        ['20220923-110407.spectra', '20220923-110440.spectra', '20220923-110513.spectra']]

filenames_05T_reverse = [['20220921-152639.spectra', '20220921-152710.spectra', '20220921-152750.spectra'],
                         ['20220927-105553.spectra', '20220927-105621.spectra', '20220927-105650.spectra'],
                         ['20220922-150658.spectra', '20220922-150726.spectra', '20220922-150753.spectra'],
                         ['20220927-160001.spectra', '20220927-160025.spectra', '20220927-160051.spectra'],
                         ['20220923-114238.spectra', '20220923-114333.spectra', '20220923-114414.spectra']]

filenames_15T_reverse = [['20220921-155128.spectra', '20220921-155233.spectra', '20220921-155308.spectra'],
                         ['20220927-111558.spectra', '20220927-111628.spectra', '20220927-111703.spectra'],
                         ['20220922-152910.spectra', '20220922-152940.spectra', '20220922-153009.spectra'],
                         ['20220927-161940.spectra', '20220927-162009.spectra', '20220927-162037.spectra'],
                         ['20220923-120435.spectra', '20220923-120508.spectra', '20220923-120541.spectra']]


Mesures_bcf10_reverse = []
Mesures_bcf10_forward = []
Mesures_bcf60_reverse = []
Mesures_bcf60_forward = []
Mesures_ej204_reverse = []
Mesures_ej204_forward = []
Mesures_bcf60_lee_reverse = []
Mesures_bcf60_lee_forward = []
Mesures_medscint_reverse = []
Mesures_medscint_forward = []

bfields=[0, 0.5, 1.5]
Mesures_reverse = np.zeros([len(scintillators), len(bfields), 137])
ckov_perp=np.zeros([len(bfields), 137])
#clear fiber
# for the clear fiber
filenames_0T_para = ['20220926-153523.spectra', '20220926-153600.spectra', '20220926-153649.spectra',
                     '20220926-153727.spectra', '20220926-153806.spectra']
filenames_0T_perp = ['20220926-160539.spectra', '20220926-160614.spectra', '20220926-160644.spectra',
                     '20220926-160715.spectra', '20220926-160746.spectra']
filenames_05T_para = ['20220926-154526.spectra', '20220926-154559.spectra', '20220926-154634.spectra',
                      '20220926-154713.spectra', '20220926-154753.spectra']
filenames_05T_perp = ['20220926-160941.spectra', '20220926-161017.spectra', '20220926-161049.spectra',
                      '20220926-161124.spectra', '20220926-161159.spectra']
filenames_15T_para = ['20220926-155146.spectra', '20220926-155221.spectra', '20220926-155249.spectra',
                      '20220926-155319.spectra', '20220926-155350.spectra']
filenames_15T_perp = ['20220926-161520.spectra', '20220926-161558.spectra', '20220926-161635.spectra',
                      '20220926-161713.spectra', '20220926-161751.spectra']

Resultat = []
for f in filenames_0T_para:  # for 0T
    filepath = path + f
    with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
    Resultat.append(spectra)
ckov_para_0T = np.mean(Resultat, axis=0)

Resultat = []
for f in filenames_05T_para:  # for 0T
    filepath = path + f
    with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
    Resultat.append(spectra)
ckov_para_05T = np.mean(Resultat, axis=0)

Resultat = []
for f in filenames_15T_para:  # for 0T
    filepath = path + f
    with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
    Resultat.append(spectra)
ckov_para_15T = np.mean(Resultat, axis=0)

Resultat = []
for f in filenames_0T_perp:  # for 0T
    filepath = path + f
    with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
    Resultat.append(spectra)
ckov_perp_0T = np.mean(Resultat, axis=0)

Resultat = []
for f in filenames_05T_perp:  # for 0T
    filepath = path + f
    with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
    Resultat.append(spectra)
ckov_perp_05T = np.mean(Resultat, axis=0)

Resultat = []
for f in filenames_15T_perp:  # for 0T
    filepath = path + f
    with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
    Resultat.append(spectra)
ckov_perp_15T = np.mean(Resultat, axis=0)
ckov_perp[0, :] =ckov_perp_0T
ckov_perp[1, :] =ckov_perp_05T
ckov_perp[2, :] =ckov_perp_15T
plt.subplot(121)
plt.plot(ckov_para_0T / np.sum(ckov_para_0T), ':', label='0 T')
plt.plot(ckov_para_05T / np.sum(ckov_para_0T), '--', label='0.5 T')
plt.plot(ckov_para_15T / np.sum(ckov_para_0T), '-.', label='1.5 T')
plt.ylabel('Intensity(B)/Area under the curve(0T)')
plt.xlabel('Pixel number')
plt.legend(framealpha=1, frameon=True)
# plt.legend(framealpha=1, frameon=True, loc='center right', bbox_to_anchor=(1.25, 0.5))
plt.title('Parallel')

plt.subplot(122)
plt.plot(ckov_perp_0T / np.sum(ckov_perp_0T), ':', label='0 T')
plt.plot(ckov_perp_05T / np.sum(ckov_perp_0T), '--', label='0.5 T')
plt.plot(ckov_perp_15T / np.sum(ckov_perp_0T), '-.', label='1.5 T')
plt.xlabel('Pixel number')
plt.legend(framealpha=1, frameon=True)
plt.title('Perpendicular')
plt.tight_layout()
plt.show()


# Data from det

for s in range(len(scintillators)):
    spectra_reverse_0T = []
    for f in filenames_0T_reverse[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0 or s == 3:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)
    spectra_reverse_0T = np.mean(Resultat, axis=0)
    Mesures_reverse[s, 0, :] = spectra_reverse_0T
    if s == 0:
        Mesures_bcf10_reverse.append(spectra_reverse_0T)
    elif s == 1:
        Mesures_bcf60_reverse.append(spectra_reverse_0T)
    elif s == 2:
        Mesures_ej204_reverse.append(spectra_reverse_0T)
    elif s == 3:
        Mesures_bcf60_lee_reverse.append(spectra_reverse_0T)
    elif s == 4:
        Mesures_medscint_reverse.append(spectra_reverse_0T)


    Resultat = []


    for f in filenames_05T_reverse[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0 or s == 3:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_reverse_05T = np.mean(Resultat, axis=0)
    Mesures_reverse[s, 1, :] = spectra_reverse_05T

    if s == 0:
        Mesures_bcf10_reverse.append(spectra_reverse_05T)
    elif s == 1:
        Mesures_bcf60_reverse.append(spectra_reverse_05T)
    elif s == 2:
        Mesures_ej204_reverse.append(spectra_reverse_05T)
    elif s == 3:
        Mesures_bcf60_lee_reverse.append(spectra_reverse_05T)
    elif s == 4:
        Mesures_medscint_reverse.append(spectra_reverse_05T)
    Resultat = []



    spectra_reverse_15T = []
    for f in filenames_15T_reverse[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0 or s == 3:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_reverse_15T = np.mean(Resultat, axis=0)
    Mesures_reverse[s, 2, :] = spectra_reverse_15T

    if s == 0:
        Mesures_bcf10_reverse.append(spectra_reverse_15T)
    elif s == 1:
        Mesures_bcf60_reverse.append(spectra_reverse_15T)
    elif s == 2:
        Mesures_ej204_reverse.append(spectra_reverse_15T)
    elif s == 3:
        Mesures_bcf60_lee_reverse.append(spectra_reverse_15T)
    elif s == 4:
        Mesures_medscint_reverse.append(spectra_reverse_15T)
    Resultat = []

# Do the susbstraction
temp=131
for b in range(len(bfields)):
    for s in range(len(scintillators)):
        plt.subplot(temp)
        plt.plot(Mesures_reverse[s, b, :], ':', color=colors[s],  )
        plt.plot(Mesures_reverse[s, b, :] - ckov_perp[b, :], '-', color=colors[s], label=scintillators[s])
    plt.xlabel('Pixel number')
    plt.legend(framealpha=1, frameon=True)
    plt.title(str(bfields[b]) + ' T')
    temp = temp + 1
plt.show()

'''
# # This is to get the weights 

# BCF-10, ch1
scint_file = folder + '20220920-141652-scint.spectra'
fluo_file = folder + '20220920-143043-fluo.spectra'
ckov1_file = folder + '20220920-161025-cerenkov1.spectra'
ckov2_file = folder + '20220920-161218-cerenkov2.spectra'
ckov3_file = folder + '20220920-161924-cerenkov3.spectra'
ckov4_file = folder + '20220920-162104-cerenkov4.spectra'
dose_file = folder + '20220921-105631-normalization_all.spectra'

scint = open_spectra(scint_file)
fluo = open_spectra(fluo_file)
ckov1 = open_spectra(ckov1_file)
ckov2 = open_spectra(ckov2_file)
ckov3 = open_spectra(ckov3_file)
ckov4 = open_spectra(ckov4_file)
ckovA = abs(ckov1 - ckov2)
ckovB = abs(ckov3 - ckov4)
dose = open_spectra(dose_file, channel=0)

# To obtain the abundance
calib_spectrum = np.atleast_2d(dose).T  # nb_wavelength x nb_measure
R = make_calib_spectra(scint, fluo, ckov1, ckov2, ckov3, ckov4)  # nb_wavelength x nb_spectra
Mesures_bcf10_reverse = np.atleast_2d(Mesures_bcf10_reverse).T
calib_doseval = 500
x1_bcf10_reverse = compute_abundances(R, Mesures_bcf10_reverse)
xcal1 = compute_abundances(R, calib_spectrum)
# dose1 = (x1_bcf10_reverse / xcal1 * calib_doseval)

# Cerenkov contribution
Ckov_w_bcf10_reverse = x1_bcf10_reverse[2, :] + x1_bcf10_reverse[3, :]
Scint_w_bcf10_reverse = x1_bcf10_reverse[0]
Fluo_w_bcf10_reverse = x1_bcf10_reverse[1]

# EJ-204, ch2

dose_file = folder + '20220922-115954-normalization_all.spectra'

scint = open_spectra(scint_file)
fluo = open_spectra(fluo_file)
ckov1 = open_spectra(ckov1_file)
ckov2 = open_spectra(ckov2_file)
ckov3 = open_spectra(ckov3_file)
ckov4 = open_spectra(ckov4_file)
ckovA = abs(ckov1 - ckov2)
ckovB = abs(ckov3 - ckov4)
dose = open_spectra(dose_file, channel=1)

# To obtain the abundance
calib_spectrum = np.atleast_2d(dose).T  # nb_wavelength x nb_measure
R = make_calib_spectra(scint, fluo, ckov1, ckov2, ckov3, ckov4)  # nb_wavelength x nb_spectra
Mesures_ej204_reverse = np.atleast_2d(Mesures_ej204_reverse).T
calib_doseval = 500
x1_ej204_reverse = compute_abundances(R, Mesures_ej204_reverse)
xcal1 = compute_abundances(R, calib_spectrum)
# dose1 = (x1_ej204_reverse / xcal1 * calib_doseval)

# Cerenkov contribution
Ckov_w_ej204_reverse = x1_ej204_reverse[2, :] + x1_ej204_reverse[3, :]
Scint_w_ej204_reverse = x1_ej204_reverse[0]
Fluo_w_ej204_reverse = x1_ej204_reverse[1]


# BCF-60, ch2
scint_file = folder + '20220927-085310-scint.spectra'
fluo_file = folder + '20220927-090118-fluo.spectra'
ckov1_file = folder + '20220927-092943-cerenkov1.spectra'
ckov2_file = folder + '20220927-093133-cerenkov2.spectra'
ckov3_file = folder + '20220927-093609-cerenkov3.spectra'
ckov4_file = folder + '20220927-093753-cerenkov4.spectra'
dose_file = folder + '20220927-095656-normalization_all.spectra'

scint = open_spectra(scint_file)
fluo = open_spectra(fluo_file)
ckov1 = open_spectra(ckov1_file)
ckov2 = open_spectra(ckov2_file)
ckov3 = open_spectra(ckov3_file)
ckov4 = open_spectra(ckov4_file)
ckovA = abs(ckov1 - ckov2)
ckovB = abs(ckov3 - ckov4)
dose = open_spectra(dose_file, channel=1)

# To obtain the abundance
calib_spectrum = np.atleast_2d(dose).T  # nb_wavelength x nb_measure
R = make_calib_spectra(scint, fluo, ckov1, ckov2, ckov3, ckov4)  # nb_wavelength x nb_spectra
Mesures_bcf60_reverse = np.atleast_2d(Mesures_bcf60_reverse).T
calib_doseval = 500
x1_bcf60_reverse = compute_abundances(R, Mesures_bcf60_reverse)
xcal1 = compute_abundances(R, calib_spectrum)
dose1 = (x1_bcf60_reverse / xcal1 * calib_doseval)

# Cerenkov contribution
Ckov_w_bcf60_reverse = x1_bcf60_reverse[2, :] + x1_bcf60_reverse[3, :]
Scint_w_bcf60_reverse = x1_bcf60_reverse[0]
Fluo_w_bcf60_reverse = x1_bcf60_reverse[1]

# BCF-60 Lee filter , ch1
scint_file = folder + '20220927-085310-scint.spectra'
fluo_file = folder + '20220927-090118-fluo.spectra'
ckov1_file = folder + '20220927-092943-cerenkov1.spectra'
ckov2_file = folder + '20220927-093133-cerenkov2.spectra'
ckov3_file = folder + '20220927-093609-cerenkov3.spectra'
ckov4_file = folder + '20220927-093753-cerenkov4.spectra'
dose_file = folder + '20220927-140521-normalization_all.spectra'

scint = open_spectra(scint_file)
fluo = open_spectra(fluo_file)
ckov1 = open_spectra(ckov1_file)
ckov2 = open_spectra(ckov2_file)
ckov3 = open_spectra(ckov3_file)
ckov4 = open_spectra(ckov4_file)
ckovA = abs(ckov1 - ckov2)
ckovB = abs(ckov3 - ckov4)
dose = open_spectra(dose_file, channel=0)

# To obtain the abundance
calib_spectrum = np.atleast_2d(dose).T  # nb_wavelength x nb_measure
R = make_calib_spectra(scint, fluo, ckov1, ckov2, ckov3, ckov4)  # nb_wavelength x nb_spectra
Mesures_bcf60_lee_reverse = np.atleast_2d(Mesures_bcf60_lee_reverse).T
calib_doseval = 500
x1_bcf60_lee_reverse = compute_abundances(R, Mesures_bcf60_lee_reverse)
xcal1 = compute_abundances(R, calib_spectrum)
dose1 = (x1_bcf60_lee_reverse / xcal1 * calib_doseval)

# Cerenkov contribution
Ckov_w_bcf60_lee_reverse = x1_bcf60_lee_reverse[2, :] + x1_bcf60_lee_reverse[3, :]
Scint_w_bcf60_lee_reverse = x1_bcf60_lee_reverse[0]
Fluo_w_bcf60_lee_reverse = x1_bcf60_lee_reverse[1]


# Medscint, ch2

scint_file = folder + '20220922-164231-scint.spectra'
fluo_file = folder + '20220922-165214-fluo.spectra'
ckov1_file = folder + '20220922-170040-cerenkov1.spectra'
ckov2_file = folder + '20220922-170232-cerenkov2.spectra'
ckov3_file = folder + '20220922-170618-cerenkov3.spectra'
ckov4_file = folder + '20220922-170807-cerenkov4.spectra'
dose_file = folder + '20220923-104901-normalization_all.spectra'

scint = open_spectra(scint_file)
fluo = open_spectra(fluo_file)
ckov1 = open_spectra(ckov1_file)
ckov2 = open_spectra(ckov2_file)
ckov3 = open_spectra(ckov3_file)
ckov4 = open_spectra(ckov4_file)
ckovA = abs(ckov1 - ckov2)
ckovB = abs(ckov3 - ckov4)
dose = open_spectra(dose_file, channel=1)

# To obtain the abundance
calib_spectrum = np.atleast_2d(dose).T  # nb_wavelength x nb_measure
R = make_calib_spectra(scint, fluo, ckov1, ckov2, ckov3, ckov4)  # nb_wavelength x nb_spectra
Mesures_medscint_reverse = np.atleast_2d(Mesures_medscint_reverse).T
calib_doseval = 500
x1_medscint_reverse = compute_abundances(R, Mesures_medscint_reverse)
xcal1 = compute_abundances(R, calib_spectrum)
dose1 = (x1_medscint_reverse / xcal1 * calib_doseval)

# Cerenkov contribution
Ckov_w_medscint_reverse = x1_medscint_reverse[2, :] + x1_medscint_reverse[3, :]
Scint_w_medscint_reverse = x1_medscint_reverse[0]
Fluo_w_medscint_reverse = x1_medscint_reverse[1]
'''

