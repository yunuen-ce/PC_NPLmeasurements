import os
import matplotlib.pyplot as plt
from xarray import open_dataarray
import numpy as np
from functions import *

# Program made with Ben, the algorithm has been verified against the experimental data.

folder = os.path.expanduser('/Users/yun/Library/CloudStorage/OneDrive-Medscintinc/SpectraOutput/')
files = os.listdir('/Users/yun/Library/CloudStorage/OneDrive-Medscintinc/SpectraOutput/')

path = '/Users/yun/Library/CloudStorage/OneDrive-Medscintinc/SpectraOutput/'
Resultat = []
Mesures_bcf10_reverse = []
Mesures_bcf10_forward = []
Mesures_bcf60_reverse = []
Mesures_bcf60_forward = []
Mesures_ej204_reverse = []
Mesures_ej204_forward = []


scintillators=['bcf10', 'bcf60', 'ej204']

# for BCF-10 , ch1
# for Bcf60, ch
# for EJ204, ch2

# For the moment, only taking the 7x7 cm2 for 0, 0.2, 0.35, 0.5, 1 and 1.5 T
# Reverse
filenames_0T_reverse = [['20220921-145005.spectra', '20220921-145038.spectra', '20220921-145115.spectra'],
                        ['20220927-101247.spectra', '20220927-101316.spectra', '20220927-101344.spectra'],
                        ['20220922-141408.spectra', '20220922-141442.spectra', '20220922-141517.spectra']]
filenames_02T_reverse = [['20220921-150444.spectra', '20220921-150509.spectra', '20220921-150538.spectra'],
                         ['20220927-103143.spectra', '20220927-103211.spectra', '20220927-103240.spectra'],
                         ['20220922-143502.spectra', '20220922-143535.spectra', '20220922-143604.spectra']]
filenames_035T_reverse = [['20220921-151611.spectra', '20220921-151643.spectra', '20220921-151717.spectra'],
                          ['20220927-104113.spectra', '20220927-104154.spectra', '20220927-104222.spectra'],
                          ['20220922-145052.spectra', '20220922-145123.spectra', '20220922-145200.spectra']]
filenames_05T_reverse = [['20220921-152639.spectra', '20220921-152710.spectra', '20220921-152750.spectra'],
                         ['20220927-105553.spectra', '20220927-105621.spectra', '20220927-105650.spectra'],
                         ['20220922-150658.spectra', '20220922-150726.spectra', '20220922-150753.spectra']]
filenames_1T_reverse = [['20220921-153845.spectra', '20220921-153925.spectra', '20220921-153925.spectra'],
                        ['20220927-110519.spectra', '20220927-110552.spectra', '20220927-110622.spectra'],
                        ['20220922-151843.spectra', '20220922-151916.spectra', '20220922-151943.spectra']]
filenames_15T_reverse = [['20220921-155128.spectra', '20220921-155233.spectra', '20220921-155308.spectra'],
                         ['20220927-111558.spectra', '20220927-111628.spectra', '20220927-111703.spectra'],
                         ['20220922-152910.spectra', '20220922-152940.spectra', '20220922-153009.spectra']]
# Forward
filenames_0T_forward = [['20220922-102039.spectra', '20220922-102112.spectra', '20220922-102146.spectra'],
                        ['20220927-112556.spectra', '20220927-112623.spectra', '20220927-112649.spectra'],
                        ['20220922-125020.spectra', '20220922-125056.spectra', '20220922-125129.spectra']]
filenames_02T_forward = [['20220922-103743.spectra', '20220922-103813.spectra', '20220922-103845.spectra'],
                         ['20220927-114308.spectra', '20220927-114340.spectra', '20220927-114410.spectra'],
                        ['20220922-130647.spectra', '20220922-130727.spectra', '20220922-130802.spectra']]
filenames_035T_forward = [['20220922-104559.spectra', '20220922-104633.spectra', '20220922-104718.spectra'],
                          ['20220927-115231.spectra', '20220927-115303.spectra', '20220927-115333.spectra'],
                        ['20220922-131852.spectra', '20220922-131921.spectra', '20220922-131951.spectra']]
filenames_05T_forward = [['20220922-110026.spectra', '20220922-110059.spectra', '20220922-110126.spectra'],
                         ['20220927-120219.spectra', '20220927-120248.spectra', '20220927-120317.spectra'],
                         ['20220922-133307.spectra', '20220922-133335.spectra', '20220922-133407.spectra']]
filenames_1T_forward = [['20220922-111244.spectra', '20220922-111313.spectra', '20220922-111349.spectra'],
                        ['20220927-121301.spectra', '20220927-121330.spectra', '20220927-121401.spectra'],
                        ['20220922-134506.spectra', '20220922-134433.spectra', '20220922-134506.spectra']]
filenames_15T_forward = [['20220922-112258.spectra', '20220922-112327.spectra', '20220922-112355.spectra'],
                         ['20220927-122403.spectra', '20220927-122436.spectra', '20220927-122505.spectra'],
                         ['20220922-140151.spectra', '20220922-140229.spectra', '20220922-140304.spectra']]


for s in range(len(scintillators)):
    spectra_reverse_0T = []
    for f in filenames_0T_reverse[s]:  # for 0T
            filepath = path + f
            with open_dataarray(filepath, engine='h5netcdf') as spectra:                   #raw data
                if s == 0:
                    spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
                else :
                    spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
            Resultat.append(spectra)

    spectra_reverse_0T = np.mean(Resultat, axis=0)
    plt.plot(spectra_reverse_0T, label=scintillators[s] + ' 0 T')

    if s == 0:
        Mesures_bcf10_reverse.append(spectra_reverse_0T)
    elif s == 1:
        Mesures_bcf60_reverse.append(spectra_reverse_0T)
    elif s == 2:
        Mesures_ej204_reverse.append(spectra_reverse_0T)
    Resultat = []

    spectra_reverse_02T = []
    for f in filenames_02T_reverse[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_reverse_02T = np.mean(Resultat, axis=0)
    plt.plot(spectra_reverse_02T, label=scintillators[s] + ' 0.2 T')

    if s == 0:
        Mesures_bcf10_reverse.append(spectra_reverse_02T)
    elif s == 1:
        Mesures_bcf60_reverse.append(spectra_reverse_02T)
    elif s == 2:
        Mesures_ej204_reverse.append(spectra_reverse_02T)
    Resultat = []



    for f in filenames_035T_reverse[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_reverse_035T = np.mean(Resultat, axis=0)
    plt.plot(spectra_reverse_035T, label=scintillators[s] + ' 0.35 T')

    if s == 0:
        Mesures_bcf10_reverse.append(spectra_reverse_035T)
    elif s == 1:
        Mesures_bcf60_reverse.append(spectra_reverse_035T)
    elif s == 2:
        Mesures_ej204_reverse.append(spectra_reverse_035T)
    Resultat = []

    spectra_reverse_05T = []
    for f in filenames_05T_reverse[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_reverse_05T = np.mean(Resultat, axis=0)
    plt.plot(spectra_reverse_05T, label=scintillators[s] + ' 0.5 T')

    if s == 0:
        Mesures_bcf10_reverse.append(spectra_reverse_05T)
    elif s == 1:
        Mesures_bcf60_reverse.append(spectra_reverse_05T)
    elif s == 2:
        Mesures_ej204_reverse.append(spectra_reverse_05T)
    Resultat = []

    spectra_reverse_1T = []
    for f in filenames_1T_reverse[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_reverse_1T = np.mean(Resultat, axis=0)
    plt.plot(spectra_reverse_1T, label=scintillators[s] + ' 1 T')

    if s == 0:
        Mesures_bcf10_reverse.append(spectra_reverse_1T)
    elif s == 1:
        Mesures_bcf60_reverse.append(spectra_reverse_1T)
    elif s == 2:
        Mesures_ej204_reverse.append(spectra_reverse_1T)
    Resultat = []

    spectra_reverse_15T = []
    for f in filenames_15T_reverse[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_reverse_15T = np.mean(Resultat, axis=0)
    plt.plot(spectra_reverse_15T, label=scintillators[s] + ' 1.5 T')

    if s == 0:
        Mesures_bcf10_reverse.append(spectra_reverse_15T)
    elif s == 1:
        Mesures_bcf60_reverse.append(spectra_reverse_15T)
    elif s == 2:
        Mesures_ej204_reverse.append(spectra_reverse_15T)
    Resultat = []

    spectra_forward_0T = []
    for f in filenames_0T_forward[s]:  # for 0T
            filepath = path + f
            with open_dataarray(filepath, engine='h5netcdf') as spectra:                   #raw data
                if s == 0:
                    spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
                else :
                    spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
            Resultat.append(spectra)

    spectra_forward_0T = np.mean(Resultat, axis=0)
    plt.plot(spectra_forward_0T, label=scintillators[s] + ' 0 T')

    if s == 0:
        Mesures_bcf10_forward.append(spectra_forward_0T)
    elif s == 1:
        Mesures_bcf60_forward.append(spectra_forward_0T)
    elif s == 2:
        Mesures_ej204_forward.append(spectra_forward_0T)
    Resultat = []

    spectra_forward_02T = []
    for f in filenames_02T_forward[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_forward_02T = np.mean(Resultat, axis=0)
    plt.plot(spectra_forward_02T, label=scintillators[s] + ' 0.2 T')

    if s == 0:
        Mesures_bcf10_forward.append(spectra_forward_02T)
    elif s == 1:
        Mesures_bcf60_forward.append(spectra_forward_02T)
    elif s == 2:
        Mesures_ej204_forward.append(spectra_forward_02T)
    Resultat = []



    for f in filenames_035T_forward[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_forward_035T = np.mean(Resultat, axis=0)
    plt.plot(spectra_forward_035T, label=scintillators[s] + ' 0.35 T')

    if s == 0:
        Mesures_bcf10_forward.append(spectra_forward_035T)
    elif s == 1:
        Mesures_bcf60_forward.append(spectra_forward_035T)
    elif s == 2:
        Mesures_ej204_forward.append(spectra_forward_035T)
    Resultat = []

    spectra_forward_05T = []
    for f in filenames_05T_forward[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_forward_05T = np.mean(Resultat, axis=0)
    plt.plot(spectra_forward_05T, label=scintillators[s] + ' 0.5 T')

    if s == 0:
        Mesures_bcf10_forward.append(spectra_forward_05T)
    elif s == 1:
        Mesures_bcf60_forward.append(spectra_forward_05T)
    elif s == 2:
        Mesures_ej204_forward.append(spectra_forward_05T)
    Resultat = []

    spectra_forward_1T = []
    for f in filenames_1T_forward[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_forward_1T = np.mean(Resultat, axis=0)
    plt.plot(spectra_forward_1T, label=scintillators[s] + ' 1 T')

    if s == 0:
        Mesures_bcf10_forward.append(spectra_forward_1T)
    elif s == 1:
        Mesures_bcf60_forward.append(spectra_forward_1T)
    elif s == 2:
        Mesures_ej204_forward.append(spectra_forward_1T)
    Resultat = []

    spectra_forward_15T = []
    for f in filenames_15T_forward[s]:  # for 0T
        filepath = path + f
        with open_dataarray(filepath, engine='h5netcdf') as spectra:  # raw data
            if s == 0:
                spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')
            else:
                spectra = spectra.isel(spectrum_or_temp=0, channel=1).sum(dim='stack')
        Resultat.append(spectra)

    spectra_forward_15T = np.mean(Resultat, axis=0)
    plt.plot(spectra_forward_15T, label=scintillators[s] + ' 1.5 T')

    if s == 0:
        Mesures_bcf10_forward.append(spectra_forward_15T)
    elif s == 1:
        Mesures_bcf60_forward.append(spectra_forward_15T)
    elif s == 2:
        Mesures_ej204_forward.append(spectra_forward_15T)
    Resultat = []


plt.legend(framealpha=1, frameon=True)
plt.show()

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
Mesures_bcf10_forward = np.atleast_2d(Mesures_bcf10_forward).T
calib_doseval = 500
x1_bcf10_reverse = compute_abundances(R, Mesures_bcf10_reverse)
x1_bcf10_forward = compute_abundances(R, Mesures_bcf10_forward)
xcal1 = compute_abundances(R, calib_spectrum)
#dose1 = (x1_bcf10_reverse / xcal1 * calib_doseval)

#Cerenkov contribution
Ckov_w_bcf10_reverse= x1_bcf10_reverse[2, :] + x1_bcf10_reverse[3, :]
Ckov_w_bcf10_forward= x1_bcf10_forward[2, :] + x1_bcf10_forward[3, :]
Scint_w_bcf10_reverse = x1_bcf10_reverse[0]
Fluo_w_bcf10_reverse = x1_bcf10_reverse[1]
Scint_w_bcf10_forward = x1_bcf10_forward[0]
Fluo_w_bcf10_forward = x1_bcf10_forward[1]

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
Mesures_ej204_forward = np.atleast_2d(Mesures_ej204_forward).T
calib_doseval = 500
x1_ej204_reverse = compute_abundances(R, Mesures_ej204_reverse)
x1_ej204_forward = compute_abundances(R, Mesures_ej204_forward)
xcal1 = compute_abundances(R, calib_spectrum)
#dose1 = (x1_ej204_reverse / xcal1 * calib_doseval)


#Cerenkov contribution
Ckov_w_ej204_reverse = x1_ej204_reverse[2, :] + x1_ej204_reverse[3, :]
Ckov_w_ej204_forward = x1_ej204_forward[2, :] + x1_ej204_forward[3, :]
Scint_w_ej204_reverse = x1_ej204_reverse[0]
Fluo_w_ej204_reverse = x1_ej204_reverse[1]
Scint_w_ej204_forward= x1_ej204_forward[0]
Fluo_w_ej204_forward = x1_ej204_forward[1]

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
Mesures_bcf60_forward = np.atleast_2d(Mesures_bcf60_forward).T
calib_doseval = 500
x1_bcf60_reverse = compute_abundances(R, Mesures_bcf60_reverse)
x1_bcf60_forward = compute_abundances(R, Mesures_bcf60_forward)
xcal1 = compute_abundances(R, calib_spectrum)
dose1 = (x1_bcf60_reverse / xcal1 * calib_doseval)

# Cerenkov contribution
Ckov_w_bcf60_reverse= x1_bcf60_reverse[2, :] + x1_bcf60_reverse[3, :]
Ckov_w_bcf60_forward= x1_bcf60_forward[2, :] + x1_bcf60_forward[3, :]
Scint_w_bcf60_reverse = x1_bcf60_reverse[0]
Fluo_w_bcf60_reverse = x1_bcf60_reverse[1]
Scint_w_bcf60_forward = x1_bcf60_forward[0]
Fluo_w_bcf60_forward = x1_bcf60_forward[1]


bfields=[0, 0.2, 0.35, 0.5, 1,  1.5]
fig, axes = plt.subplots(nrows=2, ncols=1,  gridspec_kw={'height_ratios': [2, 1]})
plt.subplot(211)
plt.plot(bfields, Ckov_w_bcf10_forward / Ckov_w_bcf10_forward[0] , '+-', color='tomato', label='BCF-10, e- \u2192 stem ')
plt.plot(bfields, Ckov_w_bcf60_forward / Ckov_w_bcf60_forward[0] , '+-', color='deepskyblue', label='BCF-60, e- \u2192 stem')
plt.plot(bfields, Ckov_w_ej204_forward / Ckov_w_ej204_forward[0] , '+-', color='mediumseagreen',  label='EJ-204, e- \u2192 stem')
plt.plot(bfields, Ckov_w_bcf10_reverse / Ckov_w_bcf10_reverse[0], '+:',color='tomato', label='BCF-10, e- \u2192 tip')
plt.plot(bfields, Ckov_w_bcf60_reverse / Ckov_w_bcf60_reverse[0], '+:',color='deepskyblue', label='BCF-60, e- \u2192 tip')
plt.plot(bfields, Ckov_w_ej204_reverse / Ckov_w_ej204_reverse[0], '+:', color='mediumseagreen', label='EJ2-04, e- \u2192 tip')
plt.ylabel('Cerenkov abundance [B]/ Cerenkov abundance [0 T]')

plt.subplot(212)
plt.plot(bfields, (Ckov_w_bcf10_forward / Ckov_w_bcf10_forward[0] - Ckov_w_bcf10_reverse / Ckov_w_bcf10_reverse[0])/ (Ckov_w_bcf10_reverse / Ckov_w_bcf10_reverse[0]) , '+:',color='tomato', label='BCF-10')
plt.plot(bfields, (Ckov_w_bcf60_forward / Ckov_w_bcf60_forward[0] - Ckov_w_bcf60_reverse / Ckov_w_bcf60_reverse[0])/ (Ckov_w_bcf60_reverse / Ckov_w_bcf60_reverse[0]) , '+:',color='deepskyblue', label='BCF-60')
plt.plot(bfields, (Ckov_w_ej204_forward / Ckov_w_ej204_forward[0] - Ckov_w_ej204_reverse / Ckov_w_ej204_reverse[0])/ (Ckov_w_ej204_reverse / Ckov_w_ej204_reverse[0]) , '+:', color='mediumseagreen', label='EJ-204')
plt.xlabel('Magnetic field [T]')
plt.ylabel('Relative difference \n between orientations')
plt.legend(framealpha=1, frameon=True)
plt.tight_layout()
plt.show()

plt.plot(bfields, Scint_w_bcf10_forward / Scint_w_bcf10_forward[0], '+-', color='tomato', label='BCF-10, e- \u2192 stem ')
plt.plot(bfields, Scint_w_bcf60_forward / Scint_w_bcf60_forward[0], '+-', color='deepskyblue', label='BCF-60, e- \u2192 stem')
plt.plot(bfields, Scint_w_ej204_forward / Scint_w_ej204_forward[0], '+-', color='mediumseagreen',  label='EJ-204, e- \u2192 stem')
plt.plot(bfields, Scint_w_bcf10_reverse / Scint_w_bcf10_reverse[0], '+:', color='tomato', label='BCF-10, e- \u2192 tip')
plt.plot(bfields, Scint_w_bcf60_reverse / Scint_w_bcf60_reverse[0], '+:', color='deepskyblue', label='BCF-60, e- \u2192 tip')
plt.plot(bfields, Scint_w_ej204_reverse / Scint_w_ej204_reverse[0], '+:', color='mediumseagreen', label='EJ-204, e- \u2192 tip')

plt.xlabel('Magnetic field [T]')
plt.ylabel('Scint abundance / Scint abundance [0 T]')
plt.legend(framealpha=1, frameon=True)
plt.tight_layout()
plt.show()

plt.plot(bfields, Fluo_w_bcf10_forward / Fluo_w_bcf10_forward[0], '+-',  color='tomato', label='BCF-10, e- \u2192 stem ')
plt.plot(bfields, Fluo_w_bcf60_forward / Fluo_w_bcf60_forward[0], '+-', color='deepskyblue', label='BCF-60, e- \u2192 stem')
plt.plot(bfields, Fluo_w_ej204_forward / Fluo_w_ej204_forward[0], '+-', color='mediumseagreen',  label='EJ204, e- \u2192 stem')
plt.plot(bfields, Fluo_w_bcf10_reverse / Fluo_w_bcf10_reverse[0], '+:', color='tomato', label='BCF-10, e- \u2192 tip')
plt.plot(bfields, Fluo_w_bcf60_reverse / Fluo_w_bcf60_reverse[0], '+:', color='deepskyblue', label='BCF-60, e- \u2192 tip')
plt.plot(bfields, Fluo_w_ej204_reverse / Fluo_w_ej204_reverse[0], '+:',  color='mediumseagreen', label='EJ204, e- \u2192 tip')
plt.xlabel('Magnetic field [T]')
plt.ylabel('Fluo abundance / Fluo abundance [0 T]')
plt.legend(framealpha=1, frameon=True)
plt.tight_layout()
plt.show()

legend=['0 T', '0.2 T', '0.35 T', '0.5 T', '1 T',  '1.5 T']
plt.subplot(231)
plt.plot(Mesures_bcf10_reverse / np.sum(Mesures_bcf10_reverse[:, 0]), label=legend)
plt.legend(framealpha=1, frameon=True)
plt.title('BCF-10 _reverse')
plt.subplot(232)
plt.plot(Mesures_ej204_reverse / np.sum(Mesures_ej204_reverse[:, 0]), label=legend)
plt.title('EJ204 _reverse')
plt.legend(framealpha=1, frameon=True)
plt.subplot(233)
plt.plot(Mesures_bcf60_reverse / np.sum(Mesures_bcf60_reverse[:, 0]), label=legend)
plt.title('BCF-60 _reverse')
plt.legend(framealpha=1, frameon=True)
plt.subplot(234)
plt.plot(Mesures_bcf10_forward / np.sum(Mesures_bcf10_forward[:, 0]), label=legend)
plt.legend(framealpha=1, frameon=True)
plt.title('BCF-10 _forward')
plt.subplot(235)
plt.plot(Mesures_ej204_forward/ np.sum(Mesures_ej204_forward[:, 0]), label=legend)
plt.title('EJ204 _forward')
plt.legend(framealpha=1, frameon=True)
plt.subplot(236)
plt.plot(Mesures_bcf60_forward / np.sum(Mesures_bcf60_forward[:, 0]), label=legend)
plt.title('BCF-60 _forward')
plt.legend(framealpha=1, frameon=True)
plt.tight_layout()
plt.show()
