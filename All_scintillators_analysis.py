import pandas as pd
from numpy import linalg, ndarray, vstack
from base64 import decodebytes as b64_decode
from json import loads
import matplotlib.pyplot as plt
import glob
import numpy as np
from xarray import open_dataarray
import os


def compute_icm(components):  # où component est une liste des spectres (des numpy array)
    r_matrix = vstack(components)
    return (linalg.inv(r_matrix @ r_matrix.transpose()) @ r_matrix)


def generate_weights(icm,
                     spectrum):  # où icm est la matrice calculee par compute icm et spectrum est un numpy array. Retourne les poids correspondants aux components de l'icm
    transposed_spectrum = vstack(spectrum)
    weights = (icm @ transposed_spectrum).squeeze()
    return weights


def agnostic_read_spectra(file_path):
    with open(file_path, "rb") as f:
        content_lines = f.readlines()
    content_str = [str(b64_decode(encoded_line), "utf-8") for encoded_line in content_lines]
    payload = "".join(content_str[2:])
    payload_dict = loads(payload)
    spectra_per_channel = payload_dict["channels_spectra"]
    return spectra_per_channel


def summed_spectra(file_path, channel_id="channel_1"):
    spectra_dict = agnostic_read_spectra(file_path)
    channel_spectra = spectra_dict[channel_id]["spectra"]
    for ii, spectrum in enumerate(channel_spectra):
        if ii == 0:
            array_spectrum = np.array(spectrum["spectrum"])
        else:
            array_spectrum += np.array(spectrum["spectrum"])
    array_spectrum_n = array_spectrum / np.sum(array_spectrum)
    return array_spectrum


def spectrum_chopper(spectrum, limits):
    spectrum = spectrum[limits[0]:limits[1]]
    return spectrum


def summed_spectra_old(file_path, channel_nb=0):
    with open_dataarray(file_path, engine='h5netcdf') as spectra:
        array_spectrum_n = spectra.isel(spectrum_or_temp=0, channel=channel_nb).sum(dim='stack')
    return np.array(array_spectrum_n)


# ----------Experimental data from Excel file---------------
# Read data
path = '/Users/yun/PycharmProjects/NPLmeasurements/'
filename = path + 'NPLdata.txt'
df_all = pd.read_csv(filename, sep='\t')
df = df_all[['Detector', 'Orientation', 'Current', 'Bfield', 'Field_size', 'Dose_Avg', 'D_SDOM', 'Diff_0T', 'OF',
             'OF_Diff_0T']]

fields = [0.5, 1, 7]
bfields = [0.2, 0.35, 0.5, 1, 1.5]
orientations = ['perpendicular', 'parallel']
currents = ['Forward', 'Reverse']
detectors = ['BCF-60', 'BCF-60 Lee filter', 'BCF-10', 'EJ-204', 'Medscint 197-2']
colors = ['tomato', 'deepskyblue', 'mediumseagreen', 'purple', 'grey', 'darkorange', 'plum', 'black', ]
lin = ['solid', 'dashed']
symbol = ['d', '^', 's', 'o', '*' ]

# All det


# --- read data ----

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
Resultat = []
Resultat_0T = []
Data = np.zeros([5, 42, 137])
Mean_dose = np.zeros([5, 12])
Mean_fluo = np.zeros([5, 12])
Mean_ckov = np.zeros([5, 12])
std_dose = np.zeros([5, 12])
std_fluo = np.zeros([5, 12])
std_ckov = np.zeros([5, 12])
Mean_spectrum = np.zeros([5, 12, 137])

bfields = [0, 0.2, 0.35, 0.5, 1, 1.5]

colors = ['tomato', 'deepskyblue', 'mediumseagreen', 'purple', 'grey', 'darkorange', 'plum', 'black']
lin = [':', 'dashed']

scintillators = ['bcf10', 'bcf60', 'ej204', 'bcf60-Lee', 'Medscint']
scint_label = ['BCF-10', 'BCF-60', 'EJ-204', 'BCF-60 Lee filter', 'Medscint']
path = '/Users/yun/Library/CloudStorage/OneDrive-Medscintinc/SpectraOutput/'
folder = os.path.expanduser(path)
files = os.listdir(path)
channel = [0, 1, 1, 0, 1]

# The abundance in the order of scintillators
# For the moment, only taking the 7x7 cm2 for 0, 0.2, 0.35, 0.5, 1 and 1.5 T
# Reverse

filenames_02T_reverse = [['20220921-150444.spectra', '20220921-150509.spectra', '20220921-150538.spectra'],
                         ['20220927-103143.spectra', '20220927-103211.spectra', '20220927-103240.spectra'],
                         ['20220922-143502.spectra', '20220922-143535.spectra', '20220922-143604.spectra'],
                         ['20220927-153942.spectra', '20220927-154009.spectra', '20220927-154039.spectra'],
                         ['20220923-111908.spectra', '20220923-111941.spectra', '20220923-112016.spectra']]
filenames_035T_reverse = [['20220921-151611.spectra', '20220921-151643.spectra', '20220921-151717.spectra'],
                          ['20220927-104113.spectra', '20220927-104154.spectra', '20220927-104222.spectra'],
                          ['20220922-145052.spectra', '20220922-145123.spectra', '20220922-145200.spectra'],
                          ['20220927-154917.spectra', '20220927-154944.spectra', '20220927-155011.spectra'],
                          ['20220923-113218.spectra', '20220923-113249.spectra', '20220923-113320.spectra']]
filenames_05T_reverse = [['20220921-152639.spectra', '20220921-152710.spectra', '20220921-152750.spectra'],
                         ['20220927-105553.spectra', '20220927-105621.spectra', '20220927-105650.spectra'],
                         ['20220922-150658.spectra', '20220922-150726.spectra', '20220922-150753.spectra'],
                         ['20220927-160001.spectra', '20220927-160025.spectra', '20220927-160051.spectra'],
                         ['20220923-114238.spectra', '20220923-114333.spectra', '20220923-114414.spectra']]
filenames_1T_reverse = [['20220921-153845.spectra', '20220921-153925.spectra', '20220921-153925.spectra'],
                        ['20220927-110519.spectra', '20220927-110552.spectra', '20220927-110622.spectra'],
                        ['20220922-151843.spectra', '20220922-151916.spectra', '20220922-151943.spectra'],
                        ['20220927-160906.spectra', '20220927-160931.spectra', '20220927-160956.spectra'],
                        ['20220923-115351.spectra', '20220923-115424.spectra', '20220923-115500.spectra']]
filenames_15T_reverse = [['20220921-155128.spectra', '20220921-155233.spectra', '20220921-155308.spectra'],
                         ['20220927-111558.spectra', '20220927-111628.spectra', '20220927-111703.spectra'],
                         ['20220922-152910.spectra', '20220922-152940.spectra', '20220922-153009.spectra'],
                         ['20220927-161940.spectra', '20220927-162009.spectra', '20220927-162037.spectra'],
                         ['20220923-120435.spectra', '20220923-120508.spectra', '20220923-120541.spectra']]
filenames_0T_reverse = [['20220921-145005.spectra', '20220921-145038.spectra', '20220921-145115.spectra',
                         '20220921-160423.spectra', '20220921-160454.spectra', '20220921-160524.spectra'],
                        ['20220927-101247.spectra', '20220927-101316.spectra', '20220927-101344.spectra',
                         '20220927-112556.spectra', '20220927-112623.spectra', '20220927-112649.spectra'],
                        ['20220922-141408.spectra', '20220922-141442.spectra', '20220922-141517.spectra',
                         '20220922-154301.spectra', '20220922-154359.spectra', '20220922-154429.spectra'],
                        ['20220927-152227.spectra', '20220927-152301.spectra', '20220927-152336.spectra',
                         '20220927-163000.spectra', '20220927-163025.spectra', '20220927-163051.spectra'],
                        ['20220923-110407.spectra', '20220923-110440.spectra', '20220923-110513.spectra',
                         '20220923-121616.spectra', '20220923-121653.spectra', '20220923-121727.spectra']]
# Forward
filenames_0T_forward = [['20220922-102039.spectra', '20220922-102112.spectra', '20220922-102146.spectra',
                         '20220922-113527.spectra', '20220922-113743.spectra', '20220922-113816.spectra'],
                        ['20220927-112556.spectra', '20220927-112623.spectra', '20220927-112649.spectra',
                         '20220927-123652.spectra', '20220927-123724.spectra', '20220927-123826.spectra'],
                        ['20220922-125020.spectra', '20220922-125056.spectra', '20220922-125129.spectra',
                         '20220922-141408.spectra', '20220922-141442.spectra', '20220922-141517.spectra'],
                        ['20220927-141540.spectra', '20220927-141620.spectra', '20220927-141644.spectra',
                         '20220927-152227.spectra', '20220927-152301.spectra', '20220927-152336.spectra'],
                        ['20220923-130641.spectra', '20220923-130744.spectra', '20220923-130829.spectra',
                         '20220923-142044.spectra', '20220923-142120.spectra', '20220923-142152.spectra']]
filenames_02T_forward = [['20220922-103743.spectra', '20220922-103813.spectra', '20220922-103845.spectra'],
                         ['20220927-114308.spectra', '20220927-114340.spectra', '20220927-114410.spectra'],
                         ['20220922-130647.spectra', '20220922-130727.spectra', '20220922-130802.spectra'],
                         ['20220927-142548.spectra', '20220927-142614.spectra', '20220927-142642.spectra'],
                         ['20220923-132323.spectra', '20220923-132356.spectra', '20220923-132429.spectra']]
filenames_035T_forward = [['20220922-104559.spectra', '20220922-104633.spectra', '20220922-104718.spectra'],
                          ['20220927-115231.spectra', '20220927-115303.spectra', '20220927-115333.spectra'],
                          ['20220922-131852.spectra', '20220922-131921.spectra', '20220922-131951.spectra'],
                          ['20220927-144958.spectra', '20220927-145025.spectra', '20220927-145058.spectra'],
                          ['20220923-133320.spectra', '20220923-133353.spectra', '20220923-133431.spectra']]
filenames_05T_forward = [['20220922-110026.spectra', '20220922-110059.spectra', '20220922-110126.spectra'],
                         ['20220927-120219.spectra', '20220927-120248.spectra', '20220927-120317.spectra'],
                         ['20220922-133307.spectra', '20220922-133335.spectra', '20220922-133407.spectra'],
                         ['20220927-143806.spectra', '20220927-143837.spectra', '20220927-143837.spectra'],
                         ['20220923-134436.spectra', '20220923-134513.spectra', '20220923-134547.spectra']]
filenames_1T_forward = [['20220922-111244.spectra', '20220922-111313.spectra', '20220922-111349.spectra'],
                        ['20220927-121301.spectra', '20220927-121330.spectra', '20220927-121401.spectra'],
                        ['20220922-134506.spectra', '20220922-134433.spectra', '20220922-134506.spectra'],
                        ['20220927-150051.spectra', '20220927-150118.spectra', '20220927-150145.spectra'],
                        ['20220923-135530.spectra', '20220923-135609.spectra', '20220923-135644.spectra']]
filenames_15T_forward = [['20220922-112258.spectra', '20220922-112327.spectra', '20220922-112355.spectra'],
                         ['20220927-122403.spectra', '20220927-122436.spectra', '20220927-122505.spectra'],
                         ['20220922-140151.spectra', '20220922-140229.spectra', '20220922-140304.spectra'],
                         ['20220927-151141.spectra', '20220927-151208.spectra', '20220927-151233.spectra'],
                         ['20220923-140638.spectra', '20220923-140707.spectra', '20220923-140753.spectra']]

for s in range(len(scintillators)):
    tobeanalyse = [filenames_0T_reverse[s], filenames_02T_reverse[s], filenames_035T_reverse[s],
                   filenames_05T_reverse[s], filenames_1T_reverse[s], filenames_15T_reverse[s],
                   filenames_0T_forward[s], filenames_02T_forward[s], filenames_035T_forward[s],
                   filenames_05T_forward[s], filenames_1T_forward[s], filenames_15T_forward[s]]

    for item in tobeanalyse:
        for f in item:
            filepath = path + f
            # print(filepath)
            Resultat.append(summed_spectra_old(filepath, channel[s]))


# fig, axes = plt.subplots(nrows=2, ncols=2,  gridspec_kw={'height_ratios': [2, 1]})
n = 10 * 3 + 2 * 6
for s in range(len(scintillators)):
    # print(s * 36, (s + 1) * 36)
    Data[s] = np.array(Resultat[s * n:(s + 1) * n])

    if s == 0:  # bcf10
        scint_file = folder + '20220920-141652-scint.spectra'
        fluo_file = folder + '20220920-143043-fluo.spectra'
        ckov1_file = folder + '20220920-161025-cerenkov1.spectra'
        ckov2_file = folder + '20220920-161218-cerenkov2.spectra'
        ckov3_file = folder + '20220920-161924-cerenkov3.spectra'
        ckov4_file = folder + '20220920-162104-cerenkov4.spectra'
        dose_file = folder + '20220921-105631-normalization_all.spectra'
    elif s == 1:  # bcf60
        scint_file = folder + '20220927-085310-scint.spectra'
        fluo_file = folder + '20220927-090118-fluo.spectra'
        ckov1_file = folder + '20220927-092943-cerenkov1.spectra'
        ckov2_file = folder + '20220927-093133-cerenkov2.spectra'
        ckov3_file = folder + '20220927-093609-cerenkov3.spectra'
        ckov4_file = folder + '20220927-093753-cerenkov4.spectra'
        dose_file = folder + '20220927-095656-normalization_all.spectra'
    elif s == 2:  # ej-204
        scint_file = folder + '20220920-141652-scint.spectra'
        fluo_file = folder + '20220920-143043-fluo.spectra'
        ckov1_file = folder + '20220920-161025-cerenkov1.spectra'
        ckov2_file = folder + '20220920-161218-cerenkov2.spectra'
        ckov3_file = folder + '20220920-161924-cerenkov3.spectra'
        ckov4_file = folder + '20220920-162104-cerenkov4.spectra'
        dose_file = folder + '20220922-115954-normalization_all.spectra'
    elif s == 3:  # bcf60 lee
        scint_file = folder + '20220927-085310-scint.spectra'
        fluo_file = folder + '20220927-090118-fluo.spectra'
        ckov1_file = folder + '20220927-092943-cerenkov1.spectra'
        ckov2_file = folder + '20220927-093133-cerenkov2.spectra'
        ckov3_file = folder + '20220927-093609-cerenkov3.spectra'
        ckov4_file = folder + '20220927-093753-cerenkov4.spectra'
        dose_file = folder + '20220927-140521-normalization_all.spectra'
    else:
        scint_file = folder + '20220922-164231-scint.spectra'
        fluo_file = folder + '20220922-165214-fluo.spectra'
        ckov1_file = folder + '20220922-170040-cerenkov1.spectra'
        ckov2_file = folder + '20220922-170232-cerenkov2.spectra'
        ckov3_file = folder + '20220922-170618-cerenkov3.spectra'
        ckov4_file = folder + '20220922-170807-cerenkov4.spectra'
        dose_file = folder + '20220923-104901-normalization_all.spectra'

    scint = summed_spectra_old(scint_file, channel[s])
    fluo = summed_spectra_old(fluo_file, channel[s])
    ckov1 = summed_spectra_old(ckov1_file, channel[s])
    ckov2 = summed_spectra_old(ckov2_file, channel[s])
    ckov3 = summed_spectra_old(ckov3_file, channel[s])
    ckov4 = summed_spectra_old(ckov4_file, channel[s])
    ckovA = abs(ckov1 - ckov2)
    ckovB = abs(ckov3 - ckov4)
    dose = summed_spectra_old(dose_file, channel[s])
    calib_doseval = 500

    # To obtain the abundance

    R = compute_icm([scint, fluo, ckovA, ckovB])
    Ref = generate_weights(R, dose)
    Doses = []
    Fluo = []
    Ckov = []
    Ckov1 = []
    Ckov2 = []
    for i in range(len(Data[s])):
        Weight = generate_weights(R, Data[s][i])
        Doses.append(Weight[0] / Ref[0] * calib_doseval)
        Fluo.append(Weight[1] / Ref[1] * calib_doseval)
        Ckov.append((Weight[2] + Weight[3]) / (Ref[2] + Ref[3]) * calib_doseval)


    Moy_doses = []
    Moy_fluo = []
    Moy_ckov = []
    dev_doses = []
    dev_fluo = []
    dev_ckov = []
    Moy_spectrum = []

    for i in range(int(len(Data[s]) / 3)):

        if i == 0 or i == 7:
            Moy_doses.append(np.mean([Doses[i * 3], Doses[i * 3 + 1], Doses[i * 3 + 2], Doses[i * 3 + 3],
                                      Doses[i * 3 + 4], Doses[i * 3 + 5]]))
            Moy_fluo.append(np.mean([Fluo[i * 3], Fluo[i * 3 + 1], Fluo[i * 3 + 2], Fluo[i * 3 + 3], Fluo[i * 3 + 4],
                                     Fluo[i * 3 + 5]]))
            Moy_ckov.append(np.mean([Ckov[i * 3], Ckov[i * 3 + 1], Ckov[i * 3 + 2], Ckov[i * 3 + 3], Ckov[i * 3 + 4],
                                     Ckov[i * 3 + 5]]))
            dev_doses.append(np.std([Doses[i * 3], Doses[i * 3 + 1], Doses[i * 3 + 2], Doses[i * 3 + 3],
                                      Doses[i * 3 + 4], Doses[i * 3 + 5]]))
            dev_fluo.append(np.std([Fluo[i * 3], Fluo[i * 3 + 1], Fluo[i * 3 + 2], Fluo[i * 3 + 3], Fluo[i * 3 + 4],
                                     Fluo[i * 3 + 5]]))
            dev_ckov.append(np.std([Ckov[i * 3], Ckov[i * 3 + 1], Ckov[i * 3 + 2], Ckov[i * 3 + 3], Ckov[i * 3 + 4],
                                     Ckov[i * 3 + 5]]))
            Moy_spectrum.append(np.mean(Data[s][i * 3:i * 3 + 5], axis=0))



            # print(i, Doses[i * 3], Doses[i * 3 + 1], Doses[i * 3 + 2], Doses[i * 3 + 3],
            #                          Doses[i * 3 + 4], Doses[i * 3 + 5], Moy_doses,  '\n')
        elif i == 1 or i == 8:
            continue
        else:
            Moy_doses.append(np.mean([Doses[i * 3], Doses[i * 3 + 1], Doses[i * 3 + 2]]))
            Moy_fluo.append(np.mean([Fluo[i * 3], Fluo[i * 3 + 1], Fluo[i * 3 + 2]]))
            Moy_ckov.append(np.mean([Ckov[i * 3], Ckov[i * 3 + 1], Ckov[i * 3 + 2]]))
            dev_doses.append(np.std([Doses[i * 3], Doses[i * 3 + 1], Doses[i * 3 + 2]]))
            dev_fluo.append(np.std([Fluo[i * 3], Fluo[i * 3 + 1], Fluo[i * 3 + 2]]))
            dev_ckov.append(np.std([Ckov[i * 3], Ckov[i * 3 + 1], Ckov[i * 3 + 2]]))
            Moy_spectrum.append(np.mean(Data[s][i * 3:i * 3 + 2], axis=0))

    Mean_dose[s] = Moy_doses
    Mean_fluo[s] = Moy_fluo
    Mean_ckov[s] = Moy_ckov
    std_dose[s] = dev_doses
    std_fluo[s] = dev_fluo
    std_ckov[s] = dev_ckov
    Mean_spectrum[s] = Moy_spectrum

    # Absolute spectra
    if s == 2:

        plt.plot(scint/sum(scint), label='Scintillation')
        plt.plot(ckovA/sum(ckovA), label='Cherenkov')
        plt.plot(fluo/sum(fluo), label='Fluorescence')
        plt.ylabel('Normalized intensity (a.u.)')
        plt.xlabel('Wavelength (a.u.)')
        plt.legend()
        plt.tight_layout()
        plt.savefig("spectrum_composition.eps")
        plt.show()

#print(std_dose/Mean_dose *100)



# Figures for MRinRT

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7), gridspec_kw={'height_ratios': [2, 1]})
for s in range(len(scintillators) - 2):
    plt.subplot(211)
    plt.errorbar(bfields, Mean_fluo[s][6:12] / Mean_fluo[s][6],
                 yerr=np.sqrt((std_fluo[s][6:12] / Mean_fluo[s][6:12]) ** 2 + (std_fluo[s][6] / Mean_fluo[s][6]) ** 2) *
                      (Mean_fluo[s][6:12] / Mean_fluo[s][6]), markerfacecolor=colors[s], fmt=symbol[s], label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_fluo[s][0:6] / Mean_fluo[s][0],
                 yerr=np.sqrt((std_fluo[s][0:6] / Mean_fluo[s][0:6]) ** 2 + (std_fluo[s][0] / Mean_fluo[s][0]) ** 2) * (
                      Mean_fluo[s][0:6] / Mean_fluo[s][0]),  markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle=':',
                 label=scint_label[s] + ', e- \u2192 tip')
    #plt.xlabel('Magnetic field [T]')
    plt.ylabel('fluo / fluo [0 T]')
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    #plt.xlabel('Magnetic field [T]')

    plt.subplot(212)
    plt.plot(bfields, 100*((Mean_fluo[s][6:12] / Mean_fluo[s][6]) - (Mean_fluo[s][0:6] / Mean_fluo[s][0])) /
            (Mean_fluo[s][0:6] / Mean_fluo[s][0]), color=colors[s], marker=symbol[s], label=scint_label[s])
    plt.ylabel('Relative difference \n in orientation [%]')

   # plt.ylim([0, 1])
    plt.xlabel('Magnetic field [T]')

    plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
for s in range(len(scintillators) - 2):
    plt.subplot(211)
    plt.errorbar(bfields, Mean_ckov[s][6:12] / Mean_ckov[s][6],
                 yerr=np.sqrt((std_ckov[s][6:12] / Mean_ckov[s][6:12]) ** 2 + (std_ckov[s][6] / Mean_ckov[s][6]) ** 2) *
                      (Mean_ckov[s][6:12] / Mean_ckov[s][6]),  color=colors[s], fmt=symbol[s], linestyle='-', label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_ckov[s][0:6] / Mean_ckov[s][0],
                 yerr=np.sqrt((std_ckov[s][0:6] / Mean_ckov[s][0:6]) ** 2 + (std_ckov[s][0] / Mean_ckov[s][0]) ** 2) * (
                      Mean_ckov[s][0:6] / Mean_ckov[s][0]),  markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle='--',
                 label=scint_label[s] + ', e- \u2192 tip')
    #plt.xlabel('Magnetic field [T]')
    plt.ylabel('Cerenkov / Cerenkov [0 T]')
    plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    #plt.xlabel('Magnetic field [T]')

    plt.subplot(212)
    plt.plot(bfields, 100*((Mean_ckov[s][6:12] / Mean_ckov[s][6]) - (Mean_ckov[s][0:6] / Mean_ckov[s][0])) /
            (Mean_ckov[s][0:6] / Mean_ckov[s][0]), color=colors[s], marker=symbol[s], label=scint_label[s])
    plt.ylabel('Relative difference \n in orientation [%]')

   # plt.ylim([0, 1])
    plt.xlabel('Magnetic field [T]')

    plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
plt.tight_layout()
plt.savefig("ckov.eps")
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
for s in range(len(scintillators) - 2):
    plt.subplot(211)
    plt.errorbar(bfields, Mean_dose[s][6:12] / Mean_dose[s][6],
                 yerr=np.sqrt((std_dose[s][6:12] / Mean_dose[s][6:12]) ** 2 + (std_dose[s][6] / Mean_dose[s][6]) ** 2) *
                      (Mean_dose[s][6:12] / Mean_dose[s][6]), color=colors[s], fmt=symbol[s], linestyle='-',
                 label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_dose[s][0:6] / Mean_dose[s][0],
                 yerr=np.sqrt((std_dose[s][0:6] / Mean_dose[s][0:6]) ** 2 + (std_dose[s][0] / Mean_dose[s][0]) ** 2) * (
                      Mean_dose[s][0:6] / Mean_dose[s][0]), markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle=':',
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Dose / dose [0 T]')
    plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    #plt.xlabel('Magnetic field [T]')

    plt.subplot(212)
    plt.plot(bfields, 100*((Mean_dose[s][6:12] / Mean_dose[s][6]) - (Mean_dose[s][0:6] / Mean_dose[s][0])) /
             (Mean_dose[s][0:6] / Mean_dose[s][0]), color=colors[s], marker=symbol[s], label=scint_label[s])
    plt.ylabel('Relative difference \n in orientation [%]')

    plt.ylim([0, 1])
    plt.xlabel('Magnetic field [T]')

    plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
plt.tight_layout()
plt.savefig("scint.eps")
plt.show()

# Absolute spectra
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), sharey=True)
plt.subplot(132)
s == 0
plt.plot(Mean_spectrum[0][0], label=str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[0][5],  ls='--',label=str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[0][10], label=str(bfields[5]) + ' T' + ', e- \u2192 stem')

plt.xlabel('Wavelength (a.u.)')
plt.ylim([0, 75000])
plt.title(scint_label[0])
plt.legend()
plt.subplot(133)
s == 1
plt.plot(Mean_spectrum[1][0], label=str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[1][5],  ls='--',label=str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[1][10], label=str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.xlabel('Wavelength (a.u.)')

plt.ylim([0, 75000])
plt.title(scint_label[1])
plt.legend()
plt.subplot(131)
s == 2
plt.plot(Mean_spectrum[2][0], label=str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[2][5],  ls='--',label=str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[2][10],  label=str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.xlabel('Wavelength (a.u.)')
plt.ylabel('Intensity (a.u.)' )
plt.ylim([0, 75000])
plt.title(scint_label[2])
plt.legend()
plt.tight_layout()
plt.savefig("spectrum.eps")
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
plt.subplot(132)
s == 0
plt.plot(Mean_spectrum[0][0]/sum(Mean_spectrum[0][0]), label=str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[0][5]/sum(Mean_spectrum[0][5]),  ls='--', label=str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[0][10]/sum(Mean_spectrum[0][10]), label=str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.ylabel('Intensity')
plt.xlabel('Arbitrary units')
plt.ylim([0, 0.026])
plt.title(scint_label[0])
plt.legend()
plt.subplot(133)
s == 1
plt.plot(Mean_spectrum[1][0]/sum(Mean_spectrum[1][0]), label=str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[1][5]/sum(Mean_spectrum[1][5]),  ls='--', label=str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[1][10]/sum(Mean_spectrum[1][10]), label=str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.xlabel('Arbitrary units')
plt.ylim([0, 0.026])
plt.title(scint_label[1])
plt.legend()
plt.subplot(131)
s == 2
plt.plot(Mean_spectrum[2][0]/sum(Mean_spectrum[2][0]), label=str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[2][5]/sum(Mean_spectrum[2][5]),  ls='--', label=str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[2][10]/sum(Mean_spectrum[2][10]),  label=str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.xlabel('Arbitrary units')
plt.ylim([0, 0.026])
plt.title(scint_label[2])
plt.legend()
plt.tight_layout()
plt.savefig("norm_spectrum.eps")
plt.show()

# ---------


# Spectra with respect to the area under the curve of 0 T
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))

for s in range(len(scintillators)-2):
    temp0 = 230
    temp1 = temp0 + s + 1
    temp2 = temp0 + s + 4
    for b in range(len(bfields)):
        plt.subplot(temp1)
        plt.plot(Mean_spectrum[s][b]/sum(Mean_spectrum[s][0]), label = str(bfields[b]) + ' T')
        if temp1 == 231:
            plt.ylabel('Normalized intensity to \n area under the curve [0 T]')
        plt.ylim([0, 0.026])
        plt.title(scint_label[s]+', e- \u2192 tip')
        plt.legend()
        plt.subplot(temp2)
        plt.plot(Mean_spectrum[s][b+6] / sum(Mean_spectrum[s][6]), label= str(bfields[b]) + ' T')
        plt.xlabel('Arbitrary units')
        if temp2 == 234:
            plt.ylabel('Normalized Intensity to \n area under the curve [0 T]')
        plt.ylim([0, 0.026])
        plt.title(scint_label[s] + ', e- \u2192 stem')
        plt.legend()
plt.tight_layout()
plt.show()
# Spectra with respect to the area under the curve
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
for s in range(len(scintillators)-2):
    temp0 = 230
    temp1 = temp0 + s + 1
    temp2 = temp0 + s + 4
    for b in range(len(bfields)):
        plt.subplot(temp1)
        plt.plot(Mean_spectrum[s][b]/sum(Mean_spectrum[s][b]), label = str(bfields[b]) + ' T')
        if temp1 == 231:
            plt.ylabel('Normalized intensity to \n area under the curve')
        plt.ylim([0, 0.026])
        plt.title(scint_label[s]+', e- \u2192 tip')
        plt.legend()
        plt.subplot(temp2)
        plt.plot(Mean_spectrum[s][b+6] / sum(Mean_spectrum[s][b+6]), label= str(bfields[b]) + ' T')
        plt.xlabel('')
        plt.xlabel('')
        if temp2 == 234:
            plt.ylabel('Normalized Intensity to \n area under the curve')
        plt.ylim([0, 0.026])
        plt.title(scint_label[s] + ', e- \u2192 stem')
        plt.legend()
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
for s in range(len(scintillators)-2):
    temp0 = 230
    temp1 = temp0 + s + 1
    temp2 = temp0 + s + 4
    for b in range(len(bfields)):
        plt.subplot(temp1)
        plt.plot(Mean_spectrum[s][b], label = str(bfields[b]) + ' T')
        if temp1 == 231:
            plt.ylabel('Intensity')
        #plt.ylim([0, 0.026])
        plt.title(scint_label[s]+', e- \u2192 tip')
        plt.legend()
        plt.subplot(temp2)
        plt.plot(Mean_spectrum[s][b+6] , label= str(bfields[b]) + ' T')
        plt.xlabel('')
        plt.xlabel('')
        if temp2 == 234:
            plt.ylabel('Intensity')

        #plt.ylim([0, 0.026])
        plt.title(scint_label[s] + ', e- \u2192 stem')
        plt.legend()
plt.tight_layout()
plt.show()

#
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 6), gridspec_kw={'height_ratios': [2, 1]})
for s in range(len(scintillators) - 2):
    plt.subplot(221)
    plt.errorbar(bfields, Mean_dose[s][6:12] / Mean_dose[s][6],
                 yerr=np.sqrt((std_dose[s][6:12] / Mean_dose[s][6:12]) ** 2 + (std_dose[s][6] / Mean_dose[s][6]) ** 2) *
                      (Mean_dose[s][6:12] / Mean_dose[s][6]), color=colors[s],
                 label=scintillators[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_dose[s][0:6] / Mean_dose[s][0],
                 yerr=np.sqrt((std_dose[s][0:6] / Mean_dose[s][0:6]) ** 2 + (std_dose[s][0] / Mean_dose[s][0]) ** 2) * (
                      Mean_dose[s][0:6] / Mean_dose[s][0]), fmt='--', color=colors[s],
                 label=scintillators[s] + ', e- \u2192 tip')
    plt.ylabel('Scint / scint [0 T]')
    plt.xlabel('Magnetic field [T]')

    plt.subplot(223)
    plt.plot(bfields, 100*((Mean_dose[s][6:12] / Mean_dose[s][6]) - (Mean_dose[s][0:6] / Mean_dose[s][0])) /
             (Mean_dose[s][0:6] / Mean_dose[s][0]), color=colors[s], label=scintillators[s])
    plt.ylabel('Relative difference \n between orientations [%]')
    plt.xlabel('Magnetic field [T]')

    plt.subplot(222)
    plt.errorbar(bfields, Mean_dose[s][6:12], yerr=std_dose[s][6:12] , color=colors[s],
                 label=scintillators[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_dose[s][0:6], yerr=std_dose[s][0:6], fmt='--', color=colors[s],
                 label=scintillators[s] + ', e- \u2192 tip')
    plt.ylabel('Scint ')
    plt.xlabel('Magnetic field [T]')

    plt.subplot(224)
    plt.plot(bfields, 100*(Mean_dose[s][6:12] - Mean_dose[s][0:6]) / Mean_dose[s][0:6], color=colors[s],
             label=scintillators[s])
    plt.ylabel('Relative difference \n between orientations [%]')
    plt.ylim([0, 0.01])
    plt.xlabel('Magnetic field [T]')

plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 6), gridspec_kw={'height_ratios': [2, 1]})
for s in range(len(scintillators) - 2):
    plt.subplot(221)
    plt.errorbar(bfields, Mean_dose[s][6:12] / Mean_dose[s][6],
                 yerr=np.sqrt((std_dose[s][6:12] / Mean_dose[s][6:12]) ** 2 + (std_dose[s][6] / Mean_dose[s][6]) ** 2) *
                      (Mean_dose[s][6:12] / Mean_dose[s][6]), color=colors[s],
                 label=scintillators[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_dose[s][0:6] / Mean_dose[s][0],
                 yerr=np.sqrt((std_dose[s][0:6] / Mean_dose[s][0:6]) ** 2 + (std_dose[s][0] / Mean_dose[s][0]) ** 2) * (
                      Mean_dose[s][0:6] / Mean_dose[s][0]), fmt='--', color=colors[s],
                 label=scintillators[s] + ', e- \u2192 tip')
    plt.ylabel('Scint / scint [0 T]')
    plt.xlabel('Magnetic field [T]')

    plt.subplot(222)
    plt.errorbar(bfields, Mean_ckov[s][6:12] / Mean_ckov[s][6],
                 yerr=np.sqrt((std_ckov[s][6:12] / Mean_ckov[s][6:12]) ** 2 + (std_ckov[s][6] / Mean_ckov[s][6]) ** 2) *
                      (Mean_ckov[s][6:12] / Mean_ckov[s][6]), color=colors[s], label=scintillators[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_ckov[s][0:6] / Mean_ckov[s][0],
                 yerr=np.sqrt((std_ckov[s][0:6] / Mean_ckov[s][0:6]) ** 2 + (std_ckov[s][0] / Mean_ckov[s][0]) ** 2) * (
                      Mean_ckov[s][0:6] / Mean_ckov[s][0]), fmt='--', color=colors[s],
                 label=scintillators[s] + ', e- \u2192 tip')
    plt.xlabel('Magnetic field [T]')
    plt.ylabel('Cerenkov / Cerenkov [0 T]')
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

    plt.subplot(223)
    plt.plot(bfields, ((Mean_dose[s][6:12] / Mean_dose[s][6]) - (Mean_dose[s][0:6] / Mean_dose[s][0])) /
             (Mean_dose[s][0:6] / Mean_dose[s][0]), color=colors[s], label=scintillators[s])
    plt.ylabel('Relative difference \n between orientations')
    plt.xlabel('Magnetic field [T]')
    plt.ylim([0, 0.01])

    plt.subplot(224)
    plt.plot(bfields, ((Mean_ckov[s][6:12] / Mean_ckov[s][6]) - (Mean_ckov[s][0:6] / Mean_ckov[s][0])) /
             (Mean_ckov[s][0:6] / Mean_ckov[s][0]), color=colors[s], label=scintillators[s])
    plt.xlabel('Magnetic field [T]')

plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.show()


for s in range(len(scintillators) - 2):
    plt.subplot(221)
    plt.errorbar(bfields, Mean_ckov[s][6:12] / Mean_ckov[s][6],
                 yerr=np.sqrt((std_ckov[s][6:12] / Mean_ckov[s][6:12]) ** 2 + (std_ckov[s][6] / Mean_ckov[s][6]) ** 2) *
                      (Mean_ckov[s][6:12] / Mean_ckov[s][6]), color=colors[s], label=scintillators[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_ckov[s][0:6] / Mean_ckov[s][0],
                 yerr=np.sqrt((std_ckov[s][0:6] / Mean_ckov[s][0:6]) ** 2 + (std_ckov[s][0] / Mean_ckov[s][0]) ** 2) * (
                      Mean_ckov[s][0:6] / Mean_ckov[s][0]), fmt='--', color=colors[s],
                 label=scintillators[s] + ', e- \u2192 tip')
    plt.ylabel('Scint / scint [0 T]')
    plt.xlabel('Magnetic field [T]')

    plt.subplot(222)
    plt.errorbar(bfields, Mean_ckov[s][6:12], yerr=std_ckov[s][6:12], color=colors[s], label=scintillators[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_ckov[s][0:6], yerr=(std_ckov[s][0:6]), fmt='--', color=colors[s],
                 label=scintillators[s] + ', e- \u2192 tip')
    plt.xlabel('Magnetic field [T]')
    plt.ylabel('Cerenkov / Cerenkov [0 T]')
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

    plt.subplot(223)
    plt.plot(bfields, ((Mean_ckov[s][6:12] / Mean_ckov[s][6]) - (Mean_ckov[s][0:6] / Mean_ckov[s][0])) /
             (Mean_ckov[s][0:6] / Mean_ckov[s][0]), color=colors[s], label=scintillators[s])
    plt.ylabel('Relative difference \n between orientations')
    plt.xlabel('Magnetic field [T]')
    # plt.ylim([0, 0.01])

    plt.subplot(224)
    plt.plot(bfields, 100*(Mean_ckov[s][6:12]  - (Mean_ckov[s][0:6] ) /
             (Mean_ckov[s][0:6] )), color=colors[s], label=scintillators[s])
    plt.xlabel('Magnetic field [T]')

plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.show()

# for bcf60 and bcf 60 Lee

# Spectra
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
filter = [1, 3]
temp0 = 220
for s in filter:

    temp1 = temp0 + 1
    temp2 = temp0 + 3

    for b in range(len(bfields)):
        plt.subplot(temp1)
        plt.plot(Mean_spectrum[s][b]/sum(Mean_spectrum[s][0]), label = str(bfields[b]) + ' T')
        if temp1 == 221:
            plt.ylabel('Normalized intensity to \n area under the curve [0 T] ')
        plt.ylim([0, 0.03])
        plt.title(scint_label[s]+', e- \u2192 tip')
        plt.legend()
        plt.subplot(temp2)
        plt.plot(Mean_spectrum[s][b+6] / sum(Mean_spectrum[s][6]), label= str(bfields[b]) + ' T')
        plt.xlabel('Arbitrary units')
        if temp2 == 223:
            plt.ylabel('Normalized Intensity to \n area under the curve [0 T] ')
        plt.ylim([0, 0.03])
        plt.title(scint_label[s] + ', e- \u2192 stem')
        plt.legend()
    temp0 = temp0 + 1
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 6), gridspec_kw={'height_ratios': [2, 1]})
for s in filter:
    plt.subplot(221)
    plt.errorbar(bfields, Mean_dose[s][6:12] / Mean_dose[s][6],
                 yerr=np.sqrt((std_dose[s][6:12] / Mean_dose[s][6:12]) ** 2 + (std_dose[s][6] / Mean_dose[s][6]) ** 2) *
                      (Mean_dose[s][6:12] / Mean_dose[s][6]), color=colors[s],
                 label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_dose[s][0:6] / Mean_dose[s][0],
                 yerr=np.sqrt((std_dose[s][0:6] / Mean_dose[s][0:6]) ** 2 + (std_dose[s][0] / Mean_dose[s][0]) ** 2) * (
                      Mean_dose[s][0:6] / Mean_dose[s][0]), fmt='--', color=colors[s],
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Scint / scint [0 T]')
    plt.xlabel('Magnetic field [T]')

    plt.subplot(222)
    plt.errorbar(bfields, Mean_ckov[s][6:12] / Mean_ckov[s][6],
                 yerr=np.sqrt((std_ckov[s][6:12] / Mean_ckov[s][6:12]) ** 2 + (std_ckov[s][6] / Mean_ckov[s][6]) ** 2) *
                      (Mean_ckov[s][6:12] / Mean_ckov[s][6]), color=colors[s], label=scintillators[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_ckov[s][0:6] / Mean_ckov[s][0],
                 yerr=np.sqrt((std_ckov[s][0:6] / Mean_ckov[s][0:6]) ** 2 + (std_ckov[s][0] / Mean_ckov[s][0]) ** 2) * (
                      Mean_ckov[s][0:6] / Mean_ckov[s][0]), fmt='--', color=colors[s],
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.xlabel('Magnetic field [T]')
    plt.ylabel('Cerenkov / Cerenkov [0 T]')
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

    plt.subplot(223)
    plt.plot(bfields, ((Mean_dose[s][6:12] / Mean_dose[s][6]) - (Mean_dose[s][0:6] / Mean_dose[s][0])) /
             (Mean_dose[s][0:6] / Mean_dose[s][0]), color=colors[s], label=scintillators[s])
    plt.ylabel('Relative difference \n between orientations')
    plt.xlabel('Magnetic field [T]')
   # plt.ylim([0, 0.01])

    plt.subplot(224)
    plt.plot(bfields, ((Mean_ckov[s][6:12] / Mean_ckov[s][6]) - (Mean_ckov[s][0:6] / Mean_ckov[s][0])) /
             (Mean_ckov[s][0:6] / Mean_ckov[s][0]), color=colors[s], label=scintillators[s])
    plt.xlabel('Magnetic field [T]')

plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.show()
''''
    plt.subplot(133)

    plt.errorbar(bfields, Mean_fluo[s][6:12] / Mean_fluo[s][6],
                 yerr=np.sqrt((std_fluo[s][6:12] / Mean_fluo[s][6:12]) ** 2 + (std_fluo[s][6] / Mean_fluo[s][6]) ** 2) *
                      (Mean_fluo[s][6:12] / Mean_fluo[s][6]), color=colors[s],
                 label=scintillators[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_fluo[s][0:6] / Mean_fluo[s][0],
                 yerr=np.sqrt((std_fluo[s][0:6] / Mean_fluo[s][0:6]) ** 2 + (std_fluo[s][0] / Mean_fluo[s][0]) ** 2) *
                      (Mean_fluo[s][0:6] / Mean_fluo[s][0]), fmt='--', color=colors[s],
                 label=scintillators[s] + ', e- \u2192 tip')
    plt.xlabel('Magnetic field [T]')
    plt.ylabel('Fluo  / Fluo [0 T]')
    '''



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 6), gridspec_kw={'height_ratios': [2, 1]})
for s in range(len(scintillators)):
    plt.subplot(221)
    plt.errorbar(bfields, Mean_dose[s][6:12] / Mean_dose[s][6],
                 yerr=np.sqrt((std_dose[s][6:12] / Mean_dose[s][6:12]) ** 2 + (std_dose[s][6] / Mean_dose[s][6]) ** 2) *
                      (Mean_dose[s][6:12] / Mean_dose[s][6]), color=colors[s],
                 label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_dose[s][0:6] / Mean_dose[s][0],
                 yerr=np.sqrt((std_dose[s][0:6] / Mean_dose[s][0:6]) ** 2 + (std_dose[s][0] / Mean_dose[s][0]) ** 2) * (
                      Mean_dose[s][0:6] / Mean_dose[s][0]), fmt='--', color=colors[s],
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Scint / scint [0 T]')
    plt.xlabel('Magnetic field [T]')

    plt.subplot(222)
    plt.errorbar(bfields, Mean_ckov[s][6:12] / Mean_ckov[s][6],
                 yerr=np.sqrt((std_ckov[s][6:12] / Mean_ckov[s][6:12]) ** 2 + (std_ckov[s][6] / Mean_ckov[s][6]) ** 2) *
                      (Mean_ckov[s][6:12] / Mean_ckov[s][6]), color=colors[s], label=scintillators[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_ckov[s][0:6] / Mean_ckov[s][0],
                 yerr=np.sqrt((std_ckov[s][0:6] / Mean_ckov[s][0:6]) ** 2 + (std_ckov[s][0] / Mean_ckov[s][0]) ** 2) * (
                      Mean_ckov[s][0:6] / Mean_ckov[s][0]), fmt='--', color=colors[s],
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.xlabel('Magnetic field [T]')
    plt.ylabel('Cerenkov / Cerenkov [0 T]')
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

    plt.subplot(223)
    plt.plot(bfields, ((Mean_dose[s][6:12] / Mean_dose[s][6]) - (Mean_dose[s][0:6] / Mean_dose[s][0])) /
             (Mean_dose[s][0:6] / Mean_dose[s][0]), color=colors[s], label=scintillators[s])
    plt.ylabel('Relative difference \n between orientations')
    plt.xlabel('Magnetic field [T]')
   # plt.ylim([0, 0.01])

    plt.subplot(224)
    plt.plot(bfields, ((Mean_ckov[s][6:12] / Mean_ckov[s][6]) - (Mean_ckov[s][0:6] / Mean_ckov[s][0])) /
             (Mean_ckov[s][0:6] / Mean_ckov[s][0]), color=colors[s], label=scintillators[s])
    plt.xlabel('Magnetic field [T]')

plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.show()

