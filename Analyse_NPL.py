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


# ------------Analyse-------------------#
bfields = [0, 0.2, 0.35, 0.5, 1, 1.5]

colors = ['tomato', 'deepskyblue', 'mediumseagreen', 'purple', 'grey', 'darkorange', 'plum', 'black']
lin = ['solid', 'dashed']

scintillators = ['Medscint']
Mesures_medscint_reverse = []
Mesures_medscint_forward = []

path = '/Users/yun/Library/CloudStorage/OneDrive-Medscintinc/SpectraOutput/'
folder = os.path.expanduser(path)
files = os.listdir(path)

# The abundances for Medscint probe
filenames_0T_reverse = ['20220923-110407.spectra', '20220923-110440.spectra', '20220923-110513.spectra']
filenames_02T_reverse = ['20220923-111908.spectra', '20220923-111941.spectra', '20220923-112016.spectra']
filenames_035T_reverse = ['20220923-113218.spectra', '20220923-113249.spectra', '20220923-113320.spectra']
filenames_05T_reverse = ['20220923-114238.spectra', '20220923-114333.spectra', '20220923-114414.spectra']
filenames_1T_reverse = ['20220923-115351.spectra', '20220923-115424.spectra', '20220923-115500.spectra']
filenames_15T_reverse = ['20220923-120435.spectra', '20220923-120508.spectra', '20220923-120541.spectra']

# Forward
filenames_0T_forward = ['20220923-130641.spectra', '20220923-130744.spectra', '20220923-130829.spectra']
filenames_02T_forward = ['20220923-132323.spectra', '20220923-132356.spectra', '20220923-132429.spectra']
filenames_035T_forward = ['20220923-133320.spectra', '20220923-133353.spectra', '20220923-133431.spectra']
filenames_05T_forward = ['20220923-134436.spectra', '20220923-134513.spectra', '20220923-134547.spectra']
filenames_1T_forward = ['20220923-135530.spectra', '20220923-135609.spectra', '20220923-135644.spectra']
filenames_15T_forward = ['20220923-140638.spectra', '20220923-140707.spectra', '20220923-140753.spectra']

tobeanalyse = [filenames_0T_reverse, filenames_02T_reverse, filenames_035T_reverse, filenames_05T_reverse,
               filenames_1T_reverse, filenames_15T_reverse, filenames_0T_forward, filenames_02T_forward,
               filenames_035T_forward, filenames_05T_forward, filenames_1T_forward, filenames_15T_forward]

Resultat = []
for item in tobeanalyse:
    for f in item:
        filepath = path + f
        Resultat.append(summed_spectra_old(filepath, 1))

scint_file = folder + '20220922-164231-scint.spectra'
fluo_file = folder + '20220922-165214-fluo.spectra'
ckov1_file = folder + '20220922-170040-cerenkov1.spectra'
ckov2_file = folder + '20220922-170232-cerenkov2.spectra'
ckov3_file = folder + '20220922-170618-cerenkov3.spectra'
ckov4_file = folder + '20220922-170807-cerenkov4.spectra'
dose_file = folder + '20220923-104901-normalization_all.spectra'  # 104725 or 104901-normalization_all.spectra'

scint = summed_spectra_old(scint_file, 1)
fluo = summed_spectra_old(fluo_file, 1)
ckov1 = summed_spectra_old(ckov1_file, 1)
ckov2 = summed_spectra_old(ckov2_file, 1)
ckov3 = summed_spectra_old(ckov3_file, 1)
ckov4 = summed_spectra_old(ckov4_file, 1)
ckovA = abs(ckov1 - ckov2)
ckovB = abs(ckov3 - ckov4)
dose = summed_spectra_old(dose_file, 1)
calib_doseval = 500

# To obtain the abundance

R = compute_icm([scint, fluo, ckovA, ckovB])
Ref = generate_weights(R, dose)
Doses = []
Fluo = []
Ckov = []
for i in range(len(Resultat)):
    Weight = generate_weights(R, Resultat[i])
    Doses.append(Weight[0] / Ref[0] * calib_doseval)
    Fluo.append(Weight[1] / Ref[1] * calib_doseval)
    Ckov.append((Weight[2] + Weight[3]) / (Ref[2] + Ref[3]) * calib_doseval)

Moy_doses = []
Moy_fluo = []
Moy_ckov = []
for i in range(int(len(Resultat) / 3)):
    Moy_doses.append(np.mean([Doses[i * 3], Doses[i * 3 + 1], Doses[i * 3 + 2]]))
    Moy_fluo.append(np.mean([Fluo[i * 3], Fluo[i * 3 + 1], Fluo[i * 3 + 2]]))
    Moy_ckov.append(np.mean([Ckov[i * 3], Ckov[i * 3 + 1], Ckov[i * 3 + 2]]))

plt.subplot(131)
plt.plot(bfields, Moy_doses[0:6] / Moy_doses[0], label='e- \u2192 tip')
plt.plot(bfields, Moy_doses[6:12] / Moy_doses[6], label='e- \u2192 stem')
plt.ylabel('Scint / scint [0 T]')
plt.xlabel('Magnetic field [T]')
plt.subplot(132)
plt.plot(bfields, Moy_fluo[0:6] / Moy_fluo[0], label='e- \u2192 tip')
plt.plot(bfields, Moy_fluo[6:12] / Moy_fluo[6], label='e- \u2192 stem')
plt.title('Medscint probe')
plt.xlabel('Magnetic field [T]')
plt.ylabel('Fluo / fluo [0 T]')
plt.subplot(133)
plt.plot(bfields, Moy_ckov[0:6] / Moy_ckov[0], label='e- \u2192 tip')
plt.plot(bfields, Moy_ckov[6:12] / Moy_ckov[6], label='e- \u2192 stem')
plt.xlabel('Magnetic field [T]')
plt.ylabel('Cerenkov / Cerenkov [0 T]')

plt.legend(framealpha=1, frameon=True)
plt.tight_layout()
plt.show()
