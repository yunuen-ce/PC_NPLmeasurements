from numpy import linalg, ndarray, vstack
from base64 import decodebytes as b64_decode
from json import loads
import numpy as np
from xarray import open_dataarray



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

def calculate_weigths(cal_scint, cal_fluo, cal_ckovA, cal_ckovB, cal_dose_file, Data, calib_doseval=500):
    Weights = np.zeros([5, 42, 4])
    Mean_dose = np.zeros([5, 12])
    Mean_fluo = np.zeros([5, 12])
    Mean_ckov = np.zeros([5, 12])
    std_dose = np.zeros([5, 12])
    std_fluo = np.zeros([5, 12])
    std_ckov = np.zeros([5, 12])
    std_weight = np.zeros([5, 12, 4])
    Mean_spectrum = np.zeros([5, 12, 137])
    Mean_weight = np.zeros([5, 12, 4])

    for s in range(5):
        # Normalized calibration files
        scint = cal_scint[s]/sum(cal_scint[s])
        fluo = cal_fluo[s]/sum(cal_fluo[s])
        ckovA = cal_ckovA[s]/sum(cal_ckovA[s])
        ckovB = cal_ckovB[s]/sum(cal_ckovB[s])
        dose = cal_dose_file[s]

        # To obtain the abundance
        R = compute_icm([scint, fluo, ckovA, ckovB])
        Ref = generate_weights(R, dose)

        Doses = []
        Fluo = []
        Ckov = []

        for i in range(len(Data[s])):
            Weight = generate_weights(R, Data[s][i])
            Doses.append(Weight[0]  / Ref[0] * calib_doseval)
            Fluo.append(Weight[1] ) #/ Ref[1] * calib_doseval)
            Ckov.append(((Weight[2] + Weight[3])) ) # / (Ref[2] + Ref[3])) * calib_doseval)
            Weights[s, i, :] = Weight

        Moy_doses = []
        Moy_fluo = []
        Moy_ckov = []
        dev_doses = []
        dev_fluo = []
        dev_ckov = []
        Moy_weights = []
        dev_weights = []
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
                Moy_weights.append(np.mean(Weights[s][i * 3:i * 3 + 5], axis=0))
                dev_weights.append(np.std(Weights[s][i * 3:i * 3 + 5], axis=0))

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
                Moy_weights.append(np.mean(Weights[s][i * 3:i * 3 + 2], axis=0))
                dev_weights.append(np.std(Weights[s][i * 3:i * 3 + 2], axis=0))

        Mean_dose[s] = Moy_doses
        Mean_fluo[s] = Moy_fluo
        Mean_ckov[s] = Moy_ckov
        std_dose[s] = dev_doses
        std_fluo[s] = dev_fluo
        std_ckov[s] = dev_ckov

        Mean_spectrum[s] = Moy_spectrum
        Mean_weight[s] = Moy_weights
        std_weight[s] = dev_weights

    return  Mean_dose, Mean_fluo, Mean_ckov,  std_dose, std_fluo, std_ckov, Mean_spectrum, Mean_weight, std_weight

