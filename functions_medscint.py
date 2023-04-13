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