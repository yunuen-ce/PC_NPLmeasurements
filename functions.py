import numpy as np
import xarray as xr
from numpy import linalg, ndarray, vstack
from base64 import decodebytes as b64_decode
from json import loads
import glob
from xarray import open_dataarray
import os


def generate_x_y_diff_of_bfield(data_frame,
                                detector: str,
                                orientation: str,
                                current: str,
                                field_size: float, l: str):
    data_frame_detector = data_frame[(data_frame['Detector'] == detector) &
                                     (data_frame['Orientation'] == orientation) &
                                     (data_frame['Current'] == current) &
                                     (data_frame['Field_size'] == field_size)]
    x = data_frame_detector.Bfield
    y = data_frame_detector.Diff_0T

    legend = generate_legend(detector, orientation, current, field_size, 'cm', l)
    return x, y, legend


def generate_x_y_of(data_frame,
                    detector: str,
                    orientation: str,
                    current: str,
                    bfield: float, leg: str):
    data_frame_detector = data_frame[(data_frame['Detector'] == detector) &
                                     (data_frame['Orientation'] == orientation) &
                                     (data_frame['Current'] == current) &
                                     (data_frame['Bfield'] == bfield)]
    x = data_frame_detector.Field_size
    y = data_frame_detector.OF_Diff_0T

    legend = generate_legend(detector, orientation, current, bfield, 'T', leg)
    return x, y, legend


def generate_x_y_dose_bfield(data_frame,
                             detector: str,
                             orientation: str,
                             current: str,
                             field_size: float, l: str):
    dose = np.zeros((3, 6))
    data_frame_detector = data_frame[(data_frame['Detector'] == detector) &
                                     (data_frame['Orientation'] == orientation) &
                                     (data_frame['Current'] == current) &
                                     (data_frame['Field_size'] == field_size)]
    dose[0] = data_frame_detector.Bfield
    dose[1] = data_frame_detector.Dose_Avg
    dose[2] = data_frame_detector.D_SDOM

    legend = generate_legend(detector, orientation, current, field_size, 'cm', l)
    return dose, legend


def generate_legend(detector: str, orientation: str, current: str,
                    field_size: float, unit: str, l: str):
    temp_legend = []

    if int(l[0]) == 1:
        temp_legend.append(detector)

    if int(l[1]) == 1:
        temp_legend.append(orientation)

    if int(l[2]) == 1:
        temp_legend.append(current)

    if int(l[3]) == 1:
        temp_legend.append(str(field_size))

    if (len(temp_legend) == 1) & int(l[3]) == 1:
        legend = temp_legend[0] + ' ' + unit
    elif len(temp_legend) == 1:
        legend = temp_legend[0]
    if (len(temp_legend) == 2) & int(l[3]) == 1:
        legend = temp_legend[0] + ', ' + temp_legend[1] + ' ' + unit
    elif len(temp_legend) == 2:
        legend = temp_legend[0] + ', ' + temp_legend[1]
    if (len(temp_legend) == 3) & int(l[3]) == 1:
        legend = temp_legend[0] + ', ' + temp_legend[1] + ', ' + temp_legend[2] + ' ' + unit
    elif len(temp_legend) == 3:
        legend = temp_legend[0] + ', ' + temp_legend[1] + ', ' + temp_legend[2]
    if (len(temp_legend) == 4) & int(l[3]) == 1:
        legend = temp_legend[0] + ', ' + temp_legend[1] + ', ' + temp_legend[2] + ', ' + temp_legend[3] + ' ' + unit
    elif len(temp_legend) == 4:
        legend = temp_legend[0] + ', ' + temp_legend[1] + ', ' + temp_legend[2] + ', ' + temp_legend[3]

    return legend


# Functions Boby
def open_spectra(filename, channel=0):
    xr_init = xr.open_dataarray(filename, engine="h5netcdf")
    spectra = xr_init.isel(spectrum_or_temp=0, channel=channel).sum(dim="stack")
    return spectra


# Pour avoir le spectre total (la somme de chaque frame), faire 'spectra.sum(dim="stack")'
# Pour convertir en array numpy, faire numpy.array(spectra)

def compute_abundances(R: np.ndarray, M: np.ndarray):
    """
    Applies the "RTRmachin" (Least square algorithm) to compute abundances from the measurements and the corresponding
    endmembers.
    @param R: Spectra matrix (nb_channel x nb_spectra).
    @param M: Measurement matrix (nb_channel x nb_measurements).
    @return: Abundance matrix (nb_spectra x nb_measurements).
    """
    F = np.linalg.inv(R.T @ R) @ R.T
    x = F @ M
    return x


def make_calib_spectra(scint, fluo, ckov1, ckov2, ckov3, ckov4):
    ckovA = abs(ckov1 - ckov2)
    ckovB = abs(ckov3 - ckov4)

    R_xr = xr.concat((scint, fluo, ckovA, ckovB), dim="temp")
    R_norm = R_xr / R_xr.sum(axis=0)
    R = np.array(R_norm)
    return R.T

def compute_dose(R: np.ndarray, M: np.ndarray, calib_spectrum: np.ndarray, calib_doseval: float):
    """
    Computes the dose given a known dose measurement.
    @param R: Endmember matrix (nb_channel x nb_endmember).
    @param M: Measurement matrix (nb_channel x nb_measurements).
    @param calib_spectrum: Spectrum of known dose.
    @param calib_doseval: Known dose.
    @param method: Method for abundance computation.
    @return: Absolute Dose matrix (nb_endmembers x nb_measurements).
    """
    R = np.atleast_2d(R)
    M = np.atleast_2d(M)
    calib_spectrum = np.atleast_2d(calib_spectrum)
    x1 = compute_abundances(R, M)
    xcal1 = compute_abundances(R, calib_spectrum)
    dose1 = (x1 / xcal1 * calib_doseval)
    return dose1

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

