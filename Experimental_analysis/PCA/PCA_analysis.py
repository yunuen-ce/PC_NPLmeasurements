import numpy as np
from sklearn.decomposition import PCA
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from typing import Optional

# functions
def make_PCA(calib_spectra_norm: np.ndarray, dose_spectra_norm: np.ndarray):
    """
    Applies the PCA algorithm onto the data.
    @param calib_spectra: Array of calibration spectra (nb_spectra x nb_channel, typically 4 x 136).
    @param dose_spectra: Array of dose spectra (nb_spectra x nb_channel, typically M x 136).
    @return: Reduced array of principal components (for calib and dose spectra).
    """
    # Make sure spectra are normalized
    #calib_spectra_norm = calib_spectra / calib_spectra.sum(axis=1)
    #dose_spectra_norm = dose_spectra / dose_spectra.sum(axis=1)
    # Apply PCA
    pca = PCA(n_components=calib_spectra_norm.shape[0])
    pc_spectra = pca.fit_transform(deepcopy(calib_spectra_norm))
    pc_dataset = pca.transform(dose_spectra_norm)

    return pc_spectra, pc_dataset

def plot_simplex(ax, *array_pts: np.ndarray, consider_last_component: bool = False):
    """
    Plots a simplex given a series of pts arrays.
    @param array_pts: The arrays of the points (2D or 3D points)
    @param consider_last_component: If True, the last component of the PCA will be considered (for example,
    a 3-endmembers PCA will be plotted in a 3D space).
    @return: None
    """
    if len(array_pts) == 4:
        if consider_last_component:
            print("It is impossible to plot more than 3 components in the PCA space.")
        else:
            ax.plot([array_pts[0][0], array_pts[1][0]], [array_pts[0][1], array_pts[1][1]],
                    [array_pts[0][2], array_pts[1][2]], color='k')
            ax.plot([array_pts[0][0], array_pts[2][0]], [array_pts[0][1], array_pts[2][1]],
                    [array_pts[0][2], array_pts[2][2]], color='k')
            ax.plot([array_pts[0][0], array_pts[3][0]], [array_pts[0][1], array_pts[3][1]],
                    [array_pts[0][2], array_pts[3][2]], color='k')
            ax.plot([array_pts[1][0], array_pts[2][0]], [array_pts[1][1], array_pts[2][1]],
                    [array_pts[1][2], array_pts[2][2]], color='k')
            ax.plot([array_pts[1][0], array_pts[3][0]], [array_pts[1][1], array_pts[3][1]],
                    [array_pts[1][2], array_pts[3][2]], color='k')
            ax.plot([array_pts[2][0], array_pts[3][0]], [array_pts[2][1], array_pts[3][1]],
                    [array_pts[2][2], array_pts[3][2]], color='k')
            labels = [ 'Scint', 'Ckov-A', 'Ckov-B', 'Fluo',]
            for i, pt in enumerate(array_pts):
                ax.scatter(pt[0], pt[1], pt[2], c='r', marker='o')
                ax.text(pt[0], pt[1], pt[2], labels[i], fontsize=12)
    elif len(array_pts) == 3:
        if consider_last_component:
            ax.plot([array_pts[0][0], array_pts[1][0]], [array_pts[0][1], array_pts[1][1]],
                    [array_pts[0][2], array_pts[0][2]], color='k')
            ax.plot([array_pts[0][0], array_pts[2][0]], [array_pts[0][1], array_pts[2][1]],
                    [array_pts[1][2], array_pts[1][2]], color='k')
            ax.plot([array_pts[1][0], array_pts[2][0]], [array_pts[1][1], array_pts[2][1]],
                    [array_pts[2][2], array_pts[2][2]], color='k')
        else:
            ax.plot([array_pts[0][0], array_pts[1][0]], [array_pts[0][1], array_pts[1][1]], color='k')
            ax.plot([array_pts[0][0], array_pts[2][0]], [array_pts[0][1], array_pts[2][1]], color='k')
            ax.plot([array_pts[1][0], array_pts[2][0]], [array_pts[1][1], array_pts[2][1]], color='k')
    elif len(array_pts) == 2:
        if consider_last_component:
            ax.plot([array_pts[0][0], array_pts[1][0]], [array_pts[0][1], array_pts[1][1]], color='k')
        else:
            ax.plot([array_pts[0][0], array_pts[1][0]], np.zeros_like([array_pts[0][1], array_pts[1][1]]), color='k')
    else:
        raise AssertionError("Impossible to plot the simplex with more than 4 pure components.")

def plot_PCA(calib_pca: np.ndarray, dose_pca: np.ndarray, ax: Optional[plt.axes] = None, plot_ref: bool = False):
    """
    Plots the PCA graph of the dataset.
    @param ax: Matplotlib Axe onto which to plot the PCA graph.
    @param plot_ref: If True, the spectra used to build the PCA space will be plotted as well.
    @return: None
    """
    PCA_nb_comp = calib_pca.shape[0]
    if PCA_nb_comp > 4:
        raise AssertionError("Impossible to plot the PCA representation with %d components" % PCA_nb_comp)
    if ax is None and PCA_nb_comp == 4:
        fig, ax = plt.figure(), plt.axes(projection="3d")
    elif ax is None and PCA_nb_comp < 4:
        fig, ax = plt.subplots()

    # Define a custom normalization for the colorbar
    point_values = np.array([-1.5, -1, -0.5, -0.35, -0.2, 0, 0,  0.2, 0.35, 0.5, 1, 1.5])
    vmin, vmax = point_values.min(), point_values.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    #sc = ax.scatter(*dose_pca.T[:-1], c=np.arange(dose_pca.T.shape[1]), cmap='viridis', norm=norm)
    sc = ax.scatter(*dose_pca.T[:-1], c=point_values, cmap='Spectral', norm=norm)
    plot_simplex(ax, *calib_pca)
    #ax.scatter(*dose_pca.T[:-1])
    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label("Magnetic field")
    pca_spectra_without_last_comp = calib_pca[:, :-1]  # last component is always 0 or like 1e-18.
    if plot_ref:
        [ax.scatter(*np.atleast_2d(pca_spectra_without_last_comp[i]).T) for i in range(calib_pca.shape[0])]
    # Add labels to the points
    #labels = ['-1.5', '-1', '-0.5', '-0.35', '-0.2', '0',  '0', '0.2', '0.35', '0.5', '1', '1.5']
    #for i, pt in enumerate(dose_pca.T[:-1].T):
    #    ax.text(pt[0], pt[1], pt[2],  s=labels[i], fontsize=12)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_xticks([])
    ax.set_yticks([])
    if "zaxis" in ax.properties():
        ax.set_zlabel("PC 3")
        ax.set_zticks([])
    ax.legend(loc=0)


# Definitions
symbol = ['d', '^', 's', 'o', '*']
scintillators = ['bcf10', 'bcf60', 'ej204', 'bcf60-Lee', 'Medscint']
scint_label = ['BCF-10', 'BCF-60', 'EJ-204', 'BCF-60 Lee filter', 'Medscint']
bfields = [0, 0.2, 0.35, 0.5, 1, 1.5]
bfields_label = ['0', '0.2', '0.35', '0.5', '1', '1.5', '0', '-0.2', '-0.35', '-0.5', '-1', '-1.5']
colors = ['tomato', 'deepskyblue', 'mediumseagreen', 'purple', 'grey', 'darkorange', 'plum', 'black']
colors2 = ['salmon', 'lightskyblue', 'lightseagreen', 'mediumorchid',  'black']

# Load data
path = "/Users/yunuen/PycharmProjects/PC_NPLmeasurements/New_calibration_ckovtrans/"
data = np.load(path + 'data_new_calibration.npz')
Mean_spectrum = data['Mean_spectrum']
Mean_dose = data['Mean_dose']
Mean_fluo = data['Mean_fluo']
Mean_ckov = data['Mean_ckov']
std_dose = data['std_dose']
std_fluo = data['std_fluo']
std_ckov = data['std_ckov']
cal_dose_file = data['cal_dose_file']
cal_scint = data['cal_scint']
cal_fluo = data['cal_fluo']
cal_ckovA = data['cal_ckovA']
cal_ckovB = data['cal_ckovB']
Data = data['Data']
Mean_weight = data['Mean_weight']
std_weight = data['std_weight']
Weights = data['Weights']
Mean_dose_old = data['Mean_dose_old']
Mean_fluo_old = data['Mean_fluo_old']
Mean_ckov_old = data['Mean_ckov_old']
Mean_weight_old = data['Mean_weight_old']
std_weight_old = data['std_weight_old']
Weights_old = data['Weights_old']
std_dose_old = data['std_dose_old']
std_fluo_old = data['std_fluo_old']
std_ckov_old = data['std_ckov_old']
Mean_spectrum_old = data['Mean_spectrum_old']
dose_mc = data['dose_mc']

cal_spectra=np.zeros([5, 4, 137])
dose_spectra_norm=np.zeros([12, 137])



for s in range(len(scintillators)):
    # Normalization of spectra
    cal_spectra[s, 0, :] = cal_scint[s]/sum(cal_scint[s])
    cal_spectra[s, 1, :] = cal_ckovA[s]/sum(cal_ckovA[s])
    cal_spectra[s, 2, :] = cal_ckovB[s]/sum(cal_ckovB[s])
    cal_spectra[s, 3, :] = cal_fluo[s] / sum(cal_fluo[s])
    norm_calib_spectra=cal_spectra[s]

    # Mean spectrum has a shape 5 x 12 x 137
    # Changing the order of the bfields so i can from -1.5 T to 1.5 T
    for b in range(6):
        dose_spectra_norm[b] = Mean_spectrum[s, 11-b]/sum(Mean_spectrum[s, 11-b])
        dose_spectra_norm[b+6] = Mean_spectrum[s, b] / sum(Mean_spectrum[s, b])



    [pc_spectra,c_dataset] = make_PCA(norm_calib_spectra,dose_spectra_norm)
    plot_PCA(pc_spectra,c_dataset)
    plt.legend()
    plt.show()

cal_spectra=np.zeros([5, 4, 137])
dose_spectra_norm=np.zeros([12, 137])



for s in range(len(scintillators)):
    # Normalization of spectra
    cal_spectra[s, 0, :] = cal_scint[s]/sum(cal_scint[s])
    cal_spectra[s, 1, :] = cal_ckovA[s]/sum(cal_ckovA[s])
    cal_spectra[s, 2, :] = cal_ckovB[s]/sum(cal_ckovB[s])
    cal_spectra[s, 3, :] = cal_fluo[s] / sum(cal_fluo[s])
    norm_calib_spectra=cal_spectra[s]

    # Mean spectrum has a shape 5 x 12 x 137
    # Changing the order of the bfields so i can from -1.5 T to 1.5 T
    for b in range(6):
        dose_spectra_norm[b] = Mean_spectrum[s, 11-b]/sum(Mean_spectrum[s, 11-b])
        dose_spectra_norm[b+6] = Mean_spectrum[s, b] / sum(Mean_spectrum[s, b])



    [pc_spectra,c_dataset] = make_PCA(norm_calib_spectra,dose_spectra_norm)
    plot_PCA(pc_spectra,c_dataset)
    plt.legend()
    plt.show()



