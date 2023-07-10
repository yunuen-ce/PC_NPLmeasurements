from functions_medscint import *
import pandas as pd
from PCA_functions import *

def norm_spectrum(spectrum):
    norm_spec=spectrum/sum(spectrum)
    return norm_spec

#Manufacturer data: attenuation
path="/Users/yunuen/PycharmProjects/PC_NPLmeasurements/"
ESKA_data = pd.read_csv(path + 'att_ESKA_GH4001.csv', header=None, skiprows=1, names=['col1', 'wavelength', 'attenuation'])
k_ref = pd.DataFrame(ESKA_data)


symbol = ['d', '^', 's', 'x', 'o', '*']
scintillators = ['bcf10', 'bcf60', 'ej204', 'bcf60-Lee', 'Medscint']
scint_label = ['BCF-10', 'BCF-60', 'EJ-204', 'BCF-60 Lee filter', 'Medscint']
bfields = [0, 0.2, 0.35, 0.5, 1, 1.5]
bfields_label = ['0', '0.2', '0.35', '0.5', '1', '1.5', '0', '-0.2', '-0.35', '-0.5', '-1', '-1.5']
colors = ['tomato', 'deepskyblue', 'mediumseagreen', 'purple', 'grey', 'darkorange', 'plum', 'black']
colors2 = ['salmon', 'lightskyblue', 'lightseagreen', 'mediumorchid',  'black']
data = np.load(path + 'data_npl_measures.npz')
# raw calibration data
cal_scint = data['cal_scint']
cal_fluo = data['cal_fluo']
cal_ckovA_old = data['cal_ckovA']
cal_ckovB_old = data['cal_ckovB']
cal_dose_file = data['cal_dose_file']
# raw measurement data
Data = data['Data']


# Old calibration
Mean_dose_old = data['Mean_dose']
Mean_fluo_old = data['Mean_fluo']
Mean_ckov_old = data['Mean_ckov']
Moy_weights_old = data['Moy_weights']
dev_weights_old = data['dev_weights']
Mean_weight_old = data['Mean_weight']
std_weight_old = data['std_weight']
Weights_old = data['Weights']
std_dose_old = data['std_dose']
std_fluo_old = data['std_fluo']
std_ckov_old = data['std_ckov']
Mean_spectrum_old = data['Mean_spectrum']

# With the new calibration, Ckov A and B are translated to remove the negative weights.
# the exponential attenuation spectrum is given by

k_mes = np.log(cal_ckovB_old / cal_ckovA_old)
for s in range(len(scintillators)):

    loc_max=max(k_mes[s, 80:100])
    x= 0.027 - loc_max
    k_mes[s, :] = k_mes[s, :] + x

if False:
    for s in range(len(scintillators)):
        plt.plot(k_mes[s, :], color=colors[s], label=scint_label[s])
    plt.legend()
    plt.show()

cal_ckovA_new = np.zeros([5, 137])
cal_ckovB_new = np.zeros([5, 137])
# To move the Cherenkov:
L_A = 5
L_B = 7.5
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharey=True, sharex=True)
for s in range(len(scintillators)):
    cal_ckovA_new[s, :] = cal_ckovA_old[s, :] * (1 - k_mes[s, :] * L_A)
    cal_ckovB_new[s, :] = cal_ckovB_old[s, :] * (1 + k_mes[s, :] * L_B)

    #Figure to see the new Cherenkov
    if s < 3:
        axs[0, s].plot(cal_ckovA_old[s, :], ':', label='ckov A original')
        axs[0, s].plot(cal_ckovB_old[s, :], ':', label='ckov B original')
        axs[0, s].plot(cal_ckovA_new[s, :], label='ckov A new')
        axs[0, s].plot(cal_ckovB_new[s, :], label='ckov B new')
        axs[0, s].set_title(scint_label[s])
    else :
        axs[1, s-3].plot(cal_ckovA_old[s, :], ':', label='ckov A original')
        axs[1, s-3].plot(cal_ckovB_old[s, :], ':', label='ckov B original')
        axs[1, s-3].plot(cal_ckovA_new[s, :], label='ckov A new')
        axs[1, s-3].plot(cal_ckovB_new[s, :], label='ckov B new')
        axs[1, s-3].set_title(scint_label[s])
    for ax in axs.flat:
        ax.set(xlabel='Intensity', ylabel='Wavelength [a.u]')


axs[1, 1].legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
fig.delaxes(axs[1, 2])
plt.show()

Mean_dose_old, Mean_fluo_old, Mean_ckov_old, std_dose_old, std_fluo_old, std_ckov_old,  Mean_spectrum_old, Mean_weight_old, std_weight_old = calculate_weigths(cal_scint, cal_fluo, cal_ckovA_old, cal_ckovB_old, cal_dose_file, Data)
Mean_dose_new, Mean_fluo_new, Mean_ckov_new,  std_dose_new, std_fluo_new,  std_ckov_new,  Mean_spectrum_new, Mean_weight_new, std_weight_new = calculate_weigths(cal_scint, cal_fluo, cal_ckovA_new, cal_ckovB_new, cal_dose_file, Data)

norm_cal_scint = np.zeros([5, 137])
norm_cal_fluo = np.zeros([5, 137])
norm_cal_ckovA_old = np.zeros([5, 137])
norm_cal_ckovB_old = np.zeros([5, 137])
norm_cal_ckovA_new = np.zeros([5, 137])
norm_cal_ckovB_new = np.zeros([5, 137])

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharey=True, sharex=True)

for s in range(len(scintillators)):
    norm_cal_scint[s]= norm_spectrum(cal_scint[s])
    norm_cal_fluo[s] = norm_spectrum(cal_fluo[s])
    norm_cal_ckovA_old[s] = norm_spectrum(cal_ckovA_old[s])
    norm_cal_ckovB_old[s] = norm_spectrum(cal_ckovB_old[s])
    b = 0
    if s < 3:
        plt.title(scint_label[s] + ', ' + str(bfields[b]) + ' T')
        axs[0, s].plot(Mean_spectrum_old[s, b, :], '--', label='Measurement')
        axs[0, s].plot(norm_cal_scint[s] * Mean_weight_old[s, b, 0] +
                       norm_cal_fluo[s] * Mean_weight_old[s, b, 1] +
                       norm_cal_ckovA_old[s] * Mean_weight_old[s, b, 2] +
                       norm_cal_ckovB_old[s] * Mean_weight_old[s, b, 3], '--', label='Computed')
        axs[0, s].plot(norm_cal_scint[s] * Mean_weight_old[s, b, 0], ':', label='cal scint 0 T')
        axs[0, s].plot(norm_cal_fluo[s] * Mean_weight_old[s, b, 1], ':', label='cal fluo 0 T')
        axs[0, s].plot(norm_cal_ckovA_old[s] * Mean_weight_old[s, b, 2], ':', label='cal Cherenkov A 0 T')
        axs[0, s].plot(norm_cal_ckovB_old[s] * Mean_weight_old[s, b, 3], ':', label='cal Cherenkov B 0 T')
        axs[0, s].set_title(scint_label[s], loc="center")
    else:
        plt.title(scint_label[s] + ', ' + str(bfields[b]) + ' T')
        axs[1, s-3].plot(Mean_spectrum_old[s, b, :], '--', label='Measurement')
        axs[1, s-3].plot(norm_cal_scint[s] * Mean_weight_old[s, b, 0] +
                         norm_cal_fluo[s] * Mean_weight_old[s, b, 1] +
                         norm_cal_ckovA_old[s] * Mean_weight_old[s, b, 2] +
                         norm_cal_ckovB_old[s] * Mean_weight_old[s, b, 3], '--', label='Computed')
        axs[1, s-3].plot(norm_cal_scint[s] * Mean_weight_old[s, b, 0], ':', label='cal scint 0 T')
        axs[1, s-3].plot(norm_cal_fluo[s] * Mean_weight_old[s, b, 1], ':', label='cal fluo 0 T')
        axs[1, s-3].plot(norm_cal_ckovA_old[s] * Mean_weight_old[s, b, 2], ':', label='cal Cherenkov A 0 T')
        axs[1, s-3].plot(norm_cal_ckovB_old[s] * Mean_weight_old[s, b, 3], ':', label='cal Cherenkov B 0 T')
        axs[1, s - 3].set_title(scint_label[s],  loc="center")
    for ax in axs.flat:
        ax.set(xlabel='Intensity', ylabel='Wavelength [a.u]')

axs[1, 1].legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
fig.delaxes(axs[1, 2])
plt.savefig('negative_weights.eps')
plt.show()

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharey=True, sharex=True)
for s in range(len(scintillators)):
    norm_cal_ckovA_new[s] = norm_spectrum(cal_ckovA_new[s])
    norm_cal_ckovB_new[s] = norm_spectrum(cal_ckovB_new[s])
    b = 0
    if s < 3:
        plt.title(scint_label[s] + ', h' + str(bfields[b]) + ' T')
        axs[0, s].plot(Mean_spectrum_new[s, b, :], '--', label='Measurement')
        axs[0, s].plot(norm_cal_scint[s] * Mean_weight_new[s, b, 0] +
                       norm_cal_fluo[s] * Mean_weight_new[s, b, 1] +
                       norm_cal_ckovA_new[s] * Mean_weight_new[s, b, 2] +
                       norm_cal_ckovB_new[s] * Mean_weight_new[s, b, 3], '--', label='Computed')
        axs[0, s].plot(norm_cal_scint[s] * Mean_weight_new[s, b, 0], ':', label='cal scint 0 T')
        axs[0, s].plot(norm_cal_fluo[s] * Mean_weight_new[s, b, 1], ':', label='cal fluo 0 T')
        axs[0, s].plot(norm_cal_ckovA_new[s] * Mean_weight_new[s, b, 2], ':', label='cal Cherenkov A 0 T')
        axs[0, s].plot(norm_cal_ckovB_new[s] * Mean_weight_new[s, b, 3], ':', label='cal Cherenkov B 0 T')
        axs[0, s].set_title(scint_label[s], loc="center")
    else:
        plt.title(scint_label[s] + ', ' + str(bfields[b]) + ' T')
        axs[1, s-3].plot(Mean_spectrum_new[s, b, :], '--', label='Measurement')
        axs[1, s-3].plot(norm_cal_scint[s] * Mean_weight_new[s, b, 0] +
                         norm_cal_fluo[s] * Mean_weight_new[s, b, 1] +
                         norm_cal_ckovA_new[s] * Mean_weight_new[s, b, 2] +
                         norm_cal_ckovB_new[s] * Mean_weight_new[s, b, 3], '--', label='Computed')
        axs[1, s-3].plot(norm_cal_scint[s] * Mean_weight_new[s, b, 0], ':', label='cal scint 0 T')
        axs[1, s-3].plot(norm_cal_fluo[s] * Mean_weight_new[s, b, 1], ':', label='cal fluo 0 T')
        axs[1, s-3].plot(norm_cal_ckovA_new[s] * Mean_weight_new[s, b, 2], ':', label='cal Cherenkov A 0 T')
        axs[1, s-3].plot(norm_cal_ckovB_new[s] * Mean_weight_new[s, b, 3], ':', label='cal Cherenkov B 0 T')
        axs[1, s - 3].set_title(scint_label[s],  loc="center")
    for ax in axs.flat:
        ax.set(xlabel='Intensity', ylabel='Wavelength [a.u]')

axs[1, 1].legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
fig.delaxes(axs[1, 2])
plt.savefig('positive_weights.eps')
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2, 1]},  figsize=(11, 7), sharex=True)
s=0
plt.subplot(211)
plt.plot(bfields, Mean_ckov_old[s][6:12], 'o-', label='Original, e- \u2192 stem' )
plt.plot(bfields, Mean_ckov_new[s][6:12],'s-', label='New, e- \u2192 stem' )
plt.plot(bfields, Mean_ckov_old[s][0:6], '*:', label='Original, e- \u2192 tip' )
plt.plot(bfields, Mean_ckov_new[s][0:6],'d:', label='Old, e- \u2192 tip' )
plt.ylabel('Cherenkov abundance')
plt.legend()
plt.subplot(212)
plt.plot(bfields, 100*(Mean_ckov_new[s][6:12]-Mean_ckov_old[s][6:12])/Mean_ckov_old[s][6:12], 's-',label='e- \u2192 stem' )
plt.plot(bfields, 100*(Mean_ckov_new[s][0:6]-Mean_ckov_old[s][0:6])/Mean_ckov_old[s][0:6], 's:',label='e- \u2192 tip' )
plt.legend()
plt.ylabel('Diff[%]')
plt.xlabel('Magnetic field')
plt.savefig('sum_ckov.eps')
plt.show()

#####################
# Effect of the magnetic field on the abundances

path = path + 'Experimental_analysis/'
#Monte Carlo data
data_mc = np.load(path + 'dose_mc.npz')
bfields_mc = [-1.5, -1, -0.5, -0.35, -0.2, 0, 0.2, 0.35, 0.5, 1, 1.5]
dose_mc = data_mc['dose_mc']
j = bfields_mc.index(0)

# Scintillation of all
fig = plt.figure(figsize=(14, 7))
plt.rcParams.update({'font.size': 14})
for s in range(len(scintillators) ):
    plt.errorbar(bfields, Mean_dose_new[s][6:12] / Mean_dose_new[s][6],
                 yerr=np.sqrt((std_dose_new[s][6:12] / Mean_dose_new[s][6:12]) ** 2 + (std_dose_new[s][6] / Mean_dose_new[s][6]) ** 2) *
                      (Mean_dose_new[s][6:12] / Mean_dose_new[s][6]), color=colors[s], fmt=symbol[s], linestyle='-',
                 label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_dose_new[s][0:6] / Mean_dose_new[s][0],
                 yerr=np.sqrt((std_dose_new[s][0:6] / Mean_dose_new[s][0:6]) ** 2 + (std_dose_new[s][0] / Mean_dose_new[s][0]) ** 2) * (
                         Mean_dose_new[s][0:6] / Mean_dose_new[s][0]), markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle=':',
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Scintillation abundance / \n scintillation abundance [0 T]')
    plt.xlabel('Magnetic field [T]')
 # Monte Carlo
plt.errorbar(np.abs(bfields_mc[:j + 1]), dose_mc[1, :j + 1, 0] / dose_mc[1, j, 0],
                 yerr=np.sqrt(dose_mc[1, :j + 1, 2] ** 2 + dose_mc[1, j, 2] ** 2), fmt='-^', color='black',
                 label='Monte Carlo dose' + ' e- \u2192 stem')
plt.errorbar(np.abs(bfields_mc[j:]), dose_mc[1, j:, 0] / dose_mc[1, j, 0],
                 yerr=np.sqrt(dose_mc[1, j:, 2] ** 2 + dose_mc[1, j, 2] ** 2), fmt=':^', color='black',
                 label='Monte Carlo dose' + ' e- \u2192 tip')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("new_scint_abundance.eps")
plt.show()

fig = plt.figure(figsize=(14, 7))
plt.rcParams.update({'font.size': 14})
for s in range(len(scintillators) ):
    plt.errorbar(bfields, Mean_ckov_new[s][6:12] / Mean_ckov_new[s][6],
                 yerr=np.sqrt((std_ckov_new[s][6:12] / Mean_ckov_new[s][6:12]) ** 2 + (std_ckov_new[s][6] / Mean_ckov_new[s][6]) ** 2) *
                      (Mean_ckov_new[s][6:12] / Mean_ckov_new[s][6]), color=colors[s], fmt=symbol[s], linestyle='-',
                 label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_ckov_new[s][0:6] / Mean_ckov_new[s][0],
                 yerr=np.sqrt((std_ckov_new[s][0:6] / Mean_ckov_new[s][0:6]) ** 2 + (std_ckov_new[s][0] / Mean_ckov_new[s][0]) ** 2) * (
                         Mean_ckov_new[s][0:6] / Mean_ckov_new[s][0]), markerfacecolor='none', color=colors[s], fmt=symbol[s],
                 linestyle='--',
                 label=scint_label[s] + ', e- \u2192 tip')
plt.ylabel('Cerenkov abundance / \n Cerenkov abundance [0 T]')
plt.xlabel('Magnetic field [T]')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("new_ckov_abundance.eps")
plt.show()

cal_spectra=np.zeros([5, 3, 137])
dose_spectra_norm=np.zeros([12, 137])
import matplotlib as mpl
mpl.use('macosx')
for s in [4]: #range(len(scintillators) ):
    # Normalization of spectra
    cal_spectra[s, 0, :] = cal_scint[s]/sum(cal_scint[s])
    cal_spectra[s, 1, :] = cal_ckovA_new[s]/sum(cal_ckovA_new[s])
    cal_spectra[s, 2, :] = cal_ckovB_new[s]/sum(cal_ckovB_new[s])
    #cal_spectra[s, 3, :] = cal_fluo[s] / sum(cal_fluo[s])
    norm_calib_spectra=cal_spectra[s]

    # Mean spectrum has a shape 5 x 12 x 137
    # Changing the order of the bfields so i can from -1.5 T to 1.5 T
    for b in range(6):
        dose_spectra_norm[b] = Mean_spectrum_new[s, 11-b]/sum(Mean_spectrum_new[s, 11-b])
        dose_spectra_norm[b+6] = Mean_spectrum_new[s, b] / sum(Mean_spectrum_new[s, b])

    [pc_spectra,c_dataset] = make_PCA(norm_calib_spectra,dose_spectra_norm)
    plot_PCA(pc_spectra,c_dataset )
    plt.title(scint_label[s])
    plt.legend()
    plt.savefig('pca_medscint.eps')
    plt.show()

'''

np.savez('data_new_calibration.npz', Mean_spectrum_new=Mean_spectrum_new,
         Mean_dose_new=Mean_dose_new,
         Mean_fluo_new=Mean_fluo_new,
         Mean_ckov_new=Mean_ckov_new,
         std_dose_new=std_dose_new,
         std_fluo_new=std_fluo_new,
         std_ckov_new=std_ckov_new,
         cal_ckovA=cal_ckovA_new,
         cal_ckovB=cal_ckovB_new,
         Data=Data,
         Mean_dose_old=Mean_dose_old,
         Mean_fluo_old=Mean_fluo_old,
         Mean_ckov_old=Mean_ckov_old,
         std_dose_old=std_dose_old,
         std_fluo_old=std_fluo_old,
         std_ckov_old=std_ckov_old,
         Mean_spectrum_old=Mean_spectrum_old,
         dose_mc=data_mc)

cal_spectra=np.zeros([5, 4, 137])
dose_spectra_norm=np.zeros([12, 137])

for s in range(len(scintillators)):
    # Normalization of spectra
    cal_spectra[s, 0, :] = cal_scint_old[s]/sum(cal_scint_old[s])
    cal_spectra[s, 1, :] = cal_ckovA_old[s] / sum(cal_ckovA_old[s])
    cal_spectra[s, 2, :] = cal_ckovB_old[s] / sum(cal_ckovB_old[s])
    cal_spectra[s, 3, :] = cal_fluo_old[s] / sum(cal_fluo_old[s])
    norm_calib_spectra=cal_spectra[s]

    # Mean spectrum has a shape 5 x 12 x 137
    # Changing the order of the bfields so i can from -1.5 T to 1.5 T
    for b in range(6):
        dose_spectra_norm[b] = Mean_spectrum_old[s, 11 - b] / sum(Mean_spectrum_old[s, 11 - b])
        dose_spectra_norm[b+6] = Mean_spectrum_old[s, b] / sum(Mean_spectrum_old[s, b])

    [pc_spectra,c_dataset] = make_PCA(norm_calib_spectra, dose_spectra_norm)
    plot_PCA(pc_spectra,c_dataset)
    plt.legend()
    plt.show()
    
'''

'''
# Mean residuals
residuals = np.zeros([5, 12, 137])
mean_residuals = np.zeros([5, 12])
std_residuals = np.zeros([5, 12])
for s in range(len(scintillators)):
    for b in range(12):
        residuals[s, b, :] = Mean_spectrum_new[s, b, :] - (cal_scint_old[s] * Mean_weight_new[s, b, 0]
                                                           + cal_fluo_old[s] * Mean_weight_new[s, b, 1]
                                                           + cal_ckovA_new[s] * Mean_weight_new[s, b, 2]
                                                           + cal_ckovB_new[s] * Mean_weight_new[s, b, 3])

        mean_residuals[s, b] = np.mean(residuals[s, b])
        std_residuals[s, b] = np.std(residuals[s, b])

plt.figure(figsize=(10, 6))
for s in range(len(scintillators)):
    plt.plot(bfields, mean_residuals[s, 0:6], marker=symbol[s], linestyle=':', color=colors[s],
             label=scint_label[s] + ', e- \u2192 tip')
    plt.plot(bfields, mean_residuals[s, 6:12], marker=symbol[s], linestyle='--', color=colors[s],
             label=scint_label[s] + ', e- \u2192 stem')
    plt.xlabel('Magnetic field')
    plt.ylabel('Mean residuals')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
#plt.savefig("mean_residuals.eps")
plt.show()
'''