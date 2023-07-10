import numpy as np
from functions_medscint import *
import matplotlib.pyplot as plt
import seaborn as sns

symbol = ['d', 's', 'o','^',  '*']
scintillators = ['bcf10', 'bcf60', 'ej204', 'bcf60-Lee', 'Medscint']
scint_label = ['BCF-10', 'BCF-60', 'EJ-204', 'BCF-60 Lee filter', 'Medscint']
bfields = [0, 0.2, 0.35, 0.5, 1, 1.5]
bfields_label = ['0', '0.2', '0.35', '0.5', '1', '1.5', '0', '-0.2', '-0.35', '-0.5', '-1', '-1.5']
colors = ['dodgerblue', 'mediumseagreen', 'purple', 'tomato', 'grey', 'darkorange',  'black']
colors2 = ['light red', 'lightskyblue', 'lightseagreen', 'lightpurple',  'black']
path = '/Users/yunuen/PycharmProjects/PC_NPLmeasurements/'
#path = '/Users/yun/PycharmProjects/NPLmeasurements/'
data = np.load(path + 'data_npl_measures.npz')
Mean_dose = data['Mean_dose']
Mean_fluo = data['Mean_fluo']
Mean_ckov = data['Mean_ckov']
std_dose = data['std_dose']
std_fluo = data['std_fluo']
std_ckov = data['std_ckov']
cal_scint = data['cal_scint']
cal_fluo = data['cal_fluo']
cal_ckovA = data['cal_ckovA']
cal_ckovB = data['cal_ckovB']
Mean_spectrum = data['Mean_spectrum']
cal_dose_file = data['cal_dose_file']
Data = data['Data']
Moy_weights = data['Moy_weights']
dev_weights = data['dev_weights']
Mean_weight = data['Mean_weight']
std_weight = data['std_weight']
Weights = data['Weights']

data_mc = np.load(path + 'Experimental_analysis/dose_mc.npz')
bfields_mc = [-1.5, -1, -0.5, -0.35, -0.2, 0, 0.2, 0.35, 0.5, 1, 1.5]
dose_mc = data_mc['dose_mc']
print(bfields_mc)
j = bfields_mc.index(0)



# ----------------------------------
# Figure with all scintillator
# ----------------------------------
# Figures with  MC

# Scintillation
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
for s in range(len(scintillators) ):
    plt.subplot(211)
    plt.errorbar(bfields, Mean_dose[s][6:12] / Mean_dose[s][6],
                 yerr=np.sqrt((std_dose[s][6:12] / Mean_dose[s][6:12]) ** 2 + (std_dose[s][6] / Mean_dose[s][6]) ** 2) *
                      (Mean_dose[s][6:12] / Mean_dose[s][6]), color=colors[s], fmt=symbol[s], linestyle='-',
                 label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_dose[s][0:6] / Mean_dose[s][0],
                 yerr=np.sqrt((std_dose[s][0:6] / Mean_dose[s][0:6]) ** 2 + (std_dose[s][0] / Mean_dose[s][0]) ** 2) * (
                      Mean_dose[s][0:6] / Mean_dose[s][0]), markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle=':',
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Scintillation abundance /\n scintillation abundance [0 T]')
    plt.subplot(212)
    plt.plot(bfields, 100*((Mean_dose[s][6:12] / Mean_dose[s][6]) - (Mean_dose[s][0:6] / Mean_dose[s][0])) /
             (Mean_dose[s][0:6] / Mean_dose[s][0]), color=colors[s], marker=symbol[s], label=scint_label[s])
    plt.ylabel('Relative difference \n in orientation [%]')
    plt.xlabel('Magnetic field [T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.subplot(211)
plt.errorbar(np.abs(bfields_mc[:j + 1]), dose_mc[1, :j + 1, 0] / dose_mc[1, j, 0],
                 yerr=np.sqrt(dose_mc[1, :j + 1, 2] ** 2 + dose_mc[1, j, 2] ** 2), fmt='-^', color='black',
                 label='Monte Carlo dose' + ' e- \u2192 stem')
plt.errorbar(np.abs(bfields_mc[j:]), dose_mc[1, j:, 0] / dose_mc[1, j, 0],
                 yerr=np.sqrt(dose_mc[1, j:, 2] ** 2 + dose_mc[1, j, 2] ** 2), fmt=':^', color='black',
                 label='Monte Carlo dose' + ' e- \u2192 tip')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("scint_7x7.eps")
plt.show()

#Cerenkov
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
for s in range(len(scintillators)):
    plt.subplot(211)
    plt.errorbar(bfields, Mean_ckov[s][6:12] / Mean_ckov[s][6],
                 yerr=np.sqrt((std_ckov[s][6:12] / Mean_ckov[s][6:12]) ** 2 + (std_ckov[s][6] / Mean_ckov[s][6]) ** 2) *
                      (Mean_ckov[s][6:12] / Mean_ckov[s][6]),  color=colors[s], fmt=symbol[s], linestyle='-', label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_ckov[s][0:6] / Mean_ckov[s][0],
                 yerr=np.sqrt((std_ckov[s][0:6] / Mean_ckov[s][0:6]) ** 2 + (std_ckov[s][0] / Mean_ckov[s][0]) ** 2) * (
                      Mean_ckov[s][0:6] / Mean_ckov[s][0]),  markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle='--',
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Cerenkov abundance /\n Cerenkov abundance [0 T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplot(212)
    plt.plot(bfields, 100*((Mean_ckov[s][6:12] / Mean_ckov[s][6]) - (Mean_ckov[s][0:6] / Mean_ckov[s][0])) /
            (Mean_ckov[s][0:6] / Mean_ckov[s][0]), color=colors[s], marker=symbol[s], label=scint_label[s])
    plt.ylabel('Relative difference \n in orientation [%]')
    plt.xlabel('Magnetic field [T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("ckov_7x7.eps")
plt.show()

# ------------------------------------
# Effect of the base core material
# Comparison between EJ-204  and BCF-10
# ------------------------------------
# Absolute spectra
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), sharey=True)

plt.subplot(131)
plt.plot(Mean_spectrum[2][0], label=str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[2][5], ls='--', label=str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[2][10], label=str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.xlabel('Wavelength (a.u.)')
plt.ylabel('Intensity (a.u.)' )
plt.ylim([0, 75000])
plt.title(scint_label[2])
plt.legend()

plt.subplot(132)
plt.plot(Mean_spectrum[0][0], label=str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[0][5],  ls='--',label=str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[0][10], label=str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.xlabel('Wavelength (a.u.)')
plt.ylim([0, 75000])
plt.title(scint_label[0])
plt.legend()

plt.subplot(133)
plt.plot(Mean_spectrum[1][0], label=str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[1][5],  ls='--',label=str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[1][10], label=str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.xlabel('Wavelength (a.u.)')
plt.ylim([0, 75000])
plt.title(scint_label[1])
plt.legend()

plt.tight_layout()
plt.savefig('abs_spectrum_basecore.eps')
plt.show()

# could I use a pvalue comparison ?

# Scintillation
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
for s in [0, 2]:
    plt.subplot(211)
    plt.errorbar(bfields, Mean_dose[s][6:12] / Mean_dose[s][6],
                 yerr=np.sqrt((std_dose[s][6:12] / Mean_dose[s][6:12]) ** 2 + (std_dose[s][6] / Mean_dose[s][6]) ** 2) *
                      (Mean_dose[s][6:12] / Mean_dose[s][6]), color=colors[s], fmt=symbol[s], linestyle='-',
                 label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_dose[s][0:6] / Mean_dose[s][0],
                 yerr=np.sqrt((std_dose[s][0:6] / Mean_dose[s][0:6]) ** 2 + (std_dose[s][0] / Mean_dose[s][0]) ** 2) * (
                      Mean_dose[s][0:6] / Mean_dose[s][0]), markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle=':',
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Scintillation abundance /\n scintillation abundance [0 T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplot(212)
    plt.plot(bfields, 100*((Mean_dose[s][6:12] / Mean_dose[s][6]) - (Mean_dose[s][0:6] / Mean_dose[s][0])) /
             (Mean_dose[s][0:6] / Mean_dose[s][0]), color=colors[s], marker=symbol[s], label=scint_label[s])
    plt.ylabel('Relative difference \n in orientation [%]')
    plt.xlabel('Magnetic field [T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("scint_7x7.eps")
plt.show()

#Cerenkov
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
for s in [0, 2]:
    plt.subplot(211)
    plt.errorbar(bfields, Mean_ckov[s][6:12] / Mean_ckov[s][6],
                 yerr=np.sqrt((std_ckov[s][6:12] / Mean_ckov[s][6:12]) ** 2 + (std_ckov[s][6] / Mean_ckov[s][6]) ** 2) *
                      (Mean_ckov[s][6:12] / Mean_ckov[s][6]),  color=colors[s], fmt=symbol[s], linestyle='-', label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_ckov[s][0:6] / Mean_ckov[s][0],
                 yerr=np.sqrt((std_ckov[s][0:6] / Mean_ckov[s][0:6]) ** 2 + (std_ckov[s][0] / Mean_ckov[s][0]) ** 2) * (
                      Mean_ckov[s][0:6] / Mean_ckov[s][0]),  markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle='--',
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Cerenkov abundance /\n Cerenkov abundance [0 T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplot(212)
    plt.plot(bfields, 100*((Mean_ckov[s][6:12] / Mean_ckov[s][6]) - (Mean_ckov[s][0:6] / Mean_ckov[s][0])) /
            (Mean_ckov[s][0:6] / Mean_ckov[s][0]), color=colors[s], marker=symbol[s], label=scint_label[s])
    plt.ylabel('Relative difference \n in orientation [%]')
    plt.xlabel('Magnetic field [T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("ckov_7x7.eps")
plt.show()


# ------------------------------------
# Effect of the wavelength shifter
# Comparison between BCF-10 and BCF-60
# ------------------------------------

# Scintillation
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
for s in [0, 1]:
    plt.subplot(211)
    plt.errorbar(bfields, Mean_dose[s][6:12] / Mean_dose[s][6],
                 yerr=np.sqrt((std_dose[s][6:12] / Mean_dose[s][6:12]) ** 2 + (std_dose[s][6] / Mean_dose[s][6]) ** 2) *
                      (Mean_dose[s][6:12] / Mean_dose[s][6]), color=colors[s], fmt=symbol[s], linestyle='-',
                 label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_dose[s][0:6] / Mean_dose[s][0],
                 yerr=np.sqrt((std_dose[s][0:6] / Mean_dose[s][0:6]) ** 2 + (std_dose[s][0] / Mean_dose[s][0]) ** 2) * (
                      Mean_dose[s][0:6] / Mean_dose[s][0]), markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle=':',
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Scintillation abundance /\n scintillation abundance [0 T]')
    plt.subplot(212)
    plt.plot(bfields, 100*((Mean_dose[s][6:12] / Mean_dose[s][6]) - (Mean_dose[s][0:6] / Mean_dose[s][0])) /
             (Mean_dose[s][0:6] / Mean_dose[s][0]), color=colors[s], marker=symbol[s], label=scint_label[s])
    plt.ylabel('Relative difference \n in orientation [%]')
    plt.xlabel('Magnetic field [T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("scint_7x7.eps")
plt.show()

#Cerenkov
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
for s in [0, 2]:
    plt.subplot(211)
    plt.errorbar(bfields, Mean_ckov[s][6:12] / Mean_ckov[s][6],
                 yerr=np.sqrt((std_ckov[s][6:12] / Mean_ckov[s][6:12]) ** 2 + (std_ckov[s][6] / Mean_ckov[s][6]) ** 2) *
                      (Mean_ckov[s][6:12] / Mean_ckov[s][6]),  color=colors[s], fmt=symbol[s], linestyle='-', label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_ckov[s][0:6] / Mean_ckov[s][0],
                 yerr=np.sqrt((std_ckov[s][0:6] / Mean_ckov[s][0:6]) ** 2 + (std_ckov[s][0] / Mean_ckov[s][0]) ** 2) * (
                      Mean_ckov[s][0:6] / Mean_ckov[s][0]),  markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle='--',
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Cerenkov abundance /\n Cerenkov abundance [0 T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplot(212)
    plt.plot(bfields, 100*((Mean_ckov[s][6:12] / Mean_ckov[s][6]) - (Mean_ckov[s][0:6] / Mean_ckov[s][0])) /
            (Mean_ckov[s][0:6] / Mean_ckov[s][0]), color=colors[s], marker=symbol[s], label=scint_label[s])
    plt.ylabel('Relative difference \n in orientation [%]')
    plt.xlabel('Magnetic field [T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("ckov_7x7.eps")
plt.show()

# Absolute spectra
plt.rcParams.update({'font.size': 12})
#fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), sharey=True)
plt.title('Wavelength shifter')
plt.plot(Mean_spectrum[2][0], label=scint_label[2] + ', ' + str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[2][5], label=scint_label[2] + ', ' + str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[2][10], label=scint_label[2] + ', ' + str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.xlabel('Wavelength (a.u.)')
plt.ylabel('Intensity (a.u.)' )
plt.ylim([0, 75000])
#plt.title(scint_label[2])
plt.legend()

plt.plot(Mean_spectrum[0][0], ls='--', label=scint_label[0] + ', ' + str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[0][5],  ls='--',label=scint_label[0] + ', ' + str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[0][10], ls='--', label=scint_label[0] + ', ' + str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.xlabel('Wavelength (a.u.)')
plt.ylim([0, 75000])
#plt.title(scint_label[0])
plt.legend()

plt.tight_layout()
plt.savefig('abs_spectrum_waveshifter.eps')
plt.show()
# ------------------------------------
# Effect of the Lee filter
# Comparison between BCF-60 and BCF-60 with Lee filter
# ------------------------------------

# Scintillation
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
for s in [1, 3]:
    plt.subplot(211)
    plt.errorbar(bfields, Mean_dose[s][6:12] / Mean_dose[s][6],
                 yerr=np.sqrt((std_dose[s][6:12] / Mean_dose[s][6:12]) ** 2 + (std_dose[s][6] / Mean_dose[s][6]) ** 2) *
                      (Mean_dose[s][6:12] / Mean_dose[s][6]), color=colors[s], fmt=symbol[s], linestyle='-',
                 label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_dose[s][0:6] / Mean_dose[s][0],
                 yerr=np.sqrt((std_dose[s][0:6] / Mean_dose[s][0:6]) ** 2 + (std_dose[s][0] / Mean_dose[s][0]) ** 2) * (
                      Mean_dose[s][0:6] / Mean_dose[s][0]), markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle=':',
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Scintillation abundance /\n scintillation abundance [0 T]')
    plt.subplot(212)
    plt.plot(bfields, 100*((Mean_dose[s][6:12] / Mean_dose[s][6]) - (Mean_dose[s][0:6] / Mean_dose[s][0])) /
             (Mean_dose[s][0:6] / Mean_dose[s][0]), color=colors[s], marker=symbol[s], label=scint_label[s])
    plt.ylabel('Relative difference \n in orientation [%]')
    plt.xlabel('Magnetic field [T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("scint_7x7.eps")
plt.show()

#Cerenkov
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
for s in [1, 3]:
    plt.subplot(211)
    plt.errorbar(bfields, Mean_ckov[s][6:12] / Mean_ckov[s][6],
                 yerr=np.sqrt((std_ckov[s][6:12] / Mean_ckov[s][6:12]) ** 2 + (std_ckov[s][6] / Mean_ckov[s][6]) ** 2) *
                      (Mean_ckov[s][6:12] / Mean_ckov[s][6]),  color=colors[s], fmt=symbol[s], linestyle='-', label=scint_label[s] + ', e- \u2192 stem')
    plt.errorbar(bfields, Mean_ckov[s][0:6] / Mean_ckov[s][0],
                 yerr=np.sqrt((std_ckov[s][0:6] / Mean_ckov[s][0:6]) ** 2 + (std_ckov[s][0] / Mean_ckov[s][0]) ** 2) * (
                      Mean_ckov[s][0:6] / Mean_ckov[s][0]),  markerfacecolor='none', color=colors[s], fmt=symbol[s], linestyle='--',
                 label=scint_label[s] + ', e- \u2192 tip')
    plt.ylabel('Cerenkov abundance /\n Cerenkov abundance [0 T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplot(212)
    plt.plot(bfields, 100*((Mean_ckov[s][6:12] / Mean_ckov[s][6]) - (Mean_ckov[s][0:6] / Mean_ckov[s][0])) /
            (Mean_ckov[s][0:6] / Mean_ckov[s][0]), color=colors[s], marker=symbol[s], label=scint_label[s])
    plt.ylabel('Relative difference \n in orientation [%]')
    plt.xlabel('Magnetic field [T]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("ckov_7x7.eps")
plt.show()

# Absolute spectra
plt.rcParams.update({'font.size': 12})
#fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), sharey=True)
plt.title('Lee filter')
plt.plot(Mean_spectrum[1][0], label=scint_label[1] + ', ' + str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[1][5], label=scint_label[1] + ', ' + str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[1][10], label=scint_label[1] + ', ' + str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.xlabel('Wavelength (a.u.)')
plt.ylabel('Intensity (a.u.)' )
#plt.ylim([0, 75000])

plt.legend()

plt.plot(Mean_spectrum[3][0], ls='--', label=scint_label[3] + ', ' + str(bfields[0]) + ' T')
plt.plot(Mean_spectrum[3][5],  ls='--',label=scint_label[3] + ', ' + str(bfields[5]) + ' T' + ', e- \u2192 tip')
plt.plot(Mean_spectrum[3][10], ls='--', label=scint_label[3] + ', ' + str(bfields[5]) + ' T' + ', e- \u2192 stem')
plt.xlabel('Wavelength (a.u.)')
#plt.ylim([0, 95000])
#plt.xlim([0, 200])
#plt.title(scint_label[0])
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('abs_spectrum_Leefilter.eps')
plt.show()

