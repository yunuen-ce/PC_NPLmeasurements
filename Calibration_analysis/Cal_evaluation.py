import numpy as np
from functions_medscint import *
import matplotlib.pyplot as plt
import seaborn as sns

symbol = ['d', '^', 's', 'o', '*']
scintillators = ['bcf10', 'bcf60', 'ej204', 'bcf60-Lee', 'Medscint']
scint_label = ['BCF-10', 'BCF-60', 'EJ-204', 'BCF-60 Lee filter', 'Medscint']
bfields = [0, 0.2, 0.35, 0.5, 1, 1.5]
bfields_label = ['0', '0.2', '0.35', '0.5', '1', '1.5', '0', '-0.2', '-0.35', '-0.5', '-1', '-1.5']
colors = ['tomato', 'deepskyblue', 'mediumseagreen', 'purple', 'grey', 'darkorange', 'plum', 'black']
colors2 = ['light red', 'lightskyblue', 'lightseagreen', 'lightpurple',  'black']
path = '/Users/yun/PycharmProjects/NPLmeasurements/'
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

residuals = np.zeros([5, 12, 137])
mean_residuals = np.zeros([5, 12])
std_residuals = np.zeros([5, 12])
for s in range(len(scintillators)):
    for b in range(12):
        residuals[s, b, :] = Mean_spectrum[s, b, :] - (cal_scint[s] * Mean_weight[s, b, 0]
                                                       + cal_fluo[s] * Mean_weight[s, b, 1]
                                                       + cal_ckovA[s] * Mean_weight[s, b, 2]
                                                       + cal_ckovB[s] * Mean_weight[s, b, 3])

        mean_residuals[s, b] = np.mean(residuals[s, b])
        std_residuals[s, b] = np.std(residuals[s, b])


for s in range(len(scintillators)):

    b = 0
    plt.title(scint_label[s] + ', ' + str(bfields[b]) + ' T')
    plt.plot(Mean_spectrum[s, b, :], '--', label='Measurement')
    plt.plot(cal_scint[s] * Mean_weight[s, b, 0] + cal_fluo[s] * Mean_weight[s, b, 1]
             + cal_ckovA[s] * Mean_weight[s, b, 2] + cal_ckovB[s] * Mean_weight[s, b, 3], '--', label='Computed')
    plt.plot(cal_scint[s] * Mean_weight[s, b, 0], ':', label='cal scint 0 T')
    plt.plot(cal_fluo[s] * Mean_weight[s, b, 1], ':', label='cal fluo 0 T')
    plt.plot(cal_ckovA[s] * Mean_weight[s, b, 2], ':', label='cal Cherenkov A 0 T')
    plt.plot(cal_ckovB[s] * Mean_weight[s, b, 3], ':', label='cal Cherenkov B 0 T')
    plt.ylabel('Intensity ')
    plt.xlabel('Wavelength [a.u.]')
    plt.legend()
    plt.show()

for s in range(len(scintillators)):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 7), gridspec_kw={'height_ratios': [2, 1]})
    b = 0
    plt.subplot(2, 2, 1)
    plt.title(scint_label[s] + ', ' + str(bfields[b]) + ' T')
    plt.plot(Mean_spectrum[s, b, :], '--', label='Measurement')
    plt.plot(cal_scint[s] * Mean_weight[s, b, 0] + cal_fluo[s] * Mean_weight[s, b, 1]
             + cal_ckovA[s] * Mean_weight[s, b, 2] + cal_ckovB[s] * Mean_weight[s, b, 3], '--', label='Computed')
    plt.plot(cal_scint[s] * Mean_weight[s, b, 0], ':', label='scintillation')
    plt.plot(cal_fluo[s] * Mean_weight[s, b, 1], ':', label='fluo')
    plt.plot(cal_ckovA[s] * Mean_weight[s, b, 2], ':', label='Cherenkov A')
    plt.plot(cal_ckovB[s] * Mean_weight[s, b, 3], ':', label='Cherenkov B')
    plt.ylabel('Intensity')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(Mean_spectrum[s, b, :] - (cal_scint[s] * Mean_weight[s, b, 0] + cal_fluo[s] * Mean_weight[s, b, 1]
                                       + cal_ckovA[s] * Mean_weight[s, b, 2] + cal_ckovB[s] * Mean_weight[s, b, 3]))
    plt.ylabel('measurement - computed')
    b = 4  # at 1 T
    plt.subplot(2, 2, 2)
    plt.title(scint_label[s] + ', ' + str(bfields[b]) + ' T')
    plt.plot(Mean_spectrum[s, b, :], label='measurement')
    plt.plot(cal_scint[s] * Mean_weight[s, b, 0] + cal_fluo[s] * Mean_weight[s, b, 1]
             + cal_ckovA[s] * Mean_weight[s, b, 2] + cal_ckovB[s] * Mean_weight[s, b, 3], '--', label='Computed')
    plt.plot(cal_scint[s] * Mean_weight[s, b, 0], ':', label='scintillation')
    plt.plot(cal_fluo[s] * Mean_weight[s, b, 1], ':', label='fluo')
    plt.plot(cal_ckovA[s] * Mean_weight[s, b, 2], ':', label='Cherenkov A')
    plt.plot(cal_ckovB[s] * Mean_weight[s, b, 3], ':', label='Cherenkov B')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(Mean_spectrum[s, b, :] - (cal_scint[s] * Mean_weight[s, b, 0] + cal_fluo[s] * Mean_weight[s, b, 1]
                                       + cal_ckovA[s] * Mean_weight[s, b, 2] + cal_ckovB[s] * Mean_weight[s, b, 3]))

    plt.show()



for s in range(len(scintillators)):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 7), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    for b in [0, 4]:

        plt.subplot(2, 1, 1)
        plt.title(scint_label[s])

        plt.plot(Mean_spectrum[s, b, :], label='measurement, ' + bfields_label[b] + ' T')
        plt.plot(cal_scint[s] * Mean_weight[s, b, 0] + cal_fluo[s] * Mean_weight[s, b, 1]
                 + cal_ckovA[s] * Mean_weight[s, b, 2] + cal_ckovB[s] * Mean_weight[s, b, 3], '--',
                 label='computed, ' + bfields_label[b] + ' T')
        plt.ylabel('Intensity')
        plt.legend()
        plt.subplot(2, 1, 2)

        plt.plot(residuals[s, b, :], label=bfields_label[b] + ' T')
        plt.ylabel('measurement - computed')
        plt.xlabel('Wavelength [a.u.]')
        plt.legend()
    plt.show()

# Mean residuals
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
plt.savefig("mean_residuals.eps")
plt.show()

'''# Analysis of residuals
for s in range(len(scintillators)):
    for b in range(12):

        # Scatter plot of residuals
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(residuals[s, b, :])), residuals[s, b, :], label=scint_label[s] +', '+ bfields_label[b], alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.axhline(y=mean_residuals[s, b], color='orange', linestyle=':', label='mean residual')
        plt.xlabel("Wavelength [a.u.]")
        plt.ylabel("Residual")
        plt.legend()
        plt.title("Scatter plot of residuals")

        # Histogram of residuals
        plt.subplot(1, 2, 2)
        sns.histplot(residuals[s, b, :], kde=True, label=scint_label[s] +', '+ bfields_label[b],)
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title("Histogram of residuals")

        plt.tight_layout()
        plt.show()
'''