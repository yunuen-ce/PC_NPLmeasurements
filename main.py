import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from matplotlib import gridspec
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
colors = ['tomato', 'deepskyblue', 'mediumseagreen', 'purple', 'grey', 'darkorange',  'plum', 'black',]
lin = ['solid', 'dashed']

'''
# Bfield effect
# Perpendicular
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
plt.subplot(121)
i = 0
for f in fields:
    j = 0
    for c in currents:
        x, y, legend = generate_x_y_diff_of_bfield(df, detectors[4], orientations[0], c, f, '0011')
        plt.plot(x, y, linestyle=lin[j], color=colors[i], label=legend)
        j += 1
    i += 1
tit = detectors[4] + ', ' + orientations[0]
plt.title(tit)
plt.legend(framealpha=1, frameon=True)

# Parallel
i = 0
plt.subplot(122)
for f in fields:
    j = 0
    for c in currents:
        x, y, legend = generate_x_y_diff_of_bfield(df, detectors[4], orientations[1], c, f, '0011')
        plt.plot(x, y, linestyle=lin[j], color=colors[i], label=legend)
        j += 1
    i += 1
tit = detectors[4] + ', ' + orientations[1]
plt.title(tit)
plt.legend(framealpha=1, frameon=True)
plt.show()



# Bfield in the other detectors, only perpendicular orientation
plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
temp = 220
for d in detectors[:-1]:
    i = 0
    temp = temp + 1
    plt.subplot(temp)
    for f in fields:
        j = 0
        for c in currents:
            x, y, legend = generate_x_y_dose_bfield(df, d, orientations[0], c, f, l='0011')
            plt.plot(x, y, linestyle=lin[j], color=colors[i], label=legend)
            j = j + 1
        i = i + 1
    tit = d + ', ' + orientations[0]
    plt.title(tit)
    plt.legend(framealpha=1, frameon=True)
plt.show()

# For each field size
plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
temp = 320
for f in fields:
    i = 0
    for c in currents:
        temp = temp + 1
        plt.subplot(temp)
        j = 0
        for d in detectors:
            x, y, legend = generate_x_y_dose_bfield(df, d, orientations[0], c, f, l='1000')
            plt.plot(x, y, label=legend)
            plt.ylim(0.96, 1.07)
            j = j + 1
        i = i + 1
        tit = orientations[0] + ', ' + c + ', ' + str(f) + ' cm'
        plt.title(tit)
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.show()


# Output factors

# Perpendicular
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
plt.subplot(121)
i = 0
for b in bfields:
    j = 0
    for c in currents:
        x, y, legend = generate_x_y_of(df, detectors[4], orientations[0], c, b, '0011')
        plt.plot(x, y, linestyle=lin[j], color=colors[i],  label=legend)
        j += 1
    i += 1
tit = detectors[4] + ', ' + orientations[0]
plt.title(tit)
plt.legend(framealpha=1, frameon=True)

# Parallel
i = 0
plt.subplot(122)
for b in bfields:
    j = 0
    for c in currents:
        x, y, legend = generate_x_y_of(df, detectors[4], orientations[1], c, b, '0011')
        plt.plot(x, y, linestyle=lin[j], color=colors[i],   label=legend)
        j += 1
    i += 1
tit = detectors[4] + ', ' + orientations[1]
plt.title(tit)
plt.legend(framealpha=1, frameon=True)
plt.show()


plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
temp = 220
for d in detectors[:-1]:
    i = 0
    temp = temp + 1
    plt.subplot(temp)
    for b in bfields:
        j = 0
        for c in currents:
            x, y, legend = generate_x_y_of(df, d, orientations[0], c, b, '0011')
            plt.plot(x, y, linestyle=lin[j], color=colors[i], label=legend)
            plt.ylim(0.935, 1.02)
            j = j + 1
        i = i + 1
    tit = d + ', ' + orientations[0]
    plt.title(tit)
plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.show()
'''
# B-field effect on the waveshifter : comparison between BCF-10 and BCF-60
'''
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 5), gridspec_kw={'height_ratios': [2, 1]})
temp = 230
i = 0

for f in fields:
    temp = temp + 1
    plt.subplot(temp)
    j = 0
    i = 0
    for c in currents:
        d_bcf10, legend = generate_x_y_dose_bfield(df, 'BCF-10', orientations[0], c, f, l='1010')
        d_bcf60, legend2 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], c, f, l='1010')
        plt.errorbar(d_bcf10[0], d_bcf10[1]/d_bcf10[1][0],  yerr=np.sqrt(d_bcf10[2]**2 + d_bcf10[2][0]**2), linestyle=lin[j], color=colors[i], label=legend)
        plt.errorbar(d_bcf60[0], d_bcf60[1]/d_bcf60[1][0],  yerr=np.sqrt(d_bcf60[2]**2 + d_bcf60[2][0]**2), linestyle=lin[j], color=colors[i+1], label=legend2)
        plt.ylim([0.975, 1.06])
        if f==0.5:
            plt.ylabel('Dose / Dose (0 T)')
        j += 1
    i += 1
    tit = str(f) + ' cm'
    plt.title(tit)
plt.legend(framealpha=1, bbox_to_anchor=(1.5, 0.5))

for f in fields:
    temp = temp + 1
    plt.subplot(temp)
    j = 0
    i = 0
    for c in currents:
        d_bcf10, legend = generate_x_y_dose_bfield(df, 'BCF-10', orientations[0], c, f, l='1010')
        d_bcf60, legend2 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], c, f, l='1010')
        plt.plot(d_bcf10[0], 100*(d_bcf10[1]/d_bcf10[1][0] - d_bcf60[1]/d_bcf60[1][0] )/(d_bcf60[1]/d_bcf60[1][0]), color=colors[0], linestyle=lin[j], label=c)
        if f == 0.5:
            plt.ylabel('Relative difference [%]')
        plt.xlabel('Magnetic field')
        plt.ylim([-0.7, 2.5])
        j += 1
    i += 1
plt.tight_layout()
plt.legend(framealpha=1, bbox_to_anchor=(1.5, 0.5))
plt.show()



# B-field effect on the Polysteyrene-based core, filters blue re-excitation  : comparison between BCF-60 and BCF-60 lee

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 5), gridspec_kw={'height_ratios': [2, 1]})

temp = 230
i = 0
for f in fields:
    temp = temp + 1
    plt.subplot(temp)
    j = 0
    i = 0
    for c in currents:
        d_bcf60f, legend = generate_x_y_dose_bfield(df, 'BCF-60 Lee filter', orientations[0], c, f, l='1010')
        d_bcf60, legend2 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], c, f, l='1010')
        plt.errorbar(d_bcf60f[0], d_bcf60f[1]/d_bcf60f[1][0], yerr=np.sqrt(d_bcf60f[2]**2 + d_bcf60f[2][0]**2),
                     linestyle=lin[j], color=colors[i], label=legend)
        plt.errorbar(d_bcf60[0], d_bcf60[1]/d_bcf60[1][0], yerr=np.sqrt(d_bcf60[2]**2 + d_bcf60[2][0]**2),
                     linestyle=lin[j], color=colors[i+1], label=legend2)
        plt.ylim([0.975, 1.06])
        if f == 0.5:
            plt.ylabel('Dose / Dose (0 T)')

        j += 1
    i += 1
    tit = str(f) + ' cm'
    plt.title(tit)
plt.legend(framealpha=1, bbox_to_anchor=(1.5, 0.5))

for f in fields:
    temp = temp + 1
    plt.subplot(temp)
    j = 0
    i = 0
    for c in currents:
        d_bcf60f, legend = generate_x_y_dose_bfield(df, 'BCF-60 Lee filter', orientations[0], c, f, l='1010')
        d_bcf60, legend2 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], c, f, l='1010')
        plt.plot(d_bcf60f[0], 100*(d_bcf60f[1]/d_bcf60f[1][0] - d_bcf60[1]/d_bcf60[1][0] )/(d_bcf60[1]/d_bcf60[1][0]),
                 color=colors[0], linestyle=lin[j], label=c)
        if f == 0.5:
            plt.ylabel('Relative difference [%]')
        plt.xlabel('Magnetic field')
        plt.ylim([-1, 0.5])
        j += 1
    i += 1
plt.tight_layout()
plt.legend(framealpha=1, bbox_to_anchor=(1.5, 0.5))
plt.show()


# B-field effect on the Polyvinyltoluene-based core,  primary : comparison between BCF-10 and EJ-204

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 5), gridspec_kw={'height_ratios': [2, 1]})
temp = 230
for f in fields:
    temp = temp + 1
    plt.subplot(temp)
    j = 0
    i = 0
    for c in currents:
        d_bcf10, legend = generate_x_y_dose_bfield(df, 'BCF-10', orientations[0], c, f, l='1010')
        d_bcf60, legend1 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], c, f, l='1010')
        d_ej, legend2 = generate_x_y_dose_bfield(df, 'EJ-204', orientations[0], c, f, l='1010')
        plt.errorbar(d_bcf10[0], d_bcf10[1]/d_bcf10[1][0], yerr=np.sqrt(d_bcf10[2]**2 + d_bcf10[2][0]**2), linestyle=lin[j], color=colors[i], label=legend)
        plt.errorbar(d_bcf60[0], d_bcf60[1] / d_bcf60[1][0], yerr=np.sqrt(d_bcf60[2]**2 + d_bcf60[2][0]**2), linestyle=lin[j], color=colors[i + 1], label=legend1)
        plt.errorbar(d_ej[0], d_ej[1]/d_ej[1][0], yerr=np.sqrt(d_ej[2]**2 + d_ej[2][0]**2), linestyle=lin[j], color=colors[i+2], label=legend2)
        plt.ylim([0.975, 1.07])
        if f == 0.5:
            plt.ylabel('Dose / Dose (0 T)')
        j += 1
    i += 1
    tit = str(f) + ' cm'
    plt.title(tit)
plt.legend(framealpha=1, bbox_to_anchor=(1.5, 0.6))



for f in fields:
    temp = temp + 1
    plt.subplot(temp)
    j = 0
    i = 0
    for c in currents:
        d_bcf10, legend = generate_x_y_dose_bfield(df, 'BCF-10', orientations[0], c, f, l='1010')
        d_bcf60, legend1 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], c, f, l='1010')
        d_ej, legend2 = generate_x_y_dose_bfield(df, 'EJ-204', orientations[0], c, f, l='1010')
        plt.plot(d_bcf10[0], (d_bcf10[1]/d_bcf10[1][0] - d_bcf60[1] / d_bcf60[1][0]) /(d_bcf60[1] / d_bcf60[1][0]), linestyle=lin[j], color=colors[i], label=legend)
        plt.plot(d_ej[0], (d_ej[1]/d_ej[1][0] - d_bcf60[1] / d_bcf60[1][0]) / (d_bcf60[1] / d_bcf60[1][0]), linestyle=lin[j], color=colors[i+2], label=legend2)
        if f == 0.5:
            plt.ylabel('Relative difference')
        plt.xlabel('Magnetic field')
        plt.ylim([-0.01, 0.026])
        j += 1
    i += 1
plt.tight_layout()
plt.legend(framealpha=1, bbox_to_anchor=(0.945, 0.8))
plt.show()
'''
# Only the 7 x 7 cm field

fig, axes = plt.subplots(nrows=2, ncols=1,  gridspec_kw={'height_ratios': [2, 1]})
f=fields[2]
c2 = ['e- \u2192 stem', 'e- \u2192 tip']
plt.subplot(211)
j = 0
i = 0
for c in currents:
    d_bcf10, legend = generate_x_y_dose_bfield(df, 'BCF-10', orientations[0], c, f, l='1010')
    d_bcf60, legend1 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], c, f, l='1010')
    d_ej, legend2 = generate_x_y_dose_bfield(df, 'EJ-204', orientations[0], c, f, l='1010')
    plt.errorbar(d_bcf10[0], d_bcf10[1]/d_bcf10[1][0], yerr=np.sqrt(d_bcf10[2]**2 + d_bcf10[2][0]**2), linestyle=lin[j], color=colors[i], label='BCF-10, ' + c2[j])
    plt.errorbar(d_bcf60[0], d_bcf60[1] / d_bcf60[1][0], yerr=np.sqrt(d_bcf60[2]**2 + d_bcf60[2][0]**2), linestyle=lin[j], color=colors[i + 1], label='BCF-60, ' + c2[j])
    plt.errorbar(d_ej[0], d_ej[1]/d_ej[1][0], yerr=np.sqrt(d_ej[2]**2 + d_ej[2][0]**2), linestyle=lin[j], color=colors[i+2], label='EJ-204, ' + c2[j])
    #plt.ylim([0.975, 1.07])
    plt.ylabel('Dose / Dose (0 T)')
  #  plt.xlabel('Magnetic field')
    j = j + 1
#plt.legend(framealpha=1, bbox_to_anchor=(0.945, 0.8))
plt.legend(framealpha=1, frameon=True)
j = 0

plt.subplot(212)
d_bcf10_r, legend = generate_x_y_dose_bfield(df, 'BCF-10', orientations[0], 'Reverse', f, l='1010')
d_bcf60_r, legend1 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], 'Reverse', f, l='1010')
d_ej_r, legend2 = generate_x_y_dose_bfield(df, 'EJ-204', orientations[0], 'Reverse', f, l='1010')
d_bcf10_f, legend = generate_x_y_dose_bfield(df, 'BCF-10', orientations[0], 'Forward', f, l='1010')
d_bcf60_f, legend1 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], 'Forward', f, l='1010')
d_ej_f, legend2 = generate_x_y_dose_bfield(df, 'EJ-204', orientations[0], 'Forward', f, l='1010')
plt.plot(d_bcf10[0], ( d_bcf10_f[1] / d_bcf10_f[1][0]- d_bcf10_r[1]/d_bcf10_r[1][0]) /(d_bcf10_r[1]/d_bcf10_r[1][0]), linestyle=lin[j], color=colors[0], label='BCF-10')
plt.plot(d_bcf60[0], (d_bcf60_f[1]/d_bcf60_f[1][0] - d_bcf60_r[1] / d_bcf60_r[1][0]) /(d_bcf60_r[1] / d_bcf60_r[1][0]), linestyle=lin[j], color=colors[1], label='BCF-60')
plt.plot(d_bcf10[0], (d_ej_f[1]/d_ej_f[1][0] - d_ej_r[1] / d_ej_r[1][0]) /(d_ej_r[1] / d_ej_r[1][0]), linestyle=lin[j], color=colors[2], label='EJ-204')
plt.ylabel('Relative difference \n between orientations')
plt.xlabel('Magnetic field [T]')
#plt.ylim([-0.01, 0.026])


plt.tight_layout()
plt.legend(framealpha=1, frameon=True)
plt.show()

# All det


fig, axes = plt.subplots(nrows=2, ncols=1,  gridspec_kw={'height_ratios': [2, 1]})
f=fields[2]
c2 = ['e- \u2192 stem', 'e- \u2192 tip']
plt.subplot(211)
j = 0
i = 0
for c in currents:
    d_bcf10, legend = generate_x_y_dose_bfield(df, 'BCF-10', orientations[0], c, f, l='1010')
    d_bcf60, legend1 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], c, f, l='1010')
    d_ej, legend2 = generate_x_y_dose_bfield(df, 'EJ-204', orientations[0], c, f, l='1010')
    d_bcf60_l, legend3 = generate_x_y_dose_bfield(df, 'BCF-60 Lee filter', orientations[0], c, f, l='1010')
    d_m, legend4 = generate_x_y_dose_bfield(df, 'Medscint 197-2', orientations[0], c, f, l='0110')
    plt.errorbar(d_bcf10[0], d_bcf10[1]/d_bcf10[1][0], yerr=np.sqrt(d_bcf10[2]**2 + d_bcf10[2][0]**2), linestyle=lin[j], color=colors[i], label='BCF-10, ' + c2[j])
    plt.errorbar(d_bcf60[0], d_bcf60[1] / d_bcf60[1][0], yerr=np.sqrt(d_bcf60[2]**2 + d_bcf60[2][0]**2), linestyle=lin[j], color=colors[i + 1], label='BCF-60, ' + c2[j])
    plt.errorbar(d_ej[0], d_ej[1]/d_ej[1][0], yerr=np.sqrt(d_ej[2]**2 + d_ej[2][0]**2), linestyle=lin[j], color=colors[i+2], label='EJ-204, ' + c2[j])
    plt.errorbar(d_bcf60_l[0], d_bcf60_l[1] / d_bcf60_l[1][0], yerr=np.sqrt(d_bcf60_l[2]**2 + d_bcf60_l[2][0]**2), linestyle=lin[j], color=colors[i + 3], label='BCF-60 Lee, ' + c2[j])
    plt.errorbar(d_m[0], d_m[1] / d_m[1][0], yerr=np.sqrt(d_m[2] ** 2 + d_m[2][0] ** 2), color=colors[i + 4], linestyle=lin[j], label='Medscint, ' + c2[j])
    plt.ylabel('Dose / Dose (0 T)')

    j = j + 1
plt.legend(framealpha=1, bbox_to_anchor=(0.99, 0.9))
#plt.legend(framealpha=1, bbox_to_anchor=(1, 0.5))
j = 0

plt.subplot(212)
d_bcf10_r, legend = generate_x_y_dose_bfield(df, 'BCF-10', orientations[0], 'Reverse', f, l='1010')
d_bcf60_r, legend1 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], 'Reverse', f, l='1010')
d_ej_r, legend2 = generate_x_y_dose_bfield(df, 'EJ-204', orientations[0], 'Reverse', f, l='1010')
d_bcf60_l_r, legend1 = generate_x_y_dose_bfield(df, 'BCF-60 Lee filter', orientations[0], 'Reverse', f, l='1010')
d_m_r, legend2 = generate_x_y_dose_bfield(df, 'Medscint 197-2', orientations[0], 'Reverse', f, l='1010')
d_bcf10_f, legend = generate_x_y_dose_bfield(df, 'BCF-10', orientations[0], 'Forward', f, l='1010')
d_bcf60_f, legend1 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], 'Forward', f, l='1010')
d_ej_f, legend2 = generate_x_y_dose_bfield(df, 'EJ-204', orientations[0], 'Forward', f, l='1010')
d_bcf60_l_f, legend1 = generate_x_y_dose_bfield(df, 'BCF-60 Lee filter', orientations[0], 'Forward', f, l='1010')
d_m_f, legend2 = generate_x_y_dose_bfield(df, 'Medscint 197-2', orientations[0], 'Forward', f, l='1010')

plt.plot(d_bcf10[0], ( d_bcf10_f[1] / d_bcf10_f[1][0]- d_bcf10_r[1]/d_bcf10_r[1][0]) /(d_bcf10_r[1]/d_bcf10_r[1][0]), linestyle='--', color=colors[0], label='BCF-10')
plt.plot(d_bcf60[0], (d_bcf60_f[1]/d_bcf60_f[1][0] - d_bcf60_r[1] / d_bcf60_r[1][0]) /(d_bcf60_r[1] / d_bcf60_r[1][0]), linestyle='--', color=colors[1], label='BCF-60')
plt.plot(d_bcf10[0], (d_ej_f[1]/d_ej_f[1][0] - d_ej_r[1] / d_ej_r[1][0]) /(d_ej_r[1] / d_ej_r[1][0]), linestyle='--', color=colors[2], label='EJ-204')
plt.plot(d_bcf60[0], (d_bcf60_l_f[1]/d_bcf60_l_f[1][0] - d_bcf60_l_r[1] / d_bcf60_l_r[1][0]) /(d_bcf60_l_r[1] / d_bcf60_l_r[1][0]), linestyle='--', color=colors[3], label='BCF-60 Lee')
plt.plot(d_bcf10[0], (d_m_f[1]/d_m_f[1][0] - d_m_r[1] / d_m_r[1][0]) /(d_m_r[1] / d_m_r[1][0]), linestyle='--', color=colors[4], label='Medscint')
plt.ylabel('Relative difference \n between orientations')
plt.xlabel('Magnetic field [T]')
#plt.ylim([-0.01, 0.026])

plt.tight_layout()
plt.legend(framealpha=1, bbox_to_anchor=(0.99, 0.8))
plt.show()

#lee filter

fig, axes = plt.subplots(nrows=2, ncols=1,  gridspec_kw={'height_ratios': [2, 1]})
f=fields[2]
c2 = ['e- \u2192 stem', 'e- \u2192 tip']
plt.subplot(211)
j = 0
i = 0
for c in currents:
    d_bcf60, legend1 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], c, f, l='1010')
    d_bcf60_l, legend3 = generate_x_y_dose_bfield(df, 'BCF-60 Lee filter', orientations[0], c, f, l='1010')
    plt.errorbar(d_bcf60[0], d_bcf60[1] / d_bcf60[1][0], yerr=np.sqrt(d_bcf60[2]**2 + d_bcf60[2][0]**2), linestyle=lin[j], color=colors[i + 1], label='BCF-60, ' + c2[j])
    plt.errorbar(d_bcf60_l[0], d_bcf60_l[1] / d_bcf60_l[1][0], yerr=np.sqrt(d_bcf60_l[2]**2 + d_bcf60_l[2][0]**2), linestyle=lin[j], color=colors[i + 3], label='BCF-60 Lee, ' + c2[j])
    plt.ylabel('Dose / Dose (0 T)')

    j = j + 1
plt.legend(framealpha=1, frameon=True)
#plt.legend(framealpha=1, bbox_to_anchor=(1, 0.5))
j = 0

plt.subplot(212)
d_bcf60_r, legend1 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], 'Reverse', f, l='1010')
d_bcf60_l_r, legend1 = generate_x_y_dose_bfield(df, 'BCF-60 Lee filter', orientations[0], 'Reverse', f, l='1010')
d_bcf60_f, legend1 = generate_x_y_dose_bfield(df, 'BCF-60', orientations[0], 'Forward', f, l='1010')
d_bcf60_l_f, legend1 = generate_x_y_dose_bfield(df, 'BCF-60 Lee filter', orientations[0], 'Forward', f, l='1010')


plt.plot(d_bcf60[0], (d_bcf60_f[1]/d_bcf60_f[1][0] - d_bcf60_r[1] / d_bcf60_r[1][0]) /(d_bcf60_r[1] / d_bcf60_r[1][0]), linestyle='--', color=colors[1], label='BCF-60')
plt.plot(d_bcf60[0], (d_bcf60_l_f[1]/d_bcf60_l_f[1][0] - d_bcf60_l_r[1] / d_bcf60_l_r[1][0]) /(d_bcf60_l_r[1] / d_bcf60_l_r[1][0]), linestyle='--', color=colors[3], label='BCF-60 Lee')
plt.ylabel('Relative difference \n between orientations')
plt.xlabel('Magnetic field [T]')
#plt.ylim([-0.01, 0.026])

plt.tight_layout()
plt.legend(framealpha=1, frameon=True)
plt.show()
'''

# Medscint detector : angular dependency

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4.5))
temp = 130
for f in fields:
    temp = temp + 1
    plt.subplot(temp)
    j = 0
    i = 0
    for o in orientations:
        for c in currents:
            d_m, legend = generate_x_y_dose_bfield(df, 'Medscint 197-2', o, c, f, l='0110')
            plt.errorbar(d_m[0], d_m[1]/d_m[1][0], yerr=np.sqrt(d_m[2]**2 + d_m[2][0]**2), label=legend)
            plt.ylabel('Dose / Dose (0 T)')
            plt.xlabel('Magnetic field')
            plt.ylim([0.955, 1.04])
            print(f, o, c, d_m[0], d_m[1] / d_m[1][0], '\n')
            j += 1
    i += 1
    tit = str(f) + ' cm'
    plt.title(tit)
plt.legend(framealpha=1, frameon=True)
plt.tight_layout()
plt.show() 


j = 0
i = 0
for c in currents:

    d_ej, legend2 = generate_x_y_dose_bfield(df, 'EJ-204', orientations[0], c, f, l='1010')
    plt.errorbar(d_ej[0], d_ej[1]/d_ej[1][0], yerr=np.sqrt(d_ej[2]**2 + d_ej[2][0]**2), linestyle=lin[j], color=colors[i+2], label='EJ-204, ' + c2[j])

    #plt.ylim([0.975, 1.07])
    plt.ylabel('Dose ')
    plt.xlabel('Magnetic field')
    j = j + 1
#plt.legend(framealpha=1, bbox_to_anchor=(0.945, 0.8))
plt.legend(framealpha=1, frameon=True)
plt.tight_layout()
plt.show()

'''