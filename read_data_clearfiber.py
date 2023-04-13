import os
import matplotlib.pyplot as plt
from xarray import open_dataarray
import numpy as np

folder = os.path.expanduser('/Users/yun/Library/CloudStorage/OneDrive-Medscintinc/SpectraOutput/')
files = os.listdir('/Users/yun/Library/CloudStorage/OneDrive-Medscintinc/SpectraOutput/')

Resultat = []
spectra_para_0T = []
Resultat2 = []
files_no_0T_para = []
files_no_05T_para = []
files_no_15T_para = []
files_no_0T_perp = []
files_no_05T_perp  = []
files_no_15T_perp  = []

# for the clear fiber
filenames_0T_para = ['20220926-153523.spectra', '20220926-153600.spectra', '20220926-153649.spectra',
                     '20220926-153727.spectra', '20220926-153806.spectra']
filenames_0T_perp = ['20220926-160539.spectra', '20220926-160614.spectra', '20220926-160644.spectra',
                     '20220926-160715.spectra', '20220926-160746.spectra']
filenames_05T_para = ['20220926-154526.spectra', '20220926-154600.spectra', '20220926-154634.spectra',
                      '20220926-154713.spectra', '20220926-154753.spectra']
filenames_05T_perp = ['20220926-160941.spectra', '20220926-161017.spectra', '20220926-161049.spectra',
                      '20220926-161124.spectra', '20220926-161159.spectra']
filenames_15T_para = ['20220926-155146.spectra', '20220926-155221.spectra', '20220926-155249.spectra',
                      '20220926-155319.spectra', '20220926-155350.spectra']
filenames_15T_perp = ['20220926-161520.spectra', '20220926-161558.spectra', '20220926-161635.spectra',
                      '20220926-161713.spectra', '20220926-161751.spectra']

for f in range(len(files)):
    if files[f] in filenames_0T_para:
        files_no_0T_para.append(f)
    elif files[f] in filenames_05T_para:
        files_no_05T_para.append(f)
    elif files[f] in filenames_15T_para:
        files_no_15T_para.append(f)
    elif files[f] in filenames_0T_perp:
        files_no_0T_perp.append(f)
    elif files[f] in filenames_05T_perp:
        files_no_05T_perp.append(f)
    elif files[f] in filenames_15T_perp:
        files_no_15T_perp.append(f)

for i in files_no_0T_para:  # for 0T
    analysed_file = files[i]
    filepath = ''.join([folder, analysed_file])
    with open_dataarray(filepath, engine='h5netcdf') as spectra:
        # print(spectra)                        #raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')  # summed data in time for channel 1
    Resultat.append(spectra)
spectra_para_0T = np.mean(Resultat, axis=0)


Resultat = []
# for i in range(len(files)):
for i in files_no_05T_para:  # for 0.5 T
    analysed_file = files[i]
    filepath = ''.join([folder, analysed_file])

    with open_dataarray(filepath, engine='h5netcdf') as spectra:
        # print(spectra)                        #raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')  # summed data in time for channel 1
    Resultat.append(spectra)
spectra_para_05T = np.mean(Resultat, axis=0)

Resultat = []

for i in files_no_15T_para:  # for 1.5 T
    analysed_file = files[i]
    filepath = ''.join([folder, analysed_file])

    with open_dataarray(filepath, engine='h5netcdf') as spectra:
        #  print(spectra)                        #raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')  # summed data in time for channel 1
    Resultat.append(spectra)
spectra_para_15T = np.mean(Resultat, axis=0)

Resultat = []
for i in files_no_0T_perp: #for 0T
    analysed_file = files[i]
    filepath = ''.join([folder, analysed_file])

    with open_dataarray(filepath, engine='h5netcdf') as spectra:
        print(spectra)                        #raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')             #summed data in time for channel 1
    Resultat.append(spectra)
spectra_perp_0T = np.mean(Resultat, axis=0)


Resultat = []
for i in files_no_05T_perp:  #for 0.5 T
    analysed_file = files[i]
    filepath = ''.join([folder, analysed_file])

    with open_dataarray(filepath, engine='h5netcdf') as spectra:
        print(spectra)                        #raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')             #summed data in time for channel 1
    Resultat.append(spectra)
spectra_perp_05T = np.mean(Resultat, axis=0)

Resultat = []
for i in files_no_15T_perp:  #for 1.5 T
    analysed_file = files[i]
    filepath = ''.join([folder, analysed_file])

    with open_dataarray(filepath, engine='h5netcdf') as spectra:
        print(spectra)                        #raw data
        spectra = spectra.isel(spectrum_or_temp=0, channel=0).sum(dim='stack')             #summed data in time for channel 1
    Resultat.append(spectra)
spectra_perp_15T = np.mean(Resultat, axis=0)


#for i in range(len(files)):
 #      print(files[i],i)


print('parallel 0T: ', sum(spectra_para_0T/np.sum(spectra_para_0T)))
print('0.5T: ', sum(spectra_para_05T/np.sum(spectra_para_0T)))
print('1.5T: ', sum(spectra_para_15T/np.sum(spectra_para_0T)))
print('perpendicular 0T: ', sum(spectra_perp_0T/np.sum(spectra_perp_0T)))
print('0.5T: ', sum(spectra_perp_05T/np.sum(spectra_perp_0T)))
print('1.5T: ', sum(spectra_perp_15T/np.sum(spectra_perp_0T)))


plt.subplot(121)
plt.plot(spectra_para_0T/np.sum(spectra_para_0T), ':', label='0 T')
plt.plot(spectra_para_05T/np.sum(spectra_para_0T), '--', label='0.5 T')
plt.plot(spectra_para_15T/np.sum(spectra_para_0T), '-.', label='1.5 T')
plt.ylabel('Intensity(B)/Area under the curve(0T)')
plt.xlabel('--')
plt.legend(framealpha=1, frameon=True)
#plt.legend(framealpha=1, frameon=True, loc='center right', bbox_to_anchor=(1.25, 0.5))
plt.title('Parallel')

plt.subplot(122)
plt.plot(spectra_perp_0T/np.sum(spectra_perp_0T), ':', label='0 T')
plt.plot(spectra_perp_05T/np.sum(spectra_perp_0T), '--', label='0.5 T')
plt.plot(spectra_perp_15T/np.sum(spectra_perp_0T), '-.', label='1.5 T')
plt.xlabel('--')
plt.legend(framealpha=1, frameon=True)
plt.title('Perpendicular')
plt.tight_layout()
plt.show()

print('0T: ', sum(spectra_para_0T/np.sum(spectra_para_0T)))