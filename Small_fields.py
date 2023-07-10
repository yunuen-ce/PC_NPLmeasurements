import pandas as pd
from numpy import linalg, ndarray, vstack
from base64 import decodebytes as b64_decode
from json import loads
import matplotlib.pyplot as plt
import glob
import numpy as np
from xarray import open_dataarray
import os
from functions import *

Mesures_bcf10_reverse = []
Mesures_bcf10_forward = []
Mesures_bcf60_reverse = []
Mesures_bcf60_forward = []
Mesures_ej204_reverse = []
Mesures_ej204_forward = []
Mesures_bcf60_lee_reverse = []
Mesures_bcf60_lee_forward = []
Mesures_medscint_reverse = []
Mesures_medscint_forward = []
Resultat = []
Resultat_0T = []
# 3 field sizes, 5 scintillators, 42 = 3 measurements * 2 orientations * 11 bfields,
Data = np.zeros([5, 3, 42, 137])
Mean_dose = np.zeros([5, 3, 12])
Mean_fluo = np.zeros([5, 3, 12])
Mean_ckov = np.zeros([5, 3, 12])
std_dose = np.zeros([5, 3, 12])
std_fluo = np.zeros([5, 3, 12])
std_ckov = np.zeros([5, 3, 12])
Mean_spectrum = np.zeros([5, 3, 12, 137])

bfields = [0, 0.2, 0.35, 0.5, 1, 1.5]
field_size = [0.5, 1, 7]

colors = ['lightskyblue', 'mediumseagreen', 'black', 'purple', 'grey', 'darkorange', 'plum', 'black',
          'royalblue', 'salmon']

colors_blue = [['navy', 'blue', 'royalblue', 'lightskyblue', 'deepskyblue'],
               ['darkgreen', 'forestgreen', 'green', 'lightgreen', 'mediumseagreen' ]]
lin = ['-', ':', '--', '-', '--', ':']
mark = ['*', 'o', 'd', 's', '^']
scintillators = ['bcf10', 'bcf60', 'ej204', 'bcf60-Lee', 'Medscint']
scint_label = ['BCF-10', 'BCF-60', 'EJ-204', 'BCF-60 Lee filter', 'Medscint']
path = '/Users/yunuen/Library/CloudStorage/OneDrive-Medscintinc/SpectraOutput'
folder = os.path.expanduser(path)
files = os.listdir(path)
channel = [0, 1, 1, 0, 1]

# Read data
# For each field size : 3 measures for 5 scintillators

filenames_02T_reverse = [[['20220921-145746.spectra', '20220921-145819.spectra', '20220921-145847.spectra'],
                          ['20220927-102544.spectra', '20220927-102639.spectra', '20220927-102716.spectra'],
                          ['20220922-142856.spectra', '20220922-142928.spectra', '20220922-143003.spectra'],
                          ['20220927-153349.spectra', '20220927-153415.spectra', '20220927-153443.spectra'],
                          ['20220923-111321.spectra', '20220923-111356.spectra', '20220923-111429.spectra']],
                         # 1 x 1
                         [['20220921-150252.spectra', '20220921-150327.spectra', '20220921-150403.spectra'],
                          ['20220927-103004.spectra', '20220927-103034.spectra', '20220927-103116.spectra'],
                          ['20220922-143303.spectra', '20220922-143340.spectra', '20220922-143426.spectra'],
                          ['20220927-153809.spectra', '20220927-153835.spectra', '20220927-153909.spectra'],
                          ['20220923-111716.spectra', '20220923-111752.spectra', '20220923-111827.spectra']],
                         # 7 x 7
                         [['20220921-150444.spectra', '20220921-150509.spectra', '20220921-150538.spectra'],
                          ['20220927-103143.spectra', '20220927-103211.spectra', '20220927-103240.spectra'],
                          ['20220922-143502.spectra', '20220922-143535.spectra', '20220922-143604.spectra'],
                          ['20220927-153942.spectra', '20220927-154009.spectra', '20220927-154039.spectra'],
                          ['20220923-111908.spectra', '20220923-111941.spectra', '20220923-112016.spectra']]]
filenames_035T_reverse = [[['20220921-151008.spectra', '20220921-151043.spectra', '20220921-151120.spectra'],
                           ['20220927-103456.spectra', '20220927-103523.spectra', '20220927-103551.spectra'],
                           ['20220922-144201.spectra', '20220922-144235.spectra', '20220922-144304.spectra'],
                           ['20220927-154341.spectra', '20220927-154408.spectra', '20220927-154437.spectra'],
                           ['20220923-112250.spectra', '20220923-112406.spectra', '20220923-112720.spectra']],
                          # 1 x 1
                          [['20220921-151427.spectra', '20220921-151504.spectra', '20220921-151544.spectra'],
                           ['20220927-103938.spectra', '20220927-104006.spectra', '20220927-104036.spectra'],
                           ['20220922-144837.spectra', '20220922-144911.spectra', '20220922-145008.spectra'],
                           ['20220927-154750.spectra', '20220927-154816.spectra', '20220927-154841.spectra'],
                           ['20220923-113034.spectra', '20220923-113106.spectra', '20220923-113138.spectra']],
                          # 7 x 7
                          [['20220921-151611.spectra', '20220921-151643.spectra', '20220921-151717.spectra'],
                           ['20220927-104113.spectra', '20220927-104154.spectra', '20220927-104222.spectra'],
                           ['20220922-145052.spectra', '20220922-145123.spectra', '20220922-145200.spectra'],
                           ['20220927-154917.spectra', '20220927-154944.spectra', '20220927-155011.spectra'],
                           ['20220923-113218.spectra', '20220923-113249.spectra', '20220923-113320.spectra']]]
filenames_05T_reverse = [[['20220921-151940.spectra', '20220921-152012.spectra', '20220921-152057.spectra'],
                          ['20220927-104455.spectra', '20220927-104522.spectra', '20220927-104551.spectra'],
                          ['20220922-145958.spectra', '20220922-150045.spectra', '20220922-150118.spectra'],
                          ['20220927-155346.spectra', '20220927-155412.spectra', '20220927-155442.spectra'],
                          ['20220923-113542.spectra', '20220923-113611.spectra', '20220923-113645.spectra']],
                         # 1 x 1
                         [['20220921-152459.spectra', '20220921-152533.spectra', '20220921-152607.spectra'],
                          ['20220927-105405.spectra', '20220927-105448.spectra', '20220927-105516.spectra'],
                          ['20220922-150505.spectra', '20220922-150544.spectra', '20220922-150618.spectra'],
                          ['20220927-155827.spectra', '20220927-155858.spectra', '20220927-155931.spectra'],
                          ['20220923-113950.spectra', '20220923-114020.spectra', '20220923-114054.spectra']],
                         # 7 x 7
                         [['20220921-152639.spectra', '20220921-152710.spectra', '20220921-152750.spectra'],
                          ['20220927-105553.spectra', '20220927-105621.spectra', '20220927-105650.spectra'],
                          ['20220922-150658.spectra', '20220922-150726.spectra', '20220922-150753.spectra'],
                          ['20220927-160001.spectra', '20220927-160025.spectra', '20220927-160051.spectra'],
                          ['20220923-114238.spectra', '20220923-114333.spectra', '20220923-114414.spectra']]]
filenames_1T_reverse = [[['20220921-153321.spectra', '20220921-153405.spectra', '20220921-153434.spectra'],
                         ['20220927-105952.spectra', '20220927-110021.spectra', '20220927-110103.spectra'],
                         ['20220922-151226.spectra', '20220922-151259.spectra', '20220922-151338.spectra'],
                         ['20220927-160354.spectra', '20220927-160419.spectra', '20220927-160445.spectra'],
                         ['20220923-114759.spectra', '20220923-114831.spectra', '20220923-114905.spectra']],
                        # 1 x 1
                        [['20220921-153700.spectra', '20220921-153733.spectra', '20220921-153805.spectra'],
                         ['20220927-110343.spectra', '20220927-110410.spectra', '20220927-110435.spectra'],
                         ['20220922-151647.spectra', '20220922-151718.spectra', '20220922-151800.spectra'],
                         ['20220927-160733.spectra', '20220927-160802.spectra', '20220927-160829.spectra'],
                         ['20220923-115202.spectra', '20220923-115236.spectra', '20220923-115307.spectra']],
                        # 7 x 7
                        [['20220921-153845.spectra', '20220921-153925.spectra', '20220921-153925.spectra'],
                         ['20220927-110519.spectra', '20220927-110552.spectra', '20220927-110622.spectra'],
                         ['20220922-151843.spectra', '20220922-151916.spectra', '20220922-151943.spectra'],
                         ['20220927-160906.spectra', '20220927-160931.spectra', '20220927-160956.spectra'],
                         ['20220923-115351.spectra', '20220923-115424.spectra', '20220923-115500.spectra']]]
filenames_15T_reverse = [[['20220921-154502.spectra', '20220921-154540.spectra', '20220921-154613.spectra'],
                          ['20220927-110954.spectra', '20220927-111022.spectra', '20220927-111050.spectra'],
                          ['20220922-152329.spectra', '20220922-152359.spectra', '20220922-152431.spectra'],
                          ['20220927-161402.spectra', '20220927-161428.spectra', '20220927-161502.spectra'],
                          ['20220923-115914.spectra', '20220923-115946.spectra', '20220923-120020.spectra']],
                         # 1 x 1
                         [['20220921-154944.spectra', '20220921-155018.spectra', '20220921-155049.spectra'],
                          ['20220927-111420.spectra', '20220927-111450.spectra', '20220927-111518.spectra'],
                          ['20220922-152734.spectra', '20220922-152804.spectra', '20220922-152834.spectra'],
                          ['20220927-161759.spectra', '20220927-161828.spectra', '20220927-161853.spectra'],
                          ['20220923-120246.spectra', '20220923-120321.spectra', '20220923-120355.spectra']],
                         # 7 x 7
                         [['20220921-155128.spectra', '20220921-155233.spectra', '20220921-155308.spectra'],
                          ['20220927-111558.spectra', '20220927-111628.spectra', '20220927-111703.spectra'],
                          ['20220922-152910.spectra', '20220922-152940.spectra', '20220922-153009.spectra'],
                          ['20220927-161940.spectra', '20220927-162009.spectra', '20220927-162037.spectra'],
                          ['20220923-120435.spectra', '20220923-120508.spectra', '20220923-120541.spectra']]]
filenames_0T_reverse = [[['20220921-144826.spectra', '20220921-144859.spectra', '20220921-144933.spectra',
                          '20220921-155840.spectra', '20220921-155911.spectra', '20220921-155945.spectra'],
                         ['20220927-100658.spectra', '20220927-100732.spectra', '20220927-100800.spectra',
                          '20220927-112054.spectra', '20220927-112121.spectra', '20220927-112213.spectra'],
                         ['20220922-140759.spectra', '20220922-140831.spectra', '20220922-140902.spectra',
                          '20220922-153547.spectra', '20220922-153617.spectra', '20220922-153713.spectra'],
                         ['20220927-151644.spectra', '20220927-151709.spectra', '20220927-151736.spectra',
                          '20220927-162449.spectra', '20220927-162514.spectra', '20220927-162539.spectra'],
                         ['20220923-105750.spectra', '20220923-105823.spectra', '20220923-105858.spectra',
                          '20220923-121056.spectra', '20220923-121129.spectra', '20220923-121204.spectra']],
                        # 1 x 1
                        [['20220921-144251.spectra', '20220921-144327.spectra', '20220921-144405.spectra',
                          '20220921-160213.spectra', '20220921-160254.spectra', '20220921-160325.spectra'],
                         ['20220927-101107.spectra', '20220927-101138.spectra', '20220927-101211.spectra',
                          '20220927-112419.spectra', '20220927-112449.spectra', '20220927-112517.spectra'],
                         ['20220922-141143.spectra', '20220922-141215.spectra', '20220922-141247.spectra',
                          '20220922-154125.spectra', '20220922-154152.spectra', '20220922-154222.spectra'],
                         ['20220927-152058.spectra', '20220927-152125.spectra', '20220927-152150.spectra',
                          '20220927-162832.spectra', '20220927-162857.spectra', '20220927-162923.spectra'],
                         ['20220923-110212.spectra', '20220923-110248.spectra', '20220923-110324.spectra',
                          '20220923-121420.spectra', '20220923-121503.spectra', '20220923-121537.spectra']],
                        # 7 x 7
                        [['20220921-145005.spectra', '20220921-145038.spectra', '20220921-145115.spectra',
                          '20220921-160423.spectra', '20220921-160454.spectra', '20220921-160524.spectra'],
                         ['20220927-101247.spectra', '20220927-101316.spectra', '20220927-101344.spectra',
                          '20220927-112556.spectra', '20220927-112623.spectra', '20220927-112649.spectra'],
                         ['20220922-141408.spectra', '20220922-141442.spectra', '20220922-141517.spectra',
                          '20220922-154301.spectra', '20220922-154359.spectra', '20220922-154429.spectra'],
                         ['20220927-152227.spectra', '20220927-152301.spectra', '20220927-152336.spectra',
                          '20220927-163000.spectra', '20220927-163025.spectra', '20220927-163051.spectra'],
                         ['20220923-110407.spectra', '20220923-110440.spectra', '20220923-110513.spectra',
                          '20220923-121616.spectra', '20220923-121653.spectra', '20220923-121727.spectra']]]
# Forward
filenames_0T_forward = [[['20220922-101447.spectra', '20220922-101522.spectra', '20220922-101603.spectra',
                          '20220922-112945.spectra', '20220922-113019.spectra', '20220922-113053.spectra'],
                         ['20220927-112054.spectra', '20220927-112121.spectra', '20220927-112213.spectra',
                          '20220927-123107.spectra', '20220927-123139.spectra', '20220927-123212.spectra'],
                         ['20220922-124315.spectra', '20220922-124350.spectra', '20220922-124431.spectra',
                          '20220922-140759.spectra', '20220922-140831.spectra', '20220922-140902.spectra'],
                         ['20220927-141015.spectra', '20220927-141043.spectra', '20220927-141117.spectra',
                          '20220927-151644.spectra', '20220927-151709.spectra', '20220927-151736.spectra'],
                         ['20220923-130038.spectra', '20220923-130114.spectra', '20220923-130149.spectra',
                          '20220923-141336.spectra', '20220923-141408.spectra', '20220923-141437.spectra']],
                        # 1x1
                        [['20220922-101856.spectra', '20220922-101930.spectra', '20220922-102004.spectra',
                          '20220922-113334.spectra', '20220922-113405.spectra', '20220922-113445.spectra'],
                         ['20220927-112419.spectra', '20220927-112449.spectra', '20220927-112517.spectra',
                          '20220927-123506.spectra', '20220927-123545.spectra', '20220927-123616.spectra'],
                         ['20220922-124836.spectra', '20220922-124907.spectra', '20220922-124940.spectra',
                          '20220922-141143.spectra', '20220922-141215.spectra', '20220922-141247.spectra'],
                         ['20220927-141405.spectra', '20220927-141432.spectra', '20220927-141459.spectra',
                          '20220927-152058.spectra', '20220927-152125.spectra', '20220927-152150.spectra'],
                         ['20220923-130457.spectra', '20220923-130530.spectra', '20220923-130601.spectra',
                          '20220923-141850.spectra', '20220923-141921.spectra', '20220923-142006.spectra']],
                        # 7x7
                        [['20220922-102039.spectra', '20220922-102112.spectra', '20220922-102146.spectra',
                          '20220922-113527.spectra', '20220922-113743.spectra', '20220922-113816.spectra'],
                         ['20220927-112556.spectra', '20220927-112623.spectra', '20220927-112649.spectra',
                          '20220927-123652.spectra', '20220927-123724.spectra', '20220927-123826.spectra'],
                         ['20220922-125020.spectra', '20220922-125056.spectra', '20220922-125129.spectra',
                          '20220922-141408.spectra', '20220922-141442.spectra', '20220922-141517.spectra'],
                         ['20220927-141540.spectra', '20220927-141620.spectra', '20220927-141644.spectra',
                          '20220927-152227.spectra', '20220927-152301.spectra', '20220927-152336.spectra'],
                         ['20220923-130641.spectra', '20220923-130744.spectra', '20220923-130829.spectra',
                          '20220923-142044.spectra', '20220923-142120.spectra', '20220923-142152.spectra']]]
filenames_02T_forward = [[['20220922-103030.spectra', '20220922-103103.spectra', '20220922-103203.spectra'],
                          ['20220927-113733.spectra', '20220927-113813.spectra', '20220927-113847.spectra'],
                          ['20220922-125956.spectra', '20220922-130030.spectra', '20220922-130105.spectra'],
                          ['20220927-142055.spectra', '20220927-142123.spectra', '20220927-142150.spectra'],
                          ['20220923-131750.spectra', '20220923-131822.spectra', '20220923-131856.spectra']],
                         # 1x1
                         [['20220922-103601.spectra', '20220922-103632.spectra', '20220922-103708.spectra'],
                          ['20220927-114130.spectra', '20220927-114158.spectra', '20220927-114231.spectra'],
                          ['20220922-130435.spectra', '20220922-130507.spectra', '20220922-130533.spectra'],
                          ['20220927-142417.spectra', '20220927-142443.spectra', '20220927-142511.spectra'],
                          ['20220923-132141.spectra', '20220923-132212.spectra', '20220923-132244.spectra']],
                         # 7x7
                         [['20220922-103743.spectra', '20220922-103813.spectra', '20220922-103845.spectra'],
                          ['20220927-114308.spectra', '20220927-114340.spectra', '20220927-114410.spectra'],
                          ['20220922-130647.spectra', '20220922-130727.spectra', '20220922-130802.spectra'],
                          ['20220927-142548.spectra', '20220927-142614.spectra', '20220927-142642.spectra'],
                          ['20220923-132323.spectra', '20220923-132356.spectra', '20220923-132429.spectra']]]
filenames_035T_forward = [[['20220922-104109.spectra', '20220922-104140.spectra', '20220922-104211.spectra'],
                           ['20220927-114701.spectra', '20220927-114732.spectra', '20220927-114804.spectra'],
                           ['20220922-131220.spectra', '20220922-131251.spectra', '20220922-131322.spectra'],
                           ['20220927-144431.spectra', '20220927-144457.spectra', '20220927-144526.spectra'],
                           ['20220923-132704.spectra', '20220923-132744.spectra', '20220923-132818.spectra']],
                          # 1x1
                          [['20220922-104417.spectra', '20220922-104448.spectra', '20220922-104520.spectra'],
                           ['20220927-115049.spectra', '20220927-115121.spectra', '20220927-115149.spectra'],
                           ['20220922-131707.spectra', '20220922-131739.spectra', '20220922-131811.spectra'],
                           ['20220927-144828.spectra', '20220927-144854.spectra', '20220927-144919.spectra'],
                           ['20220923-133126.spectra', '20220923-133201.spectra', '20220923-133237.spectra']],
                          # 7x7
                          [['20220922-104559.spectra', '20220922-104633.spectra', '20220922-104718.spectra'],
                           ['20220927-115231.spectra', '20220927-115303.spectra', '20220927-115333.spectra'],
                           ['20220922-131852.spectra', '20220922-131921.spectra', '20220922-131951.spectra'],
                           ['20220927-144958.spectra', '20220927-145025.spectra', '20220927-145058.spectra'],
                           ['20220923-133320.spectra', '20220923-133353.spectra', '20220923-133431.spectra']]]
filenames_05T_forward = [[['20220922-105425.spectra', '20220922-105500.spectra', '20220922-105535.spectra'],
                          ['20220927-115651.spectra', '20220927-115720.spectra', '20220927-115749.spectra'],
                          ['20220922-132614.spectra', '20220922-132645.spectra', '20220922-132717.spectra'],
                          ['20220927-143006.spectra', '20220927-143034.spectra', '20220927-143100.spectra'],
                          ['20220923-133751.spectra', '20220923-133845.spectra', '20220923-133921.spectra']],
                         # 1x1
                         [['20220922-105835.spectra', '20220922-105904.spectra', '20220922-105945.spectra'],
                          ['20220927-120039.spectra', '20220927-120113.spectra', '20220927-120139.spectra'],
                          ['20220922-133101.spectra', '20220922-133130.spectra', '20220922-133200.spectra'],
                          ['20220927-143640.spectra', '20220927-143705.spectra', '20220927-143730.spectra'],
                          ['20220923-134249.spectra', '20220923-134322.spectra', '20220923-134357.spectra']],
                         # 7x7
                         [['20220922-110026.spectra', '20220922-110059.spectra', '20220922-110126.spectra'],
                          ['20220927-120219.spectra', '20220927-120248.spectra', '20220927-120317.spectra'],
                          ['20220922-133307.spectra', '20220922-133335.spectra', '20220922-133407.spectra'],
                          ['20220927-143806.spectra', '20220927-143837.spectra', '20220927-143837.spectra'],
                          ['20220923-134436.spectra', '20220923-134513.spectra', '20220923-134547.spectra']]]
filenames_1T_forward = [[['20220922-110501.spectra', '20220922-110534.spectra', '20220922-110608.spectra'],
                         ['20220927-120719.spectra', '20220927-120753.spectra', '20220927-120821.spectra'],
                         ['20220922-133649.spectra', '20220922-133718.spectra', '20220922-133747.spectra'],
                         ['20220927-145455.spectra', '20220927-145520.spectra', '20220927-145616.spectra'],
                         ['20220923-134915.spectra', '20220923-134948.spectra', '20220923-135017.spectra']],
                        # 1x1
                        [['20220922-110953.spectra', '20220922-111102.spectra', '20220922-111136.spectra'],
                         ['20220927-121120.spectra', '20220927-121152.spectra', '20220927-121224.spectra'],
                         ['20220922-134219.spectra', '20220922-134250.spectra', '20220922-134321.spectra'],
                         ['20220927-145920.spectra', '20220927-145951.spectra', '20220927-150017.spectra'],
                         ['20220923-135338.spectra', '20220923-135411.spectra', '20220923-135443.spectra']],
                        # 7x7
                        [['20220922-111244.spectra', '20220922-111313.spectra', '20220922-111349.spectra'],
                         ['20220927-121301.spectra', '20220927-121330.spectra', '20220927-121401.spectra'],
                         ['20220922-134506.spectra', '20220922-134433.spectra', '20220922-134506.spectra'],
                         ['20220927-150051.spectra', '20220927-150118.spectra', '20220927-150145.spectra'],
                         ['20220923-135530.spectra', '20220923-135609.spectra', '20220923-135644.spectra']]]
filenames_15T_forward = [[['20220922-111715.spectra', '20220922-111751.spectra', '20220922-111826.spectra'],
                          ['20220927-121825.spectra', '20220927-121854.spectra', '20220927-121926.spectra'],
                          ['20220922-135241.spectra', '20220922-135313.spectra', '20220922-135344.spectra'],
                          ['20220927-150600.spectra', '20220927-150624.spectra', '20220927-150650.spectra'],
                          ['20220923-140042.spectra', '20220923-140114.spectra', '20220923-140146.spectra']],
                         # 1x1
                         [['20220922-112121.spectra', '20220922-112152.spectra', '20220922-112222.spectra'],
                          ['20220927-122154.spectra', '20220927-122250.spectra', '20220927-122324.spectra'],
                          ['20220922-135636.spectra', '20220922-135941.spectra', '20220922-140041.spectra'],
                          ['20220927-151006.spectra', '20220927-151032.spectra', '20220927-151058.spectra'],
                          ['20220923-140454.spectra', '20220923-140526.spectra', '20220923-140556.spectra']],
                         # 7x7
                         [['20220922-112258.spectra', '20220922-112327.spectra', '20220922-112355.spectra'],
                          ['20220927-122403.spectra', '20220927-122436.spectra', '20220927-122505.spectra'],
                          ['20220922-140151.spectra', '20220922-140229.spectra', '20220922-140304.spectra'],
                          ['20220927-151141.spectra', '20220927-151208.spectra', '20220927-151233.spectra'],
                          ['20220923-140638.spectra', '20220923-140707.spectra', '20220923-140753.spectra']]]

for s in range(len(scintillators)):
    for f in range(len(field_size)):
        tobeanalyse = [filenames_0T_reverse[f][s], filenames_02T_reverse[f][s], filenames_035T_reverse[f][s],
                       filenames_05T_reverse[f][s], filenames_1T_reverse[f][s], filenames_15T_reverse[f][s],
                       filenames_0T_forward[f][s], filenames_02T_forward[f][s], filenames_035T_forward[f][s],
                       filenames_05T_forward[f][s], filenames_1T_forward[f][s], filenames_15T_forward[f][s]]
        # print(field_size[f], scintillators[s], tobeanalyse)

        for item in tobeanalyse:
            print(scintillators[s], field_size[f],  item)    # per bfield
            for file in item:
                filepath = path + file
                print(file)
                Resultat.append(summed_spectra_old(filepath, channel[s]))
    print(scintillators[s], field_size[f])

n = 10 * 3 + 2 * 6

for s in range(len(scintillators)):

    for f in range(len(field_size)):
        #  print(s, f, (s*n*3) + (f * n), (s*n*3) + (f+1) * n)
        Data[s][f] = np.array(Resultat[(s * n * 3) + (f * n):(s * n * 3) + (f + 1) * n])

for s in range(len(scintillators)):
    for f in range(len(field_size)):
        if s == 0:  # bcf10
            scint_file = folder + '20220920-141652-scint.spectra'
            fluo_file = folder + '20220920-143043-fluo.spectra'
            ckov1_file = folder + '20220920-161025-cerenkov1.spectra'
            ckov2_file = folder + '20220920-161218-cerenkov2.spectra'
            ckov3_file = folder + '20220920-161924-cerenkov3.spectra'
            ckov4_file = folder + '20220920-162104-cerenkov4.spectra'
            dose_file = folder + '20220921-105631-normalization_all.spectra'
        elif s == 1:  # bcf60
            scint_file = folder + '20220927-085310-scint.spectra'
            fluo_file = folder + '20220927-090118-fluo.spectra'
            ckov1_file = folder + '20220927-092943-cerenkov1.spectra'
            ckov2_file = folder + '20220927-093133-cerenkov2.spectra'
            ckov3_file = folder + '20220927-093609-cerenkov3.spectra'
            ckov4_file = folder + '20220927-093753-cerenkov4.spectra'
            dose_file = folder + '20220927-095656-normalization_all.spectra'
        elif s == 2:  # ej-204
            scint_file = folder + '20220920-141652-scint.spectra'
            fluo_file = folder + '20220920-143043-fluo.spectra'
            ckov1_file = folder + '20220920-161025-cerenkov1.spectra'
            ckov2_file = folder + '20220920-161218-cerenkov2.spectra'
            ckov3_file = folder + '20220920-161924-cerenkov3.spectra'
            ckov4_file = folder + '20220920-162104-cerenkov4.spectra'
            dose_file = folder + '20220922-115954-normalization_all.spectra'
        elif s == 3:  # bcf60 lee
            scint_file = folder + '20220927-085310-scint.spectra'
            fluo_file = folder + '20220927-090118-fluo.spectra'
            ckov1_file = folder + '20220927-092943-cerenkov1.spectra'
            ckov2_file = folder + '20220927-093133-cerenkov2.spectra'
            ckov3_file = folder + '20220927-093609-cerenkov3.spectra'
            ckov4_file = folder + '20220927-093753-cerenkov4.spectra'
            dose_file = folder + '20220927-140521-normalization_all.spectra'
        else:
            scint_file = folder + '20220922-164231-scint.spectra'
            fluo_file = folder + '20220922-165214-fluo.spectra'
            ckov1_file = folder + '20220922-170040-cerenkov1.spectra'
            ckov2_file = folder + '20220922-170232-cerenkov2.spectra'
            ckov3_file = folder + '20220922-170618-cerenkov3.spectra'
            ckov4_file = folder + '20220922-170807-cerenkov4.spectra'
            dose_file = folder + '20220923-104901-normalization_all.spectra'

        scint = summed_spectra_old(scint_file, channel[s])
        fluo = summed_spectra_old(fluo_file, channel[s])
        ckov1 = summed_spectra_old(ckov1_file, channel[s])
        ckov2 = summed_spectra_old(ckov2_file, channel[s])
        ckov3 = summed_spectra_old(ckov3_file, channel[s])
        ckov4 = summed_spectra_old(ckov4_file, channel[s])
        ckovA = abs(ckov1 - ckov2)
        ckovB = abs(ckov3 - ckov4)
        dose = summed_spectra_old(dose_file, channel[s])
        calib_doseval = 500

        # To obtain the abundance

        R = compute_icm([scint, fluo, ckovA, ckovB])
        Ref = generate_weights(R, dose)
        Doses = []
        Fluo = []
        Ckov = []
        Ckov1 = []
        Ckov2 = []
        for i in range(len(Data[s][f])):
            Weight = generate_weights(R, Data[s][f][i])
            Doses.append(Weight[0] / Ref[0] * calib_doseval)
            Fluo.append(Weight[1] / Ref[1] * calib_doseval)
            Ckov.append((Weight[2] + Weight[3]) / (Ref[2] + Ref[3]) * calib_doseval)

        Moy_doses = []
        Moy_fluo = []
        Moy_ckov = []
        dev_doses = []
        dev_fluo = []
        dev_ckov = []
        Moy_spectrum = []

        for i in range(int(len(Data[s][f]) / 3)):

            if i == 0 or i == 7:
                Moy_doses.append(np.mean([Doses[i * 3], Doses[i * 3 + 1], Doses[i * 3 + 2], Doses[i * 3 + 3],
                                          Doses[i * 3 + 4], Doses[i * 3 + 5]]))
                Moy_fluo.append(
                    np.mean([Fluo[i * 3], Fluo[i * 3 + 1], Fluo[i * 3 + 2], Fluo[i * 3 + 3], Fluo[i * 3 + 4],
                             Fluo[i * 3 + 5]]))
                Moy_ckov.append(
                    np.mean([Ckov[i * 3], Ckov[i * 3 + 1], Ckov[i * 3 + 2], Ckov[i * 3 + 3], Ckov[i * 3 + 4],
                             Ckov[i * 3 + 5]]))
                dev_doses.append(np.std([Doses[i * 3], Doses[i * 3 + 1], Doses[i * 3 + 2], Doses[i * 3 + 3],
                                         Doses[i * 3 + 4], Doses[i * 3 + 5]]))
                dev_fluo.append(np.std([Fluo[i * 3], Fluo[i * 3 + 1], Fluo[i * 3 + 2], Fluo[i * 3 + 3], Fluo[i * 3 + 4],
                                        Fluo[i * 3 + 5]]))
                dev_ckov.append(np.std([Ckov[i * 3], Ckov[i * 3 + 1], Ckov[i * 3 + 2], Ckov[i * 3 + 3], Ckov[i * 3 + 4],
                                        Ckov[i * 3 + 5]]))
                Moy_spectrum.append(np.mean(Data[s][f][i * 3:i * 3 + 5], axis=0))

            elif i == 1 or i == 8:
                continue
            else:
                Moy_doses.append(np.mean([Doses[i * 3], Doses[i * 3 + 1], Doses[i * 3 + 2]]))
                Moy_fluo.append(np.mean([Fluo[i * 3], Fluo[i * 3 + 1], Fluo[i * 3 + 2]]))
                Moy_ckov.append(np.mean([Ckov[i * 3], Ckov[i * 3 + 1], Ckov[i * 3 + 2]]))
                dev_doses.append(np.std([Doses[i * 3], Doses[i * 3 + 1], Doses[i * 3 + 2]]))
                dev_fluo.append(np.std([Fluo[i * 3], Fluo[i * 3 + 1], Fluo[i * 3 + 2]]))
                dev_ckov.append(np.std([Ckov[i * 3], Ckov[i * 3 + 1], Ckov[i * 3 + 2]]))
                Moy_spectrum.append(np.mean(Data[s][f][i * 3:i * 3 + 2], axis=0))

        Mean_dose[s][f] = Moy_doses
        Mean_fluo[s][f] = Moy_fluo
        Mean_ckov[s][f] = Moy_ckov
        std_dose[s][f] = dev_doses
        std_fluo[s][f] = dev_fluo
        std_ckov[s][f] = dev_ckov
        Mean_spectrum[s][f] = Moy_spectrum

# Figures for abstract AAPM : comparison between BCF-10 and Medscint
subset = [0, 4]
## For the supporting document

'''
# Absolute spectra

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True)
plt.subplot(121)
s == 0
plt.title(scint_label[s])
for s in subset:
    if s == 0 :
        plt.subplot(121)
        plt.title(scint_label[s])
    else:
        plt.subplot(122)
        plt.title(scint_label[s])
    for ff in range(len(field_size) - 1):
        f = ff + 1
        plt.plot(Mean_spectrum[s][f][0], color=colors_blue[ff][4], label=str(field_size[f]) + 'x' + str(
            field_size[f]) + ' cm2, ' + str(bfields[0]) + ' T')
        plt.plot(Mean_spectrum[s][f][5], ls='--', color=colors_blue[ff][2],  label= str(field_size[f]) + 'x' + str(
            field_size[f]) + ' cm2, ' + str(bfields[5]) + ' T e- \u2192 tip')
        plt.plot(Mean_spectrum[s][f][11], ls=':', color=colors_blue[ff][0],  label= str(field_size[f]) + 'x' + str(
            field_size[f]) + ' cm2, ' + str(bfields[5]) + ' T e- \u2192 stem')
        plt.xlabel('Wavelength (a.u.)')
        if s == 0:
            plt.ylabel('Intensity (a.u.)')
plt.legend(framealpha=1, frameon=True)
plt.tight_layout()
plt.savefig("AAPM_supp_specrum.eps")
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True)
plt.subplot(121)
s == 0
plt.title(scint_label[s])
for s in subset:
    if s == 0 :
        plt.subplot(121)
        plt.title(scint_label[s])
    else:
        plt.subplot(122)
        plt.title(scint_label[s])
    for ff in range(len(field_size) - 1):
        f = ff + 1
        if f == 1:
            plt.plot(Mean_spectrum[s][f][0]/sum(Mean_spectrum[s][f][0]), color='grey', label=str(field_size[f]) + 'x' + str(
            field_size[f]) + ' cm2, ' + str(bfields[0]) + ' T')
        else:
            plt.plot(Mean_spectrum[s][f][0] / sum(Mean_spectrum[s][f][0]), color='black',
                     label=str(field_size[f]) + 'x' + str(
                         field_size[f]) + ' cm2, ' + str(bfields[0]) + ' T')
        plt.plot(Mean_spectrum[s][f][5]/sum(Mean_spectrum[s][f][5]), ls='--', color=colors_blue[ff][4],  label= str(field_size[f]) + 'x' + str(
            field_size[f]) + ' cm2, ' + str(bfields[5]) + ' T e- \u2192 tip')
        plt.plot(Mean_spectrum[s][f][11]/sum(Mean_spectrum[s][f][11]), ls=':', color=colors_blue[ff][1],  label= str(field_size[f]) + 'x' + str(
            field_size[f]) + ' cm2, ' + str(bfields[5]) + ' T e- \u2192 stem')
        plt.xlabel('Wavelength (a.u.)')
        if s == 0:
            plt.ylabel('Normalized Intensity to the area under the curve')
plt.legend(framealpha=1, frameon=True)
plt.tight_layout()
plt.savefig("AAPM_supp_specrum_norm.eps")
plt.show()

# Dose and Cherenkov
for ff in range(len(field_size) - 1):
    f = ff + 1
    for s in subset:
        print(s, f, 100 * ((Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6]) - (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0]))
            / (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0]))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1]}, sharex='col')
for ff in range(len(field_size) - 1):
    f = ff + 1
    for s in subset:
        plt.subplot(221)
        plt.errorbar(bfields, Mean_dose[s][f][6:12] / Mean_dose[s][f][6],
                     yerr=np.sqrt((std_dose[s][f][6:12] / Mean_dose[s][f][6:12]) ** 2 +
                                  (std_dose[s][f][6] / Mean_dose[s][f][6]) ** 2) *
                          (Mean_dose[s][f][6:12] / Mean_dose[s][f][6]), color=colors_blue[ff][s],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(
                         field_size[f]) + ' cm2, e- \u2192 stem')
        plt.errorbar(bfields, Mean_dose[s][f][0:6] / Mean_dose[s][f][0],
                     yerr=np.sqrt((std_dose[s][f][0:6] / Mean_dose[s][f][0:6]) ** 2 + (
                             std_dose[s][f][0] / Mean_dose[s][f][0]) ** 2) * (
                                  Mean_dose[s][f][0:6] / Mean_dose[s][f][0]), fmt='--', color=colors_blue[ff][s],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(
                         field_size[f]) + ' cm2, e- \u2192 tip')
        plt.ylabel('Dose / dose [0 T]')

        plt.subplot(222)
        plt.errorbar(bfields, Mean_ckov[s, f, 6:12] / Mean_ckov[s, f, 6],
                     yerr=np.sqrt((std_ckov[s][f][6:12] / Mean_ckov[s][f][6:12]) ** 2
                                  + (std_ckov[s][f][6] / Mean_ckov[s][f][6]) ** 2) *
                          np.abs((Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6])), color=colors_blue[ff][s],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(
                         field_size[f]) + ' cm2, e- \u2192 stem')
        plt.errorbar(bfields, Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0],
                     yerr=np.sqrt((std_ckov[s][f][0:6] / Mean_ckov[s][f][0:6]) ** 2 + (
                             std_ckov[s][f][0] / Mean_ckov[s][f][0]) ** 2) * np.abs((
                             Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0])), fmt='--', color=colors_blue[ff][s],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(
                         field_size[f]) + ' cm2, e- \u2192 tip')
        plt.ylabel('Cherenkov / cherenkov [0 T]')
        plt.legend(loc='center right', bbox_to_anchor=(1.7, 0.5))

        plt.subplot(224)
        plt.plot(bfields, 100 * ((Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6])
                                 - (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0]))
                 / (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0]), color=colors_blue[ff][s],
                 label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(
                     field_size[f]) + ' cm2'),

        plt.xlabel('Magnetic field [T]')
        plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

        plt.subplot(223)
        plt.plot(bfields,
                 100 * ((Mean_dose[s][f][6:12] / Mean_dose[s][f][6]) - (Mean_dose[s][f][0:6] / Mean_dose[s][f][0])) /
                 (Mean_dose[s][f][0:6] / Mean_dose[s][f][0]), color=colors_blue[ff][s],
                 label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(
                     field_size[f]) + ' cm2')

        plt.xlabel('Magnetic field [T]')
        plt.ylabel('Relative difference \n in orientation [%]')


plt.tight_layout()
plt.savefig("AAPM_supp.eps")
plt.show()


## Absolute
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]}, sharex='col')
for ff in range(len(field_size) - 1):
    f = ff + 1
    for s in subset:
        plt.subplot(221)
        plt.errorbar(bfields, Mean_dose[s][f][6:12], yerr=(std_dose[s][f][6:12]), color=colors_blue[ff][s],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(field_size[f])
                           + ' cm2, e- \u2192 stem')
        plt.errorbar(bfields, Mean_dose[s][f][0:6], yerr=std_dose[s][f][0:6], fmt='--', color=colors_blue[ff][s],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(
                         field_size[f]) + ' cm2, e- \u2192 tip')
        plt.ylabel('Dose / dose [0 T]')

        plt.subplot(222)
        plt.errorbar(bfields, Mean_ckov[s, f, 6:12],
                     yerr=std_ckov[s][f][6:12], color=colors_blue[ff][s],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(
                         field_size[f]) + ' cm2, e- \u2192 stem')
        plt.errorbar(bfields, Mean_ckov[s][f][0:6],
                     yerr=std_ckov[s][f][0:6], fmt='--', color=colors_blue[ff][s],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(
                         field_size[f]) + ' cm2, e- \u2192 tip')
        plt.ylabel('Cherenkov / cherenkov [0 T]')
        plt.legend(loc='center right', bbox_to_anchor=(1.7, 0.5))

        plt.subplot(224)
        plt.plot(bfields, 100 * (Mean_ckov[s][f][6:12] - Mean_ckov[s][f][0:6]) / Mean_ckov[s][f][0:6],
                 color=colors_blue[ff][s], label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(
                field_size[f]) + ' cm2'),

        plt.xlabel('Magnetic field [T]')
        plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

        plt.subplot(223)
        plt.plot(bfields, 100 * (Mean_dose[s][f][6:12] - Mean_dose[s][f][0:6] ) /
                 Mean_dose[s][f][0:6] , color=colors_blue[ff][s],
                 label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(
                     field_size[f]) + ' cm2')

        plt.xlabel('Magnetic field [T]')
        plt.ylabel('Relative difference \n in orientation [%]')

plt.tight_layout()
plt.savefig("AAPM_supp_abs.eps")
plt.show()

#### absolute ckov

for ff in range(len(field_size) - 1):
    f = ff + 1
    for s in subset:
        print(s, f)
        plt.subplot(221 + ff)
        plt.title(str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')
        plt.errorbar(bfields, Mean_ckov[s, f, 6:12],
                     yerr=std_ckov[s][f][6:12], color=colors[s], label=scint_label[s] + ', e- \u2192 stem')
        plt.errorbar(bfields, Mean_ckov[s][f][0:6],
                     yerr=std_ckov[s][f][0:6], fmt='--', color=colors[s],
                     label=scint_label[s] + ', e- \u2192 tip')

        if ff == 0:
            plt.ylabel('ckov / ckov [0 T]')
        if ff == 1:
            plt.legend(loc='center right', bbox_to_anchor=(1.7, 0.5))

        plt.subplot(223 + ff)
        plt.plot(bfields, 100 * (Mean_ckov[s][f][6:12] - Mean_ckov[s][f][0:6]) / Mean_ckov[s][f][0:6], color=colors[s],
                 label=scint_label[s])

        plt.xlabel('Magnetic field [T]')
        if ff == 0:
            plt.ylabel('Relative difference \n in orientation [%]')
        if ff == 1:
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.savefig("AAPM_Ckov_field_abs.eps")
plt.show()

#####
subset = [0, 4]
# Each field individually
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]}, sharey='row',
                         sharex='col')
for ff in range(len(field_size) - 1):
    f = ff + 1
    for s in subset:
        plt.subplot(221 + ff)
        plt.title(str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')
        plt.errorbar(bfields, Mean_dose[s][f][6:12] / Mean_dose[s][f][6],
                     yerr=np.sqrt((std_dose[s][f][6:12] / Mean_dose[s][f][6:12]) ** 2 +
                                  (std_dose[s][f][6] / Mean_dose[s][f][6]) ** 2) *
                          (Mean_dose[s][f][6:12] / Mean_dose[s][f][6]), color=colors[s],
                     label=scint_label[s] + ', e- \u2192 stem')
        plt.errorbar(bfields, Mean_dose[s][f][0:6] / Mean_dose[s][f][0],
                     yerr=np.sqrt((std_dose[s][f][0:6] / Mean_dose[s][f][0:6]) ** 2 + (
                             std_dose[s][f][0] / Mean_dose[s][f][0]) ** 2) * (
                                  Mean_dose[s][f][0:6] / Mean_dose[s][f][0]), fmt='--', color=colors[s],
                     label=scint_label[s] + ', e- \u2192 tip')
        if ff == 0:
            plt.ylabel('Dose / dose [0 T]')
        if ff == 1:
            plt.legend(loc='center right', bbox_to_anchor=(1.7, 0.5))

        plt.subplot(223 + ff * 1)
        plt.plot(bfields,
                 100 * ((Mean_dose[s][f][6:12] / Mean_dose[s][f][6]) - (Mean_dose[s][f][0:6] / Mean_dose[s][f][0])) /
                 (Mean_dose[s][f][0:6] / Mean_dose[s][f][0]), color=colors[s], label=scint_label[s])

        plt.xlabel('Magnetic field [T]')
        if ff == 0:
            plt.ylabel('Relative difference \n in orientation [%]')
        if ff == 1:
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.savefig("AAPM_Scint_field.eps")
plt.show()

# Each detector separated
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]}, sharey='row',
                         sharex='col')
subset = [0, 4]
b_subset = [0.35, 0.5, 1, 1.5]
temp = 221
for s in subset:
    for ff in range(len(field_size) - 1):
        f = ff + 1
        plt.subplot(temp)
        plt.title(scint_label[s])
        plt.errorbar(bfields, Mean_dose[s][f][6:12] / Mean_dose[s][f][6],
                     yerr=np.sqrt((std_dose[s][f][6:12] / Mean_dose[s][f][6:12]) ** 2 + (
                             std_dose[s][f][6] / Mean_dose[s][f][6]) ** 2) *
                          (Mean_dose[s][f][6:12] / Mean_dose[s][f][6]), color=colors[f],
                     label=str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2, e- \u2192 stem')
        plt.errorbar(bfields, Mean_dose[s][f][0:6] / Mean_dose[s][f][0],
                     yerr=np.sqrt((std_dose[s][f][0:6] / Mean_dose[s][f][0:6]) ** 2 + (
                             std_dose[s][f][0] / Mean_dose[s][f][0]) ** 2) * (
                                  Mean_dose[s][f][0:6] / Mean_dose[s][f][0]), fmt='--', color=colors[f],
                     label=str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2, e- \u2192 tip')
        if s == 4:
            plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
        else:
            plt.ylabel('Dose / dose [0 T]')
        plt.subplot(temp + 2)
        plt.plot(bfields, 100 * (
                (Mean_dose[s][f][6:12] / Mean_dose[s][f][6]) - (Mean_dose[s][f][0:6] / Mean_dose[s][f][0])) /
                 (Mean_dose[s][f][0:6] / Mean_dose[s][f][0]), color=colors[f],
                 label=str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')

        plt.xlabel('Magnetic field [T]')
        if s == 4:
            plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
        else:
            plt.ylabel('Relative difference \n in orientation [%]')
    temp += 1
plt.tight_layout()
plt.savefig("AAPM_scint_detector.eps")
plt.show()

# Each orientation individually
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 7), sharey=True)
subset = [0, 4]
b_subset = [0.35, 0.5, 1, 1.5]
temp = 221
for s in subset:
    for ff in range(len(field_size) - 1):
        f = ff + 1
        plt.subplot(121)
        plt.title('e- \u2192 tip')
        plt.errorbar(bfields, Mean_dose[s][f][0:6] / Mean_dose[s][f][0],
                     yerr=np.sqrt((std_dose[s][f][0:6] / Mean_dose[s][f][0:6]) ** 2 + (
                             std_dose[s][f][0] / Mean_dose[s][f][0]) ** 2) * (
                                  Mean_dose[s][f][0:6] / Mean_dose[s][f][0]), fmt=lin[s], color=colors[f],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')
        plt.ylabel('Dose / dose [0 T]')
        plt.xlabel('Magnetic field [T]')

        plt.subplot(122)
        plt.title('e- \u2192 stem')
        plt.errorbar(bfields, Mean_dose[s][f][6:12] / Mean_dose[s][f][6],
                     yerr=np.sqrt((std_dose[s][f][6:12] / Mean_dose[s][f][6:12]) ** 2 + (
                             std_dose[s][f][6] / Mean_dose[s][f][6]) ** 2) *
                          (Mean_dose[s][f][6:12] / Mean_dose[s][f][6]), fmt=lin[s], color=colors[f],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')

        plt.xlabel('Magnetic field [T]')
        plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.savefig("AAPM_scint_orientation.eps")
plt.show()

# Cherenkov
# Each field individually
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]}, sharey='row',
                         sharex='col')

for ff in range(len(field_size) - 1):
    f = ff + 1
    for s in subset:
        print(s, f)
        plt.subplot(221 + ff)
        plt.title(str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')
        plt.errorbar(bfields, Mean_ckov[s, f, 6:12] / Mean_ckov[s, f, 6],
                     yerr=np.sqrt((std_ckov[s][f][6:12] / Mean_ckov[s][f][6:12]) ** 2 + (
                             std_ckov[s][f][6] / Mean_ckov[s][f][6]) ** 2) *
                          np.abs((Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6])), color=colors[s],
                     label=scint_label[s] + ', e- \u2192 stem')
        plt.errorbar(bfields, Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0],
                     yerr=np.sqrt((std_ckov[s][f][0:6] / Mean_ckov[s][f][0:6]) ** 2 + (
                             std_ckov[s][f][0] / Mean_ckov[s][f][0]) ** 2) * np.abs((
                             Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0])), fmt='--', color=colors[s],
                     label=scint_label[s] + ', e- \u2192 tip')

        if ff == 0:
            plt.ylabel('ckov / ckov [0 T]')
        if ff == 1:
            plt.legend(loc='center right', bbox_to_anchor=(1.7, 0.5))

        plt.subplot(223 + ff)
        plt.plot(bfields,
                 100 * ((Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6]) - (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0])) /
                 (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0]), color=colors[s], label=scint_label[s])

        plt.xlabel('Magnetic field [T]')
        if ff == 0:
            plt.ylabel('Relative difference \n in orientation [%]')
        if ff == 1:
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.savefig("AAPM_Ckov_field.eps")
plt.show()
'''
# Each detector separated
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]}, sharey='row',
                         sharex='col')
subset = [0, 4]
b_subset = [0.35, 0.5, 1, 1.5]
temp = 221
for s in subset:
    for ff in range(len(field_size) - 1):
        f = ff + 1

        plt.subplot(temp)
        plt.title(scint_label[s])
        plt.errorbar(bfields, Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6],
                     yerr=np.sqrt((std_ckov[s][f][6:12] / Mean_ckov[s][f][6:12]) ** 2 + (
                             std_ckov[s][f][6] / Mean_ckov[s][f][6]) ** 2) * np.abs(
                         (Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6])), color=colors[f],
                     label=str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2, e- \u2192 stem')
        plt.errorbar(bfields, Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0],
                     yerr=np.sqrt((std_ckov[s][f][0:6] / Mean_ckov[s][f][0:6]) ** 2 + (
                             std_ckov[s][f][0] / Mean_ckov[s][f][0]) ** 2) * np.abs((
                             Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0])), fmt='--', color=colors[f],
                     label=str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2, e- \u2192 tip')

        if s == 4:
            plt.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
        else:
            plt.ylabel('ckov / ckov [0 T]')



        plt.xlabel('Magnetic field [T]')

    plt.subplot(temp + 2)
    plt.plot(bfields,  (
            (Mean_ckov[s][1][6:12] / Mean_ckov[s][1][6]) - (Mean_ckov[s][2][6:12] / Mean_ckov[s][2][6])) ,
             label='e- \u2192 stem')
    plt.plot(bfields,  (
            (Mean_ckov[s][1][0:6] / Mean_ckov[s][1][0]) - (Mean_ckov[s][2][0:6] / Mean_ckov[s][2][0])), ':',
             label='e- \u2192 tip')
    plt.axhline(y=0, color='black', linestyle='-')
    if s == 4:
        plt.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
    else:
        plt.ylabel('Relative difference \n in orientation')


    temp += 1

plt.tight_layout()
plt.savefig("AAPM_Ckov_detector.eps")
plt.show()

# Each orientation individually
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 7), sharey=True)
subset = [0, 4]
b_subset = [0.35, 0.5, 1, 1.5]
temp = 221
for s in subset:
    for ff in range(len(field_size) - 1):
        f = ff + 1
        plt.subplot(121)
        plt.title('e- \u2192 tip')
        plt.errorbar(bfields, Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0],
                     yerr=np.sqrt((std_ckov[s][f][0:6] / Mean_ckov[s][f][0:6]) ** 2 + (
                             std_ckov[s][f][0] / Mean_ckov[s][f][0]) ** 2) * np.abs((
                             Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0])), fmt=lin[s], color=colors[f],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')
        plt.ylabel('ckov / ckov [0 T]')
        plt.xlabel('Magnetic field [T]')
        plt.subplot(122)
        plt.title('e- \u2192 stem')
        plt.errorbar(bfields, Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6],
                     yerr=np.sqrt((std_ckov[s][f][6:12] / Mean_ckov[s][f][6:12]) ** 2 + (
                             std_ckov[s][f][6] / Mean_ckov[s][f][6]) ** 2) * np.abs(
                         (Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6])), fmt=lin[s], color=colors[f],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')
        plt.xlabel('Magnetic field [T]')

        plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.savefig("AAPM_Ckov_orientation.eps")
plt.show()

# Fluo

# Each field individually
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]}, sharey='row',
                         sharex='col')

for ff in range(len(field_size) - 1):
    f = ff + 1
    for s in subset:
        plt.subplot(221 + ff)
        plt.title(str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')
        plt.errorbar(bfields, Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6],
                     yerr=np.sqrt((std_fluo[s][f][6:12] / Mean_fluo[s][f][6:12]) ** 2 + (
                             std_fluo[s][f][6] / Mean_fluo[s][f][6]) ** 2) *
                          np.abs((Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6])), color=colors[s],
                     label=scint_label[s] + ', e- \u2192 stem')
        plt.errorbar(bfields, Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0],
                     yerr=np.sqrt((std_fluo[s][f][0:6] / Mean_fluo[s][f][0:6]) ** 2 + (
                             std_fluo[s][f][0] / Mean_fluo[s][f][0]) ** 2) * np.abs((
                             Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0])), fmt='--', color=colors[s],
                     label=scint_label[s] + ', e- \u2192 tip')
        if ff == 0:
            plt.ylabel('fluo / fluo [0 T]')
        if ff == 1:
            plt.legend(loc='center right', bbox_to_anchor=(1.7, 0.5))
        plt.subplot(223 + ff)
        plt.plot(bfields,
                 100 * ((Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6]) - (Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0])) /
                 (Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0]), color=colors[s], label=scint_label[s])

        plt.xlabel('Magnetic field [T]')
        if ff == 0:
            plt.ylabel('Relative difference \n in orientation [%]')
        if ff == 1:
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.savefig("AAPM_fluo_field.eps")
plt.show()

# Each detector separated
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]}, sharey='row',
                         sharex='col')
subset = [0, 4]
b_subset = [0.35, 0.5, 1, 1.5]
temp = 221
for s in subset:
    for ff in range(len(field_size) - 1):
        f = ff + 1

        plt.subplot(temp)
        plt.title(scint_label[s])
        plt.errorbar(bfields, Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6],
                     yerr=np.sqrt((std_fluo[s][f][6:12] / Mean_fluo[s][f][6:12]) ** 2 + (
                             std_fluo[s][f][6] / Mean_fluo[s][f][6]) ** 2) *
                          np.abs((Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6])), color=colors[f],
                     label=str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2, e- \u2192 stem')
        plt.errorbar(bfields, Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0],
                     yerr=np.sqrt((std_fluo[s][f][0:6] / Mean_fluo[s][f][0:6]) ** 2 + (
                             std_fluo[s][f][0] / Mean_fluo[s][f][0]) ** 2) * np.abs((
                             Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0])), fmt='--', color=colors[f],
                     label=str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2, e- \u2192 tip')

        if s == 4:
            plt.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
        else:
            plt.ylabel('fluo / fluo [0 T]')

        plt.subplot(temp + 2)
        plt.plot(bfields, 100 * (
                (Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6]) - (Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0])) /
                 (Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0]), color=colors[f],
                 label=str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')

        plt.xlabel('Magnetic field [T]')
        if s == 4:
            plt.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
        else:
            plt.ylabel('Relative difference \n in orientation [%]')
    temp += 1

plt.tight_layout()
plt.savefig("AAPM_fluo_detector.eps")
plt.show()

# Each orientation individually
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 7), sharey=True)
subset = [0, 4]
b_subset = [0.35, 0.5, 1, 1.5]
temp = 221
for s in subset:
    for ff in range(len(field_size) - 1):
        f = ff + 1
        plt.subplot(121)
        plt.title('e- \u2192 tip')
        plt.errorbar(bfields, Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0],
                     yerr=np.sqrt((std_fluo[s][f][0:6] / Mean_fluo[s][f][0:6]) ** 2 + (
                             std_fluo[s][f][0] / Mean_fluo[s][f][0]) ** 2) * (
                                  Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0]), fmt=lin[s], color=colors[f],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')
        plt.ylabel('fluo / fluo [0 T]')
        plt.xlabel('Magnetic field [T]')

        plt.subplot(122)
        plt.title('e- \u2192 stem')
        plt.errorbar(bfields, Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6],
                     yerr=np.sqrt((std_fluo[s][f][6:12] / Mean_fluo[s][f][6:12]) ** 2 + (
                             std_fluo[s][f][6] / Mean_fluo[s][f][6]) ** 2) *
                          np.abs((Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6])), fmt=lin[s], color=colors[f],
                     label=scint_label[s] + ', ' + str(field_size[f]) + 'x' + str(field_size[f]) + ' cm2')
        plt.xlabel('Magnetic field [T]')
        plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.savefig("AAPM_fluo_orientation.eps")
plt.show()

'''
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})

for f in range(len(field_size)):
    for s in range(len(scintillators)):
        plt.subplot(231 + f*1)
        plt.errorbar(bfields, Mean_dose[s][f][6:12] / Mean_dose[s][f][6],
                     yerr=np.sqrt((std_dose[s][f][6:12] / Mean_dose[s][f][6:12]) ** 2 + (std_dose[s][f][6] / Mean_dose[s][f][6]) ** 2) *
                          (Mean_dose[s][f][6:12] / Mean_dose[s][f][6]), color=colors[s],
                     label=scint_label[s] + ', e- \u2192 stem')
        plt.errorbar(bfields, Mean_dose[s][f][0:6] / Mean_dose[s][f][0],
                     yerr=np.sqrt((std_dose[s][f][0:6] / Mean_dose[s][f][0:6]) ** 2 + (std_dose[s][f][0] / Mean_dose[s][f][0]) ** 2) * (
                          Mean_dose[s][f][0:6] / Mean_dose[s][f][0]), fmt='--', color=colors[s],
                     label=scint_label[s] + ', e- \u2192 tip')
        plt.ylabel('Dose / dose [0 T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

        plt.subplot(234 + f*1)
        plt.plot(bfields, 100*((Mean_dose[s][f][6:12] / Mean_dose[s][f][6]) - (Mean_dose[s][f][0:6] / Mean_dose[s][f][0])) /
                 (Mean_dose[s][f][0:6] / Mean_dose[s][f][0]), color=colors[s], label=scint_label[s])
        plt.ylabel('Relative difference \n in orientation [%]')
        plt.xlabel('Magnetic field [T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})

for f in range(len(field_size)):
    for s in range(len(scintillators)):
        plt.subplot(231 + f*1)
        plt.errorbar(bfields, Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6],
                     yerr=np.sqrt(np.abs((std_ckov[s][f][6:12] / Mean_ckov[s][f][6:12]) ** 2 + (std_ckov[s][f][6] / Mean_ckov[s][f][6]) ** 2) *
                          (Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6])), color=colors[s],
                     label=scint_label[s] + ', e- \u2192 stem')
        plt.errorbar(bfields, Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0],
                     yerr=np.sqrt(np.abs((std_ckov[s][f][0:6] / Mean_ckov[s][f][0:6]) ** 2 + (std_ckov[s][f][0] / Mean_ckov[s][f][0]) ** 2) * (
                          Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0])), fmt='--', color=colors[s],
                     label=scint_label[s] + ', e- \u2192 tip')
        plt.ylabel('ckov / ckov [0 T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

        plt.subplot(234 + f*1)
        plt.plot(bfields, 100*((Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6]) - (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0])) /
                 (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0]), color=colors[s], label=scint_label[s])
        plt.ylabel('Relative difference \n in orientation [%]')
        plt.xlabel('Magnetic field [T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})

for f in range(len(field_size)):
    for s in range(len(scintillators)):
        plt.subplot(231 + f*1)
        plt.errorbar(bfields, Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6],
                     yerr=np.sqrt(np.abs((std_fluo[s][f][6:12] / Mean_fluo[s][f][6:12]) ** 2 + (std_fluo[s][f][6] / Mean_fluo[s][f][6]) ** 2) *
                          (Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6])), color=colors[s],
                     label=scint_label[s] + ', e- \u2192 stem')
        plt.errorbar(bfields, Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0],
                     yerr=np.sqrt(np.abs((std_fluo[s][f][0:6] / Mean_fluo[s][f][0:6]) ** 2 + (std_fluo[s][f][0] / Mean_fluo[s][f][0]) ** 2) * (
                          Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0])), fmt='--', color=colors[s],
                     label=scint_label[s] + ', e- \u2192 tip')
        plt.ylabel('fluo / fluo [0 T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

        plt.subplot(234 + f*1)
        plt.plot(bfields, 100*((Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6]) - (Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0])) /
                 (Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0]), color=colors[s], label=scint_label[s])
        plt.ylabel('Relative difference \n in orientation [%]')
        plt.xlabel('Magnetic field [T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
subset = [0, 4]
b_subset = [0.35, 0.5, 1, 1.5]
for b in range(len(b_subset)):
    for s in subset:
        plt.subplot(211)
        plt.title(scintillators[s] )
        plt.errorbar(field_size, Mean_dose[s, :, b+6] / Mean_dose[s, :, 6],
                     yerr=np.sqrt((std_dose[s, :, b+6] / Mean_dose[s, :, b+6]) ** 2 + (std_dose[s, :, 6] / Mean_dose[s, :, 6]) ** 2) *
                          (Mean_dose[s, :, b+6] / Mean_dose[s, :, 6]), color=colors[b], marker = mark[s],
                     label= scintillators[s] + ', ' + str(b_subset[b]) + 'T, e- \u2192 stem')
        plt.errorbar(field_size, Mean_dose[s, :, b] / Mean_dose[s, :, 0],
                     yerr=np.sqrt((std_dose[s, :, b] / Mean_dose[s, :, b]) ** 2 + (std_dose[s, :, 0] / Mean_dose[s, :, 0]) ** 2) * (
                          Mean_dose[s, :, b] / Mean_dose[s, :, 0]), fmt='--', color=colors[b], marker = mark[s],
                     label= scintillators[s] + ', ' + str(b_subset[b]) + 'T, e- \u2192 tip')
        plt.ylabel('Dose / dose [0 T]')
        plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

        plt.subplot(212)
        plt.plot(field_size, 100*((Mean_dose[s, :,b+6] / Mean_dose[s, :, 6]) - (Mean_dose[s, :, b] / Mean_dose[s, :, 0])) /
                 (Mean_dose[s, :, b] / Mean_dose[s, :, 0]), color=colors[b], label=scintillators[s] + ', ' + str(b_subset[b])+ 'T')
        plt.ylabel('Relative difference \n in orientation [%]')
        plt.xlabel('Magnetic field [T]')
        plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})

for f in range(len(field_size)):
    for s in subset:
        plt.subplot(231 + f * 1)
        plt.title(str(field_size[f]) + 'x' + str(field_size[f]) + 'cm2')
        plt.errorbar(bfields, Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6],
                     yerr=np.sqrt(np.abs((std_ckov[s][f][6:12] / Mean_ckov[s][f][6:12]) ** 2 + (
                                 std_ckov[s][f][6] / Mean_ckov[s][f][6]) ** 2) *
                                  (Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6])), color=colors[s],
                     label=scint_label[s] + ', e- \u2192 stem')
        plt.errorbar(bfields, Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0],
                     yerr=np.sqrt(np.abs((std_ckov[s][f][0:6] / Mean_ckov[s][f][0:6]) ** 2 + (
                                 std_ckov[s][f][0] / Mean_ckov[s][f][0]) ** 2) * (
                                          Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0])), fmt='--', color=colors[s],
                     label=scint_label[s] + ', e- \u2192 tip')
        plt.ylabel('ckov / ckov [0 T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

        plt.subplot(234 + f * 1)
        plt.plot(bfields,
                 100 * ((Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6]) - (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0])) /
                 (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0]), color=colors[s], label=scint_label[s])
        plt.ylabel('Relative difference \n in orientation [%]')
        plt.xlabel('Magnetic field [T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.show()

# As function of b field
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
subset = [0, 4]
for f in range(len(field_size)):
    for s in subset:
        plt.subplot(231 + f * 1)
        plt.title(str(field_size[f]) + 'x' + str(field_size[f]) + 'cm2')
        plt.errorbar(bfields, Mean_dose[s][f][6:12] / Mean_dose[s][f][6],
                     yerr=np.sqrt((std_dose[s][f][6:12] / Mean_dose[s][f][6:12]) ** 2 + (
                                 std_dose[s][f][6] / Mean_dose[s][f][6]) ** 2) *
                          (Mean_dose[s][f][6:12] / Mean_dose[s][f][6]), color=colors[s],
                     label=scint_label[s] + ', e- \u2192 stem')
        plt.errorbar(bfields, Mean_dose[s][f][0:6] / Mean_dose[s][f][0],
                     yerr=np.sqrt((std_dose[s][f][0:6] / Mean_dose[s][f][0:6]) ** 2 + (
                                 std_dose[s][f][0] / Mean_dose[s][f][0]) ** 2) * (
                                  Mean_dose[s][f][0:6] / Mean_dose[s][f][0]), fmt='--', color=colors[s],
                     label=scint_label[s] + ', e- \u2192 tip')
        plt.ylabel('Dose / dose [0 T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

        plt.subplot(234 + f * 1)
        plt.plot(bfields,
                 100 * ((Mean_dose[s][f][6:12] / Mean_dose[s][f][6]) - (Mean_dose[s][f][0:6] / Mean_dose[s][f][0])) /
                 (Mean_dose[s][f][0:6] / Mean_dose[s][f][0]), color=colors[s], label=scint_label[s])
        plt.ylabel('Relative difference \n in orientation [%]')
        plt.xlabel('Magnetic field [T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})

for f in range(len(field_size)):
    for s in subset:
        plt.subplot(231 + f * 1)
        plt.title(str(field_size[f]) + 'x' + str(field_size[f]) + 'cm2')
        plt.errorbar(bfields, Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6],
                     yerr=np.sqrt(np.abs((std_ckov[s][f][6:12] / Mean_ckov[s][f][6:12]) ** 2 + (
                                 std_ckov[s][f][6] / Mean_ckov[s][f][6]) ** 2) *
                                  (Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6])), color=colors[s],
                     label=scint_label[s] + ', e- \u2192 stem')
        plt.errorbar(bfields, Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0],
                     yerr=np.sqrt(np.abs((std_ckov[s][f][0:6] / Mean_ckov[s][f][0:6]) ** 2 + (
                                 std_ckov[s][f][0] / Mean_ckov[s][f][0]) ** 2) * (
                                          Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0])), fmt='--', color=colors[s],
                     label=scint_label[s] + ', e- \u2192 tip')
        plt.ylabel('ckov / ckov [0 T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

        plt.subplot(234 + f * 1)
        plt.plot(bfields,
                 100 * ((Mean_ckov[s][f][6:12] / Mean_ckov[s][f][6]) - (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0])) /
                 (Mean_ckov[s][f][0:6] / Mean_ckov[s][f][0]), color=colors[s], label=scint_label[s])
        plt.ylabel('Relative difference \n in orientation [%]')
        plt.xlabel('Magnetic field [T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})

for f in range(len(field_size)):
    for s in subset:
        plt.subplot(231 + f * 1)
        plt.title(field_size[f])
        plt.errorbar(bfields, Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6],
                     yerr=np.sqrt(np.abs((std_fluo[s][f][6:12] / Mean_fluo[s][f][6:12]) ** 2 + (
                                 std_fluo[s][f][6] / Mean_fluo[s][f][6]) ** 2) *
                                  (Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6])), color=colors[s],
                     label=scint_label[s] + ', e- \u2192 stem')
        plt.errorbar(bfields, Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0],
                     yerr=np.sqrt(np.abs((std_fluo[s][f][0:6] / Mean_fluo[s][f][0:6]) ** 2 + (
                                 std_fluo[s][f][0] / Mean_fluo[s][f][0]) ** 2) * (
                                          Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0])), fmt='--', color=colors[s],
                     label=scint_label[s] + ', e- \u2192 tip')
        plt.ylabel('fluo / fluo [0 T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))

        plt.subplot(234 + f * 1)
        plt.plot(bfields,
                 100 * ((Mean_fluo[s][f][6:12] / Mean_fluo[s][f][6]) - (Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0])) /
                 (Mean_fluo[s][f][0:6] / Mean_fluo[s][f][0]), color=colors[s], label=scint_label[s])
        plt.ylabel('Relative difference \n in orientation [%]')
        plt.xlabel('Magnetic field [T]')
        if f == 2:
            plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
plt.tight_layout()
plt.show()
'''
