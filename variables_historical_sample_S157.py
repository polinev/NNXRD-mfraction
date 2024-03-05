# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:00:00 2023

@author: Victor.POLINE

Variables and function for scripts

"""

import h5py
import pandas as pd
import numpy as np

file = h5py.File('S2018_157_sinogram_layer02_XRD_powder_pack_half1-back_clean_corr_mask.h5', 'r')
list(file.keys())

# =============================================================================
#     # Load tth from h5
# =============================================================================
dset_tth = file['2theta']
list_tth = dset_tth[:]
length_list = list_tth.shape[0]
tth_min = list_tth.min()
tth_max = list_tth.max()
resolution_tth = (tth_max - tth_min) / length_list
lambdaBM2 = 0.619921 # Energy of 20 keV

# =============================================================================
#     # Phases dabatase from .hkl of Rietveld refinement from Fullprof
# =============================================================================
with open('rietveld_refinement_sum_pattern_S157.hkl', 'r') as f:
    lines = f.read().split('Pattern')
for i in range(len(lines)):
    with open("phase_%d.txt" % i, 'w') as f:
        f.write(lines[i])
np.array(pd.read_csv('phase_1.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))

db_beeswax = np.array(pd.read_csv('phase_3.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_cassiterite = np.array(pd.read_csv('phase_5.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_cerussite = np.array(pd.read_csv('phase_6.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_chlorargyrite = np.array(pd.read_csv('phase_9.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_cinnabar = np.array(pd.read_csv('phase_2.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_goethite = np.array(pd.read_csv('phase_10.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_gold = np.array(pd.read_csv('phase_8.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_gypsum = np.array(pd.read_csv('phase_1.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_hydrocerussite = np.array(pd.read_csv('phase_7.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_minium = np.array(pd.read_csv('phase_11.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_romarchite = np.array(pd.read_csv('phase_4.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))

# =============================================================================
#     # Definition of I/Icor for the 11 phases :
# =============================================================================
Icor_beeswax = 1.20 # Calculated from simulation
Icor_cassiterite = 1.91  # Calculated from simulation
Icor_cerussite = 11.72 # Calculated from simulation
Icor_chlorargyrite = 7.76 # Calculated from simulation
Icor_cinnabar = 11.5 # Calculated from simulation
Icor_goethite = 1.43 #  Calculated from simulation
Icor_gold = 19.99 # Calculated from simulation
Icor_gypsum = 2.01 # Calculated from simulation
Icor_hydrocerussite = 9.57 # Calculated from simulation
Icor_minium = 17.71 # Calculated from simulation
Icor_romarchite = 9.4 # Calculated from simulation

# =============================================================================
#     # Create python dictionnaries for corresponding index/phase, phase/db and phase/icor
# =============================================================================
dict_ph = {0: "beeswax", 1: "cassiterite", 2: "cerussite", 3: "chlorargyrite",
           4: "cinnabar", 5: "goethite", 6: "gold", 7: "gypsum",
           8: "hydrocerussite", 9: "minium", 10: "romarchite"}

dict_db_ph = {"beeswax": db_beeswax, "cassiterite": db_cassiterite, "cerussite": db_cerussite,
              "chlorargyrite": db_chlorargyrite, "cinnabar": db_cinnabar, "goethite": db_goethite,
              "gold": db_gold, "gypsum": db_gypsum, "hydrocerussite": db_hydrocerussite,
              "minium": db_minium, "romarchite": db_romarchite}

dict_icor_ph = {"beeswax": Icor_beeswax, "cassiterite": Icor_cassiterite, "cerussite": Icor_cerussite,
              "chlorargyrite": Icor_chlorargyrite, "cinnabar": Icor_cinnabar, "goethite": Icor_goethite,
              "gold": Icor_gold, "gypsum": Icor_gypsum, "hydrocerussite": Icor_hydrocerussite,
              "minium": Icor_minium, "romarchite": Icor_romarchite}

# =============================================================================
#     Definition function of combination (every possibilities instead of triplet)
# =============================================================================
def combs(a):
    if len(a) == 0:
        return [[]]
    cs = []
    for c in combs(a[1:]):
        cs += [c, c+[a[0]]]
    return cs
list_combination = combs(list(range(11)))
list_combination.pop(0)

# =============================================================================
#     Building the phases database from the preliminary dictionnaries
# =============================================================================
list_db = []
list_icor = []
for i in dict_ph.values():
    list_db.append(dict_db_ph[i])
    list_icor.append(dict_icor_ph[i])
    
n_ph = len(list_db)