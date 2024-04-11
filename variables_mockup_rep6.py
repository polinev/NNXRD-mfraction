# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:37:11 2022

@author: Victor.POLINE

Variables and function for scripts

"""

import h5py
import pandas as pd
import numpy as np

file = h5py.File('rep-6_layer_0_0002_powder_short-iback_clean_corr_new.h5', 'r')
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
#     # Phases dabatase from pretreated .csv (without line of multiple indexes and less than given I)
# =============================================================================
db_anhydrite = np.array(pd.read_csv('anhydrite.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_cinnabar = np.array(pd.read_csv('cinnabar.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))
db_romarchite = np.array(pd.read_csv('romarchite.txt', header=2, delim_whitespace=True, usecols=['2T', 'Intensity'], dtype= np.float64))

# =============================================================================
#     # Definition of I/Icor for the 3 phases :
# =============================================================================
Icor_anhydrite = 1.63 # From simulation 50/50 corundum
Icor_cinnabar = 12.92 # From simulation 50/50 corundum
Icor_romarchite = 13.49 # From simulation 50/50 corundum

# =============================================================================
#     # Create python dictionnaries for corresponding index/phase, phase/db and phase/icor
# =============================================================================
dict_ph = {0: "anhydrite", 1: "cinnabar", 2: "romarchite"}

dict_db_ph = {"anhydrite": db_anhydrite, "cinnabar": db_cinnabar, "romarchite": db_romarchite}

dict_icor_ph = {"anhydrite": Icor_anhydrite, "cinnabar": Icor_cinnabar,"romarchite": Icor_romarchite}

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
list_combination = combs(list(range(3)))
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