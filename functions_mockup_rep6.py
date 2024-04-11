# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:22:38 2022

@author: Victor.POLINE
modified by Ravi Purushottam
"""

import numpy as np

# =============================================================================
# Functions slightly adapted to work with array and arguments 
# modified to be compatible with multiprocessing library
# =============================================================================
# Function using Bragg law :
def dto2th_array(x, lambd=0.619921):
    return((180/np.pi)*2*np.arcsin(lambd/(2*x)))

# U, V, W from LaB6 refinement, R from article
def pv_peak_dec_array_TCH(x, x0, Tc, h, x1):
    # ============================ CONSTANTS ====================================== 
    lambd=0.619919
    U=-0.015755
    V=0.009865
    W=0.000866
    R=205
    # =============================================================================     
    Y = (180/np.pi)*(0.9*lambd/Tc)
    H_G = U*(np.tan(x/2*(np.pi/180)))**2+V*np.tan(x/2*(np.pi/180))+W
    H_G = np.sqrt(H_G)
    H_L = Y / np.cos(x/2*(np.pi/180))
    # Thomson-Cox-Hastings for H and eta calculation
    H = (H_G**5 + 2.69269*H_G**4*H_L + 2.42843*H_G**3*H_L**2 + 4.47163*H_G**2*H_L**3 + 0.07842*H_G*H_L**4 + H_L**5)**(1/5)
    eta = 1.36603*H_L/H - 0.47719*(H_L/H)**2 + 0.11116*(H_L/H)**3
    al = 2/(np.pi*H)
    bl = 4/H**2
    ag = (2/H)*(np.sqrt(np.log(2)/np.pi))
    bg = (4*np.log(2))/(H**2)
    x0=x0+((h/R)*np.cos((x0/2)*(np.pi/180)))*(180/np.pi)
    x= x.reshape((len(x),1))
    x0= x0.reshape((1,len(x0)))
    xx = (x-x0)**2
    num0 = eta*al
    denom0 = 1 + np.einsum("i,ij->ij",bl,xx)
    first = num0.reshape((len(num0),1)) / denom0
    num1 = (1-eta)*ag
    second = np.einsum("i,ij->ij",num1,np.exp(np.einsum("i,ij->ij",-bg,xx)))
    final_prod = np.einsum("j,ij->ij", x1, first + second)
    return final_prod


def random_ddrx_array(list_tth, ppp, hp, Tcp, list_db, list_icor):
    """time reduced by a factor of 100"""
    tth_ph_all = np.zeros(len(list_tth))
    for k in range(len(list_db)): # For each phase, we calculate its diagram
        if ppp[k] == 0:
            continue
        db_ph = list_db[k]
        # print(ppp[k], Tcp[k], hp[k])
        pv_array = pv_peak_dec_array_TCH(list_tth, db_ph[:, 0], Tcp[k], hp[k], db_ph[:, 1])
        tth_ph = np.sum(pv_array, axis=1)            
        norm = max(tth_ph) # This step normalises the diagram by its Icor and proportion
        tth_ph = (100*tth_ph/norm)*ppp[k]*list_icor[k]
        tth_ph_all = tth_ph_all + tth_ph
    return tth_ph_all


def MP_intermediate(list_tth, list_db, list_icor, sizeTest, l_comb):
    """time reduced by a factor of 100"""
    print()
    length_list = len(list_tth)
    n_ph = len(list_db)
    dataset = np.empty((0, length_list+3*n_ph), float)

    y_Tc_constant = np.array([[2318.74, 358.24, 526.23]])
    # =============================================================================
    #     # First all is free
    # =============================================================================
    for ih in range(sizeTest):
        y_pp = np.zeros((1,len(list_db)))
        y_Tc = np.zeros((1,len(list_db)))
        y_h = np.zeros((1,len(list_db)))
        # Changing the value of the list for the given list_combination
        for lc0 in l_comb:
            # Building the random phase fractions
            y_pp[0, lc0] = np.random.rand(1)
            # # Building the random cristallite sizes
            y_Tc[0, lc0] = y_Tc_constant[0, lc0]
            # Building the random phase positions
            y_h[0, lc0] =  np.random.randint(-100, 100) / 1000
        y_pp = np.round(y_pp/y_pp.sum(axis=1), 3) # Sum of list = 1
        ddrx = random_ddrx_array(list_tth, y_pp[0], y_h[0], y_Tc[0], list_db, list_icor)
        # Create the line of the dataset with the diagram and the parameters after
        toAppend = np.concatenate((ddrx, y_pp, y_Tc, y_h), axis=None)
        toAppend = toAppend.reshape(1, toAppend.shape[0])
        # Append the line to the previous ones in the dataset
        dataset = np.append(dataset, toAppend, axis=0)
    return dataset

def MP_intermediate_1(list_tth, list_db, list_icor, sizeTest, l_combo):
    """time reduced by a factor of 100"""
    length_list = len(list_tth)
    n_ph = len(list_db)
    lc0 = l_combo
    dataset = np.empty((0, length_list+3*n_ph), float)
    
    y_Tc = np.array([[2318.74, 358.24, 526.23]])
    
    for _ in range(sizeTest):
        # Building the random phase fractions
        y_pp = np.random.randint(0, 200, [1,n_ph]) / 1000
        # Changing the value of the list for the 3 major phases (given by list_combination)
        y_pp[0, lc0] = np.random.randint(200, 1000) / 1000
        y_pp = np.round(y_pp/y_pp.sum(axis=1), 3) # Sum of list = 1
        
        # Building the random cristallite sizes
        # y_Tc = np.random.randint(5000, 10000, [1,n_ph])
        
        # Building the random phase positions
        y_h = np.random.randint(-100, 100, [1,n_ph]) / 1000
        
        ddrx = random_ddrx_array(list_tth, y_pp[0], y_h[0], y_Tc[0], list_db, list_icor)
        # Create the line of the dataset with the diagram and the parameters after
        toAppend = np.concatenate((ddrx, y_pp, y_Tc, y_h), axis=None)
        toAppend = toAppend.reshape(1, toAppend.shape[0])
        # Append the line to the previous ones in the dataset
        dataset = np.append(dataset, toAppend, axis=0) 
    return dataset

def MP_intermediate_2(list_tth, list_db, list_icor, sizeTest, l_combo):
    """time reduced by a factor of 100"""
    length_list = len(list_tth)
    n_ph = len(list_db)
    lc0, lc1 = l_combo
    dataset = np.empty((0, length_list+3*n_ph), float)

    y_Tc = np.array([[2318.74, 358.24, 526.23]])
    
    for _ in range(sizeTest):
        # Building the random phase fractions
        y_pp = np.random.randint(0, 100, [1,n_ph]) / 1000
        # Changing the value of the list for the 3 major phases (given by list_combination)
        y_pp[0, lc0] = np.random.randint(600, 1000) / 1000
        y_pp[0, lc1] = np.random.randint(600, 1000) / 1000
        y_pp = np.round(y_pp/y_pp.sum(axis=1), 3) # Sum of list = 1
        
        # Building the random cristallite sizes
        # y_Tc = np.random.randint(5000, 10000, [1,n_ph])
        
        # Building the random phase positions
        y_h = np.random.randint(-100, 100, [1,n_ph]) / 1000
        
        ddrx = random_ddrx_array(list_tth, y_pp[0], y_h[0], y_Tc[0], list_db, list_icor)
        # Create the line of the dataset with the diagram and the parameters after
        toAppend = np.concatenate((ddrx, y_pp, y_Tc, y_h), axis=None)
        toAppend = toAppend.reshape(1, toAppend.shape[0])
        # Append the line to the previous ones in the dataset
        dataset = np.append(dataset, toAppend, axis=0) 
    return dataset


def MP_intermediate_3(list_tth, list_db, list_icor, sizeTest, l_combo):
    """time reduced by a factor of 100"""
    length_list = len(list_tth)
    n_ph = len(list_db)
    lc0, lc1, lc2 = l_combo
    dataset = np.empty((0, length_list+3*n_ph), float)
    
#     y_Tc = np.ones((1,n_ph))*20000.
    y_Tc = np.array([[2318.74, 358.24, 526.23]])
    
    for _ in range(sizeTest):
        # Building the random phase fractions
        y_pp = np.random.randint(0, 100, [1,n_ph]) / 1000
        # Changing the value of the list for the 3 major phases (given by list_combination)
        y_pp[0, lc0] = np.random.randint(200, 1000) / 1000
        y_pp[0, lc1] = np.random.randint(200, 1000) / 1000
        y_pp[0, lc2] = np. random.randint(200, 1000) / 1000
        y_pp = np.round(y_pp/y_pp.sum(axis=1), 3) # Sum of list = 1
        
        # Building the random cristallite sizes
        # y_Tc = np.random.randint(5000, 10000, [1,n_ph])
        
        # Building the random phase positions
        y_h = np.random.randint(-100, 100, [1,n_ph]) / 1000
        
        ddrx = random_ddrx_array(list_tth, y_pp[0], y_h[0], y_Tc[0], list_db, list_icor)
        # Create the line of the dataset with the diagram and the parameters after
        toAppend = np.concatenate((ddrx, y_pp, y_Tc, y_h), axis=None)
        toAppend = toAppend.reshape(1, toAppend.shape[0])
        # Append the line to the previous ones in the dataset
        dataset = np.append(dataset, toAppend, axis=0) 
    return dataset


def MP_Pattern_rebuilding(list_tth, list_db, list_icor, y_predict_pp, y_predict_h):
    
    length_list = len(list_tth)
    nb_ph = len(list_db)
    
    y_Tc = np.array([2157.63, 420.87, 512.48])
    y_pp = y_predict_pp
    y_h = y_predict_h
    ddrx = random_ddrx_array(list_tth, y_pp, y_h, y_Tc, list_db, list_icor)
    ddrx = ddrx.reshape(1, ddrx.shape[0])
        
    return ddrx

def Rwp(y, yc): 
    # y is an ndarray of measured intensities (y.shape = (X,))
    # yc is an ndarray of calculated intensities (yc.shape = y.shape)
    rwp = 0
    num = 0
    den = y.sum()
    if y.sum()==0 or yc.sum()==0:
        rwp = 0
    else:
        s = y.sum() / yc.sum() # scale factor between measured and calculated
        for i in range(y.shape[0]):
            if y[i] == 0:
                res = 0
            else:
                w = 1/y[i]
                res = w*(y[i]-s*yc[i])**2
            num += res
        
        rwp = 100 * (num/den)**0.5
            
    return rwp