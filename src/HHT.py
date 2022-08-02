# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:37:22 2021

@author: ebolger2
"""
from scipy.signal import hilbert, spectrogram
from scipy.sparse import csr_matrix
from PyEMD import EMD,EEMD
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from skimage.measure import block_reduce
import numpy.matlib

import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns

import os
from pathlib import Path

import scipy

def invEMD(imfs,res):
    return res + imfs.sum(axis=0)

def ihtrans(x):
    return -1*hilbert(x).imag

def hht(imfs,fs,freq_max=None,freq_min=None,FResol=None,MinThres = np.NINF):
    ''''''
    if ((freq_min is None) & (freq_max is None)):
        FRange = [0,fs/2]
    elif ((freq_min < 0) or (freq_min > freq_max)):
        raise ValueError('Input freq_min error')
    elif freq_max > (fs/2):
        raise ValueError('Input freq_max error')
    else:
        FRange = [freq_min,freq_max]
        
    if FResol is None:
        FResol = (FRange[1]-FRange[0])/100
    elif isinstance(FResol, (int, float)) !=True & (FResol > 0):
        raise ValueError('FResol must be an int or float')
    else:
        pass 
    
    # get size of input object
    n_samples  = imfs.shape[0]-1

    # setup frequency vector
    F = np.linspace(FRange[0], FRange[1], num=int(FRange[1]/FResol)+1)
    
    # setup time vector
    T = np.linspace(0, (1/fs)*n_samples, num=int((1/fs)*n_samples/(1/fs))+1)
    
    if imfs.ndim: imfs = np.expand_dims(imfs,axis =0)
        
    for n, imf in enumerate(imfs):
        
        sig = hilbert(imf)
        energy = abs(sig)**2 #.imag .real
        phaseAngle = np.angle(sig)

        #compute instantaneous frequency using phase angle
        omega  = np.gradient(np.unwrap(phaseAngle))

        #convert instantaneous frequency units to Hz
        omega = fs/(2*np.pi)*omega;

        # find out index of the frequency
        omegaIdx = np.floor((omega-F[1])/FResol)+1

        # generate distribution
        if n == 0:
            freqIdx = omegaIdx
            insf = omega
            inse = energy
        else:
            freqIdx = np.vstack((freqIdx,omegaIdx))
            insf = np.vstack((insf,omega))
            inse = np.vstack((inse,energy))
        break

    # filter out points not in the frequency range
    idxKeep = np.where( ( freqIdx>= 0 ) & ( freqIdx <= len(F) ), True, False )
    timeIdx = np.matlib.repmat(range(0,len(T)),imfs.shape[0],1)

    # check if vector or matrix
    if insf.ndim == 1:
        P = csr_matrix((insf[idxKeep], 
                        (freqIdx[idxKeep], timeIdx[0][idxKeep])), 
                       shape=(len(F),len(T) ))
    else:
        P = csr_matrix((insf[idxKeep], 
                        (freqIdx[idxKeep], timeIdx[idxKeep])), 
                       shape=(len(F),len(T) ))
    
    return T,F,P,insf,inse

def Hilbert_spectrum(s, fs, freq_div = 2, n_components = 4, frequency_resolution = 0.01, freq_min = 0, flatten = True):
    
    freq_max = fs/freq_div
    
    # decompose signal into imfs and residue
    emd = EMD(max_imfs=n_components)
    emd.emd(s)
    imfs, res = emd.get_imfs_and_residue()
    n_components = imfs.shape[0] + 2
    
    energy, raw = [], []
    for n, imf in enumerate( imfs ):

        T, F, P, insf, inse = hht(imf, fs, 
                                  FResol= frequency_resolution, 
                                  freq_min = freq_min, 
                                  freq_max = freq_max)
        
        # plt.scatter(T,insf, c = inse)
        
        energy.extend([P.toarray()])
        raw.extend([np.column_stack((T, insf, inse))])
    
    T, F, P, insf, inse = hht(res, fs, 
                              FResol= frequency_resolution, 
                              freq_min = freq_min, 
                              freq_max = freq_max)
    energy.extend([P.toarray()])
    raw.extend([np.column_stack((T, insf, inse))])
    # plt.scatter(T,insf, c = inse)
    
    energy = np.dstack(energy)
    org_shape = energy.shape
    
    if flatten:
        return T, F, block_reduce(energy, block_size=(1, 1, org_shape[2]), 
                                  func=np.max).reshape(org_shape[0],org_shape[1]), np.dstack(raw)
    else:
        return T, F, energy, np.dstack(raw)