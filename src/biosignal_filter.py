# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:53:22 2020

@author: ebolger2
"""

import numpy as np
import pywt
from scipy import signal


def denoise_biowave( y, fs = 125, mode = 'orginal_dc' ):
    '''
    This function denoises and filters the inputted biological waveform. It does this 
    my initially upsampling the input signal to 1kHz, (this is done to make the process 
    invariant to the probable changes of sampling frequency of the input signal). The 
    signal is then decomposed by means of the Discrete Wavelet Transform (DWT) with the 
    Daubechies 8 (db8) mother wavelet, with 10 decomposition levels. Then, the components 
    corresponding to the very low-frequencyrange of 0–0.25 Hz (associated with the 
    baseline wandering) and ultrahigh frequencies between 250 and 500 Hz (associated with 
    the power-line harmonics and the muscular activity artifacts)are eliminated by zeroing 
    their decomposition coefficients.This process is described below:
    
     level 0, LP: 250.0, HP: 500.0, -(Zeroing, associated with power-line harmonics and muscular activity artifacts)
     level 1, LP: 125.0, HP: 250.0, 
     level 2, LP: 62.5, HP: 125.0, 
     level 3, LP: 31.25, HP: 62.5, 
     level 4, LP: 15.625, HP: 31.25, 
     level 5, LP: 7.812, HP: 15.625, 
     level 6, LP: 3.906, HP: 7.812, 
     level 7, LP: 1.955, HP: 3.906, 
     level 8, LP: 0.976, HP: 1.953, 
     level 9, LP: 0.488, HP: 0.976,
     level 10, LP: 0.244, HP: 0.488, 
     level 10, LP: 0, HP: 0.2441 -(Zeroing, associated with baseline wandering)
    
    The final step is to perfrom conventional wavelet denoising on the remaining decomposition 
    coefficients using soft Rigrsure thresholding strategy.

    Args:
        y (numpy.array):
        fs (int): sampling frequency vector y was sampled at.
        mode (str): defines the operation, default mode: orginal_dc (orginal dc level of signal is maintained @ output),
            zero_dc (the output signal will have a dc level of zero) & raw 
    
    Return:
        returns filtered and denoised biological signal with different dc levels depending on mode used.
        
    Raises:
        ValueError: Raises an exception if y (input wavefrom) is not large enough for wavelet decomposition
        ValueError: Raises an exception if mode is not orginal_dc, zero_dc or raw
        
    Todo:
        * processing pipeline only works for signals in positive domain, this should be adjusted for all domains.
    
    Notes:
        This function implements the following denoising and filtering wavelet preprocessing 
        pipeline described in:
            Mohammad Kachuee, Mohammad Mahdi Kiani, Hoda Mohammadzade, and Mahdi
            Shabany. Cuffless blood pressure estimation algorithms for continuous health-care
            monitoring. IEEE Transactions on Biomedical Engineering, 64(4):859–869, 2016.
            
        Discussion on optimal mother wavelet and decomposition level selection:
            B. N. Singhet al., "Optimal selection of wavelet basis function applied 
            to ECG signal denoising" Digital Signal Process., vol. 16, no. 3,pp. 275–287, 2006.
            
        Wavelet denoising with soft Rigrsure thresholding strategy:
            * D. L. Donohoet al., “Ideal spatial adaptation by wavelet shrinkage,”Biometrika, 
              vol. 81, no. 3, pp. 425–455, 1994.
            * D. Donoho, “De-noising by soft-thresholding,”IEEE Trans. Inf. Theory,
              vol. 41, no. 3, pp. 613–627, May 1995.

    '''
    if mode not in ['orginal_dc','zero_dc','raw']:
        raise ValueError('mode can only be one of 2 options: raw or raw_orginal_dc')
    
    ymin = np.min(y) # required for orginal_dc mode
    fup=1000 # required for method
    wlevel = 11 # required for method
    
    n = int((len(y)/fs)*fup)
    if pywt.dwt_max_level(n, 'db8') <= 10:
        raise ValueError('waveform segiment lenght to small')
        
    y = signal.resample(y, n)
    coeff = pywt.wavedec(y,wavelet ='db8',level = wlevel)
    # define the component levels to Zero 250~500Hz, 0~0.25Hz
    zlevel = [0,11]
    for z in zlevel:
        coeff[z] = np.zeros(coeff[z].shape)
        
    n = y.shape[0]
    wav = [coeff[0]]
    # Wavelet denoising with soft Rigrsure thresholding strategy
    for c in coeff[1:-1]:
        t = (np.std(c)/np.sqrt(n))*np.sqrt(2*np.log(n))
        wav.extend([pywt.threshold(c, t, 'soft')])
    wav.extend([coeff[-1]])
    # recover signal by the construction of the decomposition & downsample to orginal fs
    dy = signal.decimate( pywt.waverec(wav,wavelet ='db8') ,int(fup/fs))
    dymin = np.min(dy)
    
    # handle DC component of orginal signal and return
    if mode == 'orginal_dc':
        if dymin < 0:
            dy += abs(dymin)
        else:
            dy -= dymin
        return dy + ymin
    elif mode == 'zero_dc':
        if dymin < 0:
            dy += abs(dymin)
        else:
            dy -= dymin
        return dy
    else: 
        return dy



