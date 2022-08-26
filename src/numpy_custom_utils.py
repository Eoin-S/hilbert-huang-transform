# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:15:15 2021

@author: ebolger2
"""
import numpy as np
import itertools

def n_closest(x, n, arr,mode='indexs'):
    ''' 
    This funciotn will return a list of n indexs clostest to the input value x
    Args:
        x: the value the function compares to
        n: number of values to aquire
        arr: numpy.array of values
    return:
        numpy.array object containing the indexs of the closest n values to x in the input arr
    TODO:
        * add check to insure n valus is less than or equal to lenght of arr
    '''
    indexarr = np.argpartition(abs(arr - x), n)[:n]
    if mode == 'indexs':
        return indexarr
    elif mode == 'values':
        return arr[indexarr]
    else:
        raise TypeError('mode can only be indexs or values, all other values are not valid.')
        
        
def repeating_elements(array,N):
    'Repeating elements of a list n times'
    return np.array(list(itertools.chain.from_iterable(itertools.repeat(x, N) for x in array)))

def drop_common(a,b):
    '''drop common values between two np.array objs, these arrays do not need to have all common values'''
    return np.array([i for i in a if i not in b])

def moving_average(x, w):
    '''
        Calculates moving average in specified window lenght
    '''
    return np.convolve(x, np.ones(w), 'valid') / w