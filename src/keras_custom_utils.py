# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:25:50 2021

@author: ebolger2

This lib is used to contain keras specific custom util functions
"""
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

def plot_history(history,name):
    '''
    Convenience method to plot loss and val_loss from Model history
    '''
    if not isinstance(name, str):
        raise TypeError('name must be a str type not %s' % (type(name)) )
    
    plt.figure(figsize=(6, 5))
    plt.title(name+' Model Loss')
    plt.plot(history["val_loss"], label='val_loss')
    plt.plot(history["loss"], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()
    
class CustomStopper(EarlyStopping):
    def __init__(self, monitor='val_loss',
             min_delta=0, patience=20, verbose=0, mode='min', start_epoch = 40): # add argument for starting epoch
        super(CustomStopper, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)