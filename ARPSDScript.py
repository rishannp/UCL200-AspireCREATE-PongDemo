# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:39:47 2026

@author: uceerjp

Implementing the described in https://doi.org/10.3758/BF03200585
"""

import scipy
from scipy.io import loadmat 

# CONFIG
fs = 256
winlen = fs
hop = round(0.04 * fs)

 
data = loadmat("P01S2R1.mat") # data['signal'] is TxC

### First: write a loop that takes small chunks of data and loops through the data

for i in data['signal'].shape[0]:
    

