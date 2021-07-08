import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from skimage.restoration import denoise_wavelet, estimate_sigma
from scipy import signal, interpolate
from scipy.signal import filtfilt, butter
from scipy.signal import argrelextrema, argrelmin, find_peaks
import pywt
from pywt import wavedec,waverec
import pdb

data = pd.read_excel("shuffle_norm_data.xlsx")
data.insert(402,'Class_',data['Class'])
data_4sec = pd.DataFrame()
print(data.shape)

data_4sec = data.iloc[:,1:402]
data_tmp = data.iloc[:,402::]
data_tmp.columns = data_4sec.columns.values
data_4sec = data_4sec.append(data_tmp)

print(data_4sec)

data_4sec.to_excel('norm_data_4sec.xlsx')
"""
for x in range(1,data.shape[1],400):

 
    print(data_2sec)
"""
