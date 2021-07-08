import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
#from skimage.restoration import denoise_wavelet, estimate_sigma
from scipy import signal, interpolate
from scipy.signal import filtfilt, butter
from scipy.signal import argrelextrema, argrelmin, find_peaks
import pywt
from pywt import wavedec,waverec
import pdb
from math import log
import glob
import ntpath
import Feature_extraction



def interpolation(data,fs,fe_interpolation):

    from scipy import interpolate
    
    t = len(data)/float(fs)
    # print(t)
    # print(len(data))
    # print(fs)
    
    #vec_t = np.arange(0,t,1/float(fs))
    vec_t = np.linspace(0,t,len(data))
    # print(len(vec_t))
    # print(vec_t)
    # plt.plot(data)
    # plt.show()
    vec_t_int = np.arange(0,t,1/float(fe_interpolation))
    f = interpolate.interp1d(vec_t, data, kind='linear',bounds_error=False, fill_value='extrapolate')
    data_interpolated = f(vec_t_int)

    return data_interpolated,fe_interpolation



def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [high], btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def wavelet_filter(data,fs,waveletname):
    level = 1
    zero_lev = []
    f = fs*pow(2,-1-level)
    while f > 0.3:
        if f>15:
            zero_lev.append(level)
        level += 1
        f = pow(2,-1-level)*fs
    #print('level')
    #print(level)
    #print(zero_lev)

    cA = []
    cD = []
    #fig, axarr = plt.subplots(nrows=12, ncols=2, figsize=(12,12))
    for ii in range(0,level):
        (data, coeff_d) = pywt.dwt(data, waveletname)
        cA.append(data)
        cD.append(coeff_d)
        """
        axarr[ii, 0].plot(data, 'r')
        axarr[ii, 1].plot(coeff_d, 'g')
        axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
        axarr[ii, 0].set_yticklabels([])
        if ii == 0:
            axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
            axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
        axarr[ii, 1].set_yticklabels([])
        """
    #plt.tight_layout()
    #plt.show()
    #print(cD)
    coeff = []
    coeff.append(np.zeros_like(cA[level-1]))
    for i in range(level-1,-1,-1):
        coeff.append(cD[i])
    rec = pywt.waverec(coeff, waveletname)
    return rec


path = r'C:\Users\zanel\Desktop\Axelife\DATA\pOpmetre\BD_Ref\Signals\BD_Ref' # use your path
all_files = glob.glob(path + "/*.csv")
results = pd.DataFrame()
norm_data = pd.DataFrame()
c = []
info = []
nbr = []
norm_results = pd.DataFrame()
count = 0
waves = []
w=pd.DataFrame()
chuncks = pd.DataFrame()
df = pd.read_excel(r'C:\Users\zanel\Desktop\Axelife\DATA\pOpmetre\BD_Ref\class.xlsx', index_col=None, header=0)
name = df['Name']
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('PPG_Classes.pdf') as pdf:
    for filename in all_files:
        head, tail = ntpath.split(filename)
        print(filename)

        for index in range(len(name)):
            #print(index)
            if str(name[index])==tail:
                #print(name[index])
                count += 1
                df = pd.read_csv(filename)
                df.columns = ['hand','feet']
                fs = 1000
                sig = df['hand'].values.flatten()
                sig = sig - np.mean(sig)
                sig = sig
                reversed_sig = sig
                waveletname = 'db4'
                rec = wavelet_filter(sig,fs,waveletname)
                low = 0.05
                high = 15
                order = 3
                filtered = butter_bandpass_filter(sig,low,high,fs,order)
                # smoothed = signal.savgol_filter(filtered,21,2)
                sig100Hz,_ = interpolation(filtered,fs,100)
                resempled = 100
                waves= pd.DataFrame(Feature_extraction.segmentation(sig100Hz,notfilt=None, nbr_samples=None,shifted=False))
                #fig = plt.figure (dpi=100)
                for index, row in waves.iterrows():
                    info.append(tail)
                # plt.title(tail)
                # plt.show()
                # #pdf.savefig()
                # plt.close(fig)
                w = w.append(waves)
                #w_notFilt = w_notFilt.append(waves_notFilt)
                nbr.append(len(waves))
                # wind = 100

                    # for i in range(0,len(sig)-wind):

                    #     sig_chunck = list(sig[i:i+wind])
                    #     # print(sig_chunck)
                    #     # plt.plot(sig_chunck)
                    #     # plt.show()
                    #     g=pd.DataFrame(sig_chunck)
                    #     chuncks = chuncks.append(g.T)
                
                    # print(chuncks)
                    # plt.plot(sig)
                    # plt.show()
    #w_notFilt.to_excel('Waves_notFilt_BD_Ref.xlsx')
    w.to_excel('waves_reallenght_classfile.xlsx') 
    # w_info = pd.DataFrame()
    # w_info['info']=info
    # w_info['nbr']=nbr

    print(w)
    wave_name = pd.DataFrame(info)
    wave_name.to_excel('names_wave.xlsx')