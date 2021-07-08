import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, argrelmin, find_peaks
def integral(data):
    from scipy.integrate import simps
    area = simps(data,dx=1)
    return area

def activity(signal):
    activity = np.var(signal)
    return activity

def mobility(signal):
    import numpy as np
    der = np.diff(signal)
    mobility = np.sqrt(activity(der)/activity(signal))
    return mobility

def complexity(signal):
    import numpy as np
    der = np.diff(signal)
    complexity = np.sqrt(mobility(der)-mobility(signal))
    return complexity


def zerocrossing(data):
    zero_cross = np.nonzero(np.diff(data > 0))[0]
    return zero_cross.size

def variability(data):
    var = np.std(data)/np.mean(data)
    return var 

def relative_power(data, fs):
    f, Pxx = signal.welch(data, fs, nperseg=2*fs)
    #plt.plot(f,Pxx)
    idx_tmp = np.logical_and(f >= 0, f <= 8)
    idx_tmp2 = np.logical_and(f >= 1, f <= 2.25)
    """
    print('indici')
    print(idx_tmp)
    print(Pxx[idx_tmp])
    print(idx_tmp2)
    print(Pxx[idx_tmp2])
    """
    from scipy.integrate import simps

    # Frequency resolution
    freq_res = f[1] - f[0]  # = 1 / 4 = 0.25
    # Compute the power by approximating the area under the curve
    total_power = simps(Pxx[idx_tmp], dx=freq_res)
   # print('total power: %.3f uV^2' % total_power)
    signal_power = simps(Pxx[idx_tmp2], dx=freq_res)
    #print('signal power: %.3f uV^2' % signal_power)
    return signal_power/total_power


data = pd.read_excel('waves_notResized.xlsx')

for index, row in data.iterrows():
    #print(row)
    print(index)

    sig = row[1::]
    sig = sig.values.flatten()
    fs = 100
    
    sig_notnan = sig[np.logical_not(np.isnan(sig))]
    sig_zeromean = sig_notnan - np.mean(sig_notnan)
    fig = plt.figure()
    plt.plot(sig_zeromean)
    
    FD = np.diff(sig_zeromean)
    SD = np.diff(FD)
    plt.plot(FD)
    plt.plot(SD)
    peaks_FD,_ = find_peaks(FD)
    peaks_SD,_ = find_peaks(SD)
    peaks_sig,_ = find_peaks(sig_zeromean)
    min_FD = argrelmin(FD)
    min_FD= min_FD[0]
    min_SD = argrelmin(SD)[0]
    min_sig = argrelmin(sig_zeromean)[0]
    plt.plot(peaks_sig,sig_zeromean[peaks_sig],'o')
    plt.plot(peaks_FD,FD[peaks_FD],'o')
    plt.plot(min_FD,FD[min_FD],'o')
    plt.plot(min_SD,SD[min_SD],'o')
    plt.plot(min_sig,sig_zeromean[min_sig],'o')
    plt.plot(peaks_SD,SD[peaks_SD],'o')
    
    plt.show()
    # plt.plot(FD)
    # plt.plot(peaks_FD,FD[peaks_FD],'o')
    # plt.plot(min_FD,FD[min_FD],'o')
    # plt.show()

    area_sig = integral(sig_zeromean)
    print(area_sig)
    zero_cr_sig = zerocrossing(sig_zeromean)
    zero_cr_FD = zerocrossing(FD)
    zero_cr_SD = zerocrossing(SD)
    print(zero_cr_sig,zero_cr_SD,zero_cr_FD)

    # Delta_T: difference in time (samples) between first and second peak of the PPG wave
    # if second peak does not exist Delta_t is given by difference between first PPG signal peak 
    # and second first derivative peak. Elgendi 2012
    delta_t = 0

    if len(peaks_sig)>1:
        delta_t = peaks_sig[1]-peaks_sig[0]
    else:
        if len(peaks_FD)>1:
            delta_t = peaks_FD[1]-peaks_sig[0]
    print('Delta T')
    print(delta_t/fs*1000)


    sys_amp_peak = 0

    if len(peaks_sig)>0:
        sys_amp_peak = sig_zeromean[peaks_sig[0]]
    print(sys_amp_peak)
    

    sys_time_peak = 0
    if len(peaks_sig)>0:
        sys_time_peak = peaks_sig[0]
    print(sys_time_peak)

    dias_amp_peak = 0

    dias_time_peak = 0

    crest_time = 0

    peak_fft = 0
    
