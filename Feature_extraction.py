import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sys, os
# from skimage.restoration import denoise_wavelet, estimate_sigma
from scipy import signal, interpolate
from scipy.signal import filtfilt, butter
from scipy.signal import argrelextrema, argrelmin, find_peaks
import pywt
from pywt import wavedec,waverec
import pdb
from math import log
# for local maxima
#argrelextrema(x, np.greater)

def variability(data):
    var = np.std(data)/np.mean(data)
    return var 

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / (fs/2)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y

def mobile_average(signal, l):
    y = []
    loop=len(signal)-l-1 # o meno 2?
    for i in range(0,len(signal)):
        if i<loop:
            y.insert(i,sum(signal[i:i+l])/l)
        else:
            y.insert(i,sum(signal[i:])/len(signal[i:]))
    return y

def derivative(signal):
    y=[]
    for i in range(len(signal)-1):
       
        y.insert(i,(signal[i+1]-signal[i]))
    return y

def corr_peaks(data,fs):
    acf = np.correlate(data,data, 'full')
    idx = np.argmax(acf)
    acf = acf[idx:idx+3*fs]
    peaks,_ = find_peaks(acf)
    if len(peaks)<2:
       tmp = pd.DataFrame([{'ACF First peak': peaks[0],'ACF Second peak': float('nan')}])
    else:  
        tmp = pd.DataFrame([{'ACF First peak': peaks[0],'ACF Second peak': peaks[1]}])

    return tmp

def eigenvalues(data):

    from numpy import linalg

    acf = np.correlate(data,data, 'full')
    idx = np.argmax(acf)
    acf_100lags = acf[idx: idx +100]
    from scipy.linalg import toeplitz
    cov_mat= toeplitz(acf_100lags)
    eigenValues,_ = linalg.eig(cov_mat)
    #print(eigenValues)
    eigenValues = np.sort(eigenValues)
    eigenValues = eigenValues[::-1]
    #I want to analyse the first nine eigenvalues
    data = [{'λ1': eigenValues[0], 'λ2': eigenValues[1], 'λ3':eigenValues[2], 'λ4':eigenValues[3], 'λ5':eigenValues[4], 'λ6':eigenValues[5], 'λ7':eigenValues[6], 'λ8':eigenValues[7], 'λ9':eigenValues[8]}]
    
    return pd.DataFrame(data)

def segmentation(data,notfilt=None, nbr_samples=None,shifted = False):
    from scipy import signal, interpolate
    import numpy as np
    fs = 100
    try:
        
        # found the mini in the signal
        pos_min = np.transpose(argrelmin(data))
        #plt.plot(pos_min,smoothed[pos_min],'o')
        # calculate derivative and its peaks
        der = derivative(data)
        peaks_der,_ = find_peaks(der)
        # threshold to select valid peak of the derivative
        threshold =np.mean(der)+np.std(der)
        validepeak = []
        pos_valid_peak = []
        y=0
        # selection of valid derivative peaks
        for i in range(0,len(peaks_der)):
            index = peaks_der[i]
            if der[index]>threshold:
                validepeak.insert(y,der[index])
                pos_valid_peak.insert(y,index)
                y=y+1
                

        # plt.plot(pos_valid_peak, validepeak)
        # plt.show()

        j = []
        segment = []
        tmp = []
        #I am searching for the first global minimum before the first derivative peak
        j = [i for i in pos_min if i < pos_valid_peak[0]]
        
        # if there is no minimum before the first derivative peak I start to segment from 0th element
        if j==[]:
            segment.insert(0,0)
        else:   
            tmp = (np.amin(data[np.asarray(j)]))
            tmp2 = [i for i in j if data[i] == tmp]
            segment.insert(0,(tmp2[0][0]))
        
        #now I serch the global minimum of the signal betwenn two adiacent derivative peaks
        for p in range(len(pos_valid_peak)-1):
            
            tmp = []
            j = [i for i in pos_min if i > pos_valid_peak[p] and i < pos_valid_peak[p+1]]
            #import pdb;pdb.set_trace() 
            #tmp2 global minimum of the function between two derivative peaks
            tmp = (np.amin(data[np.asarray(j)]))
            tmp2 = [i for i in j if data[i] == tmp]
            segment.insert(p+1,(tmp2[0][0]))
        
          
        wave = []
        wave_notFilt = []
        index_tmp = 0
        #print('len segment', len(segment))
        for i in range(len(segment)-1):
            #print(i)
            start = segment[i]
            stop = segment[i+1]
            if nbr_samples is None:
                wave.insert(i,list(data[start:stop]))
                if notfilt is not None:
                    wave_notFilt.insert(i,list(notfilt[start:stop]))
            
            if nbr_samples is not None:
                from sezanel_tools import interpolation
                from sezanel_tools import norm
                #import pdb; pdb.set_trace()
                time = len(data[start:stop])/fs
                wave_100 = interpolation(data[start:stop],fs,nbr_samples/time)
                norm01,_ = norm(wave_100)
                #import pdb; pdb.set_trace()
                wave.insert(i,norm01.values.flatten())
                #wave.insert(i,list(signal.resample(data[start:stop],nbr_samples)))
                # vec_t = np.linspace(0,len(data[start:stop])/100,len(data[start:stop]))
                # vec_t_Interpolation = np.linspace(0,1,100)
                if notfilt is not None:
                    wave_notFilt.insert(i,list(signal.resample(notfilt[start:stop],nbr_samples)))
                #f = interpolate.interp1d(vec_t, data[start:stop], kind='linear')
                #data_interpolation = f(vec_t_Interpolation)
                # plt.plot(signal.resample(data[start:stop],nbr_samples))
                # plt.plot(data[start:stop])
                # plt.plot(data_interpolation)
                # plt.show()
            if shifted == True:
                if len(data[start:stop])<100:
                    shift = 100 - len(data[start:stop])
                    if len(data)>stop+shift:
                        if shift > 0:
                            if len(data)>stop+shift+100:
                                # print(len(data[start:stop+shift]))
                                wave.insert(index_tmp,list(data[start:stop+shift]))
                                for x in range(0,100,10):
                                    sig_tmp = data[start+x:stop+shift+x]
                                    # print(x)
                                    # print(len(sig_tmp))
                                    # plt.plot(sig_tmp)
                                    # plt.show()
                                    wave.insert(index_tmp,list(sig_tmp))
                                    index_tmp =index_tmp+1


        #print('Nbr waves',len(wave))
        #print(waves)
        #plt.figure()
        # for i in range(len(wave)):
        #     plt.plot(wave[i])
        #     #plt.plot(wave_notFilt[i])
        # plt.show()
    except Exception as e:
        print(e)
        wave = []
        print('Failed in extracting PPG')   
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    return wave
   
def entropy(signal):

    data = np.asarray(signal-np.min(signal))
    n_data = len(data)
    counts = np.bincount(data.astype(int))
    probability = counts/n_data
    n_classes = np.count_nonzero(probability)

    if n_classes<=0:
        return 0
    ent = 0
    for i in probability:
        if i !=0:
            ent -= i*log(i)
    return ent

def peak_valid(data):
    peaks_der,_ = find_peaks(data)
    # threshold to select valid peak of the derivative
    threshold = np.mean(data)+np.std(data)
    validepeak = []
    pos_valid_peak = []
    y=0
    # selection of valid derivative peaks
    for i in range(0,len(peaks_der)):
        index = peaks_der[i]
        if data[index]>threshold:
            validepeak.insert(y,data[index])
            pos_valid_peak.insert(y,index)
            y=y+1
    tmp = len(validepeak)
    if validepeak == []:
        validepeak = 0
        pos_valid_peak = 0
        tmp = 0
        
    return validepeak, pos_valid_peak, tmp

def zerocrossing(data):
    zero_cross = np.nonzero(np.diff(data > 0))[0]
    return zero_cross.size

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

def integral(data):
    from scipy.integrate import simps
    area = simps(data,dx=1)
    """
    plt.plot(data)
    print(area)
    plt.show()
    """
    return area

def activity(signal):
    activity = np.var(signal)
    return activity
def mobility(signal):
    der = derivative(signal)
    mobility = np.sqrt(activity(der)/activity(signal))
    return mobility
def complexity(signal):
    der = derivative(signal)
    complexity = np.sqrt(mobility(der)-mobility(signal))
    return complexity

def hjorth_selection(signal, window, shift):

    # INPUT : signal (data to analyse), window (length of the window in samples),
    #  shift(number of sampling used to moving the window)
    #   return: averaged complexity, mobility and activity (Hjorth paramteters),
    #   SQI binaty function (1 good signal, 0 bad signal)
    #
    stop = len(signal)-window
    H_m = []
    H_c = []
    count = 0
    for i in range(0,stop,shift):
        tmp = signal[i:i+window]
        H_m.insert(count,mobility(tmp))
        H_c.insert(count,complexity(tmp))
        count = count+1

    #threshold used to implement SQI function
    mob_up_th = np.mean(H_m) + 1.4
    mob_low_th = np.mean(H_m) - 1
    comp_th = np.mean(H_c) + 3
    SQI=[]

    for i in range(len(H_m)):
        if H_m[i]>mob_low_th and H_m[i]<mob_up_th and H_c[i]<comp_th:
            SQI.insert(i,1)
        else:
            SQI.insert(i,0)


 # I would like to implement it. Insted of returning SQI, I would like to return already the segmeted signal

    return SQI

def norm(data):
    from sklearn.preprocessing import MinMaxScaler
    norm = pd.DataFrame(data)
    scaler = MinMaxScaler()
    norm_sig01 = pd.DataFrame(scaler.fit_transform(norm))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    norm_sig11 = pd.DataFrame(scaler.fit_transform(norm))
    #plt.plot(norm_sig01)
    #plt.plot(norm_sig11)
    return norm_sig01,norm_sig11

def scalogram(data,fs,name):
    #scalogram constructed taking into consideration interesting Fband 0.5-3Hz
    import pywt
    import numpy as np
    fc = pywt.central_frequency('gaus1')
    dt = 1/fs
    #s  = scal2frq(f,'morl',dt)
    scale_min = fc/(10*dt)
    scale_max = fc/(0.5*dt)
    print('SCALE')
    print(scale_max)
    print(scale_min)
    scale = np.arange(scale_min,scale_max)
    c, freqs=pywt.cwt(data,scale,'gaus1')
    print('COEFFICIENTI WAVELET')
    coef = c[1:]
    print(coef.shape)
    h = np.abs(np.inner(coef,coef))
    SC = np.divide(100*h,sum(h))
    print(sum(h))
    X = np.arange(0,(len(data)-1)/fs,(len(data)/fs)/len(SC))
    print(len(X))
    print(len(data)/fs)
    Y = np.arange(scale_min,scale_max,(scale_max-scale_min)/len(SC)) 
    print(len(Y))
    plt.contourf(X, Y, SC)
    plt.show()

def feature_extraction(data,fs,name):
   
    #sig01,sig11 = norm(data)
    tmp_peak, tmp_index, nbr_peaks= peak_valid(data)
    features = pd.DataFrame(data).describe().transpose()
    features['Signal'] = name
    # if name[0] == 'B':
    #     features['Class']= 1
    
    # if name[0] == 'M':
    #     features['Class'] = 0

    features['Zerocrossing'] = zerocrossing(data)
    features['Entropy'] = entropy(data)
    #features['Relative Power'] = relative_power(data,fs)
    features['Activity'] = activity(data)
    features['Mobility'] = mobility(data)
    features['Complexity'] = complexity(data)
    features['Number Peaks'] = nbr_peaks
    features['STD Peak interval'] = np.std(tmp_index)
    features['STD Peak amplitude']= np.std(tmp_peak)
    features['Area']=integral(data)
    features['Variability']=variability(data)
    features = pd.concat([features, eigenvalues(data)], axis = 1)
  #  features = pd.concat([features, corr_peaks(data,fs)], axis = 1)
    return features

def wavelet_filter(data,fs,waveletname):
    level = 1
    zero_lev = []
    f = fs*pow(2,-1-level)
    while f > 0.7:
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

def interpolation(data,fe):
    #Hp: highest fe is 1kHz
    from scipy import interpolate
    fe_interpolation = 1000

    if fe<fe_interpolation:
        t = len(data)/float(fe)
        vec_t = np.arange(0,t,1/float(fe))

        vec_t_int = np.arange(0,t,1/float(fe_interpolation))
        f = interpolate.interp1d(vec_t, data, kind='linear',bounds_error=False, fill_value='extrapolate')
        data_interpolated = f(vec_t_int)
    else:
        data_interpolated = data
        
    return data_interpolated,fe_interpolation


def core():
    count = 0


    import glob
    import ntpath

    data = pd.read_excel(r'C:\Users\zanel\Desktop\Axelife\Projects\rep_not_on_git\XLSX_CSV\BD_Ref_100_samples_filt_20210306_shuffled.xlsx')
    print('check',data.isnull().sum().sum())
    classes = data['Class']
    print('Entire dataset',data)
    data = data.drop(['Class'], axis = 1)
    d = data.iloc[:,1::]
    results = pd.DataFrame()
    norm_results = pd.DataFrame()


    for index, row in d.iterrows():
        count += 1
        print(count)
        sig = d.loc[[index]].values.flatten()
        fs = 100
        # high = 10
        # order = 3
        # filtered = butter_lowpass_filter(sig,high,fs,order)
        # smoothed = signal.savgol_filter(filtered,21,2)
        results = results.append(feature_extraction(sig,fs,str(count)))
        #import pdb; pdb.set_trace()

    
    #come stracazzo metto classi nello stesso file?
    #norm_results = pd.concat([norm_data,c.T], axis = 1)
    # come faccio a dare alla svm ? devo decidere in che maniera salvare i miei novi autovalori all'interno del DataFrame
    #Saving all the extracted features on an excel file.
    results.to_excel("features_100samples.xlsx")  
    print(results)
    # #drop useless features or features that cannot be standardised
    # df = results.drop(['count','Signal','min','max','Class'], axis = 1)

    # from sklearn import preprocessing
    # #scaling the  train set to a zero mean and unit variance one.
    # x = df.values #returns a numpy array
    # scaler = preprocessing.StandardScaler().fit(x)
    # x_scaled = scaler.transform(x)
    # tmp = pd.DataFrame(x_scaled)
    # tmp.columns = df.columns.values
    # #standardise dataset PLUS target variable CLASSES
    # classes = results['Class']
    # f=pd.concat([classes, tmp.set_index(classes.index)], axis=1)
    # f.to_excel('norm_features.xlsx')
    # print(f)

    # #saving the scaler in order to use it to normalise the test set
    # import pickle
    # pickle.dump(scaler, open(r'C:\Users\Serena\Documents\Axelife\Projects\Model\scaler.pkl','wb'))

    # # This part will be useful in case we want to train and test models on a fixed dataset. 
    # # We chose to use cross validation approach so this will be commented

    """
    f.to_excel("output_norm.xlsx")  
    good_signal = f[f['Class']==1]
    bad_signal = f[f['Class']==0]


    good_signal.to_excel("good_signals.xlsx")  
    bad_signal.to_excel("bad_signals.xlsx")  
    from sklearn.model_selection import train_test_split
    g_train, g_test = train_test_split(good_signal, test_size=0.3)
    b_train, b_test = train_test_split(bad_signal, test_size=0.3)
    frames_train = [g_train,b_train]
    train = pd.concat(frames_train)
    frames_test = [g_test,b_test]
    test = pd.concat(frames_test)
    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    output_train = train['Class']
    input_train = train.drop(['Class'], axis = 1)
    output_test = test['Class']
    input_test = test.drop(['Class'], axis = 1)
    """

#core()