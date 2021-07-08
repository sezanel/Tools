#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import seaborn as sns


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import filtfilt,butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y

test = pd.read_excel(r'C:\Users\zanel\Desktop\Axelife\DATA\Rest-Avignon\Signals\ALLAIC.xlsx',index_col=None)

print(test)
first_ECG = test['ECG_I']
second_ECG = test['ECG_II']
hand_pop = test['hand_POP']
feet_pop = test['feet_POP']
e4 = test['BVP']

time_pop = test['hand_POP'].values[0]
time_ECG = test['ECG_I'].values[0]
time_e4 = test['BVP'].values[0]
###import ecg_plot
low = 0.67
high =49
fs = 100
order = 3
###change filter popmetre

second = second_ECG[3::]
second_zeromean = second-np.mean(second)
second_filtered = butter_bandpass_filter(second_zeromean.dropna(),low,high,500,order)
secs = len(second_filtered)//500 # Number of seconds in signal X
samps = secs*100    # Number of samples to downsample
second_100Hz = scipy.signal.resample(second_filtered, samps)

first = first_ECG[3::]
first_zeromean = first-np.mean(first)
first_filtered = butter_bandpass_filter(first_zeromean.dropna(),low,high,500,order)
secs = len(first_filtered)//500 # Number of seconds in signal X
samps = secs*100     # Number of samples to downsample
first_100Hz = scipy.signal.resample(first_filtered, samps)

hand = hand_pop[3::]
hand_filtered = butter_bandpass_filter(hand.dropna(),0.05,15,1000,3)
secs = len(hand_filtered)//1000 # Number of seconds in signal X
samps = secs*100     # Number of samples to downsample
hand_100Hz = scipy.signal.resample(hand_filtered, samps)

feet = feet_pop[3::]
feet_filtered = butter_bandpass_filter(feet.dropna(),0.05,15,1000,3)
secs = len(feet.dropna())//1000 # Number of seconds in signal X
samps = secs*100     # Number of samples to downsample
feet_100Hz = scipy.signal.resample(feet_filtered, samps)

e4_ = e4[3::]
secs = len(e4.dropna())//64 # Number of seconds in signal X
samps = secs*100     # Number of samples to downsample
e4_100Hz = scipy.signal.resample(e4_.dropna(), samps)

gain_pop = abs(np.max(second_100Hz))/abs(np.max(hand_100Hz))
gain_e4 = abs(np.max(second_100Hz))/abs(np.max(e4_100Hz))

third_100Hz = -first_100Hz+second_100Hz

x_ECG = np.linspace(time_ECG/1000,time_ECG/1000+len(first_100Hz)/100,len(first_100Hz))
x_POP = np.linspace(time_pop/1000,time_pop/1000+len(hand_100Hz)/100,len(hand_100Hz))
x_e4 = np.linspace(time_e4/1000,time_e4/1000+len(e4_100Hz)/100,len(e4_100Hz))

time_diff = (time_pop-time_ECG)/1000
sample_diff = time_diff*fs

ECG_sync = second_filtered[int(sample_diff):int(sample_diff)+len(hand_filtered)]

import ecg_plot
import pdb; pdb.set_trace()

#plt.figure(figsize= (12, 8))
#diff = []
#f, = plt.plot(x_ECG,first_100Hz+700, c='blue', label= 'I')
#s, = plt.plot(x_ECG,second_100Hz, label='II')
#t, = plt.plot(x_ECG,third_100Hz-600, label = 'III')
#pf, = plt.plot(x_e4,e4_100Hz*gain_e4, c='green', label = 'Empatica')
#p, = plt.plot(x_POP,hand_100Hz*gain_pop, c='purple', label = 'pop')
#plt.legend(handles=[f,p,pf])
#plt.savefig('eldamA.png')
#plt.show()


# # In[9]:


# plt.figure(figsize= (20, 15))


# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(x_ECG,first_100Hz+700,c='blue', label= 'I')
# plt.title('ECG_I')


# ax[1].plot(x_e4,e4_100Hz*gain_e4,c='green', label = 'Empatica')
# plt.title('Empatica')


# ax[2].plot(x_POP,hand_100Hz*gain_pop,c='purple', label = 'pop')
# plt.title('pop')


# #plt.legend()
# plt.show()


# In[10]:


plt.figure(figsize= (12, 8))
diff = []
f, = plt.plot(x_ECG,first_100Hz+700, c='blue', label= 'I')
pf, = plt.plot(x_e4,e4_100Hz*gain_e4, c='green', label = 'Empatica')
p, = plt.plot(x_POP,hand_100Hz*gain_pop, c='purple', label = 'pop')
plt.legend(handles=[f,p,pf])
plt.show()


# In[ ]:





# In[ ]:




