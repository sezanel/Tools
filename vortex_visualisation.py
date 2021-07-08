
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def interpolation(data,fs,fe_interpolation):

    from scipy import interpolate

    t = len(data)/float(fs)
    vec_t = np.arange(0,t,1/float(fs))

    vec_t_int = np.arange(0,t,1/float(fe_interpolation))
    f = interpolate.interp1d(vec_t, data, kind='linear',bounds_error=False, fill_value='extrapolate')
    data_interpolated = f(vec_t_int)

    return data_interpolated,fe_interpolation



# df = pd.read_excel(r'C:\Users\zanel\Desktop\Axelife\DATA\pOpmetre\BD_Ref\class.xlsx', index_col=None, header=0)
# print(df)
# name = df['Name']
# print(name[0])

# for index, row in df.iterrows():
#     print(index)
#     print(name[index])



import glob
path = r'C:\Users\zanel\Desktop\Axelife\DATA\DOWNLOADED\Vortex\Vortex_100Hz' # use your path
all_files = glob.glob(path + "/*.csv")

#data_vortex = []
#data_vortex = pd.DataFrame()
from scipy import signal

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    plt.plot(df.values.flatten())
plt.show()


# import glob

# # # # # # # path = r'C:\Users\Serena\Documents\Axelife\DATA\Vortex\Vortex_100Hz' # use your path
# # # # # # # all_files = glob.glob(path + "/*.csv")

# # # # # # # #data_vortex = []
# # # # # # # data_vortex = pd.DataFrame()
# # # # # # # from scipy import signal

# # # # # # # for filename in all_files:
# # # # # # #     df = pd.read_csv(filename, index_col=None, header=0)
# # # # # # #     # print('check',df.isnull().sum().sum())
# # # # # # #     df = df.fillna(0)
# # # # # # #     # print('check',df.isnull().sum().sum())
# # # # # # #     sig = df.T
# # # # # # #     resempled = signal.resample(sig.values.flatten(),100)
# # # # # # #     print(filename)
# # # # # # #     #sig0 = [0 if x==1 else x for x in sig]
# # # # # # #     _,norm11 = norm(resempled)
# # # # # # #     # print(norm11)
# # # # # # #     # plt.plot(norm11)
# # # # # # #     # plt.show()
# # # # # # #     data_vortex = data_vortex.append(norm11.T)
# # # # # # #     # print(data_vortex)
# # # # # # #     #print(data_vortex)
# # # # # # #     # plt.plot(norm11)
    
# # # # # # #     # plt.show()

# # # # # # # df1 = pd.DataFrame(data_vortex)
# # # # # # # print(df1)
# # # # # # # df1.to_excel('data_vortex_100Hz_norm.xlsx')

    


#bad_data = pd.DataFrame()

# # # # # # def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
# # # # # #     from scipy.signal import filtfilt, butter
# # # # # #     nyq = 0.5 * fs
# # # # # #     low = lowcut / nyq
# # # # # #     high = highcut / nyq
# # # # # #     b, a = butter(order, [low, high], btype='band', analog=False)
# # # # # #     y = filtfilt(b, a, data)
# # # # # #     return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import filtfilt, butter
    nyq = 0.5 *fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [high], btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
import ntpath
import glob 

# path = r'C:\Users\zanel\Desktop\Axelife\DATA\pOpmetre\Patricia\26_05_2020\SZ_26_05_2020\CLASSIFIED' # use your path
# print('1       ',path)
# all_files = glob.glob(path + "/*.csv")
# tmp = 0
# for filename in all_files:
    
#     head, tail = ntpath.split(filename)
#     #print(tail)

#     if tail[0] == 'M':
#         df = pd.read_csv(filename, index_col=None, header=0)
#         sig = df.values.flatten()
#         low = 0.05
#         high = 15
#         order = 3
#         filtered_low = butter_bandpass_filter(sig,low,high,1000,order)

#         #print('len filtered',len(filtered_low))
#         sig_int,_ = interpolation(filtered_low,1000,100)

#         for i in range(0,len(sig_int)-100):
#             norm01,norm11 = norm(sig_int[i:i+100])
#             bad_data=bad_data.append((norm01.T))
# print(bad_data)
# path = r'C:\Users\zanel\Desktop\Axelife\DATA\pOpmetre\Patricia\26_05_2020_1\SZ_26_05_2020_1\CLASSIFIED' # use your path
# print('2   ',path)

# all_files = glob.glob(path + "/*.csv")
# for filename in all_files:
#     head, tail = ntpath.split(filename)
#     #print(tail)

#     if tail[0] == 'M':
#         df = pd.read_csv(filename, index_col=None, header=0)
#         sig = df.values.flatten()
#         low = 0.05
#         high = 15
#         order = 3
#         filtered_low = butter_bandpass_filter(sig,low,high,1000,order)
#         sig_int,_ = interpolation(filtered_low,1000,100)
#         for i in range(0,len(sig_int)-100):
#             norm01,norm11 = norm(sig_int[i:i+100])
#             bad_data=bad_data.append((norm01.T))
# print(bad_data)

# path = r'C:\Users\zanel\Desktop\Axelife\DATA\pOpmetre\Patricia\28_05_2020_1\SZ_28_05_2020_1\CLASSIFIED' # use your path
# print('3    ',path)
# all_files = glob.glob(path + "/*.csv")
# for filename in all_files:
#     head, tail = ntpath.split(filename)

#     if tail[0] == 'M':
#         df = pd.read_csv(filename, index_col=None, header=0)
#         sig = df.values.flatten()
#         low = 0.05
#         high = 15
#         order = 3
#         filtered_low = butter_bandpass_filter(sig,low,high,1000,order)
#         sig_int,_ = interpolation(filtered_low,1000,100)
#         for i in range(0,len(sig_int)-100):
#             norm01,norm11 = norm(sig_int[i:i+100])
#             bad_data=bad_data.append((norm01.T))
# print(bad_data)

# # path = r'C:\Users\zanel\Desktop\Axelife\DATA\pOpmetre\Patricia\popmetre_original\classified\noisy' # use your path
# # all_files = glob.glob(path + "/*.csv")
# # for filename in all_files:
# #     head, tail = ntpath.split(filename)
# #     print(tail)

# #     if tail[0] == 'M':
# #         df = pd.read_csv(filename, index_col=None, header=0)
# #         sig = df.values.flatten()
# #         low = 0.05
# #         high = 15
# #         order = 3
# #         filtered_low = butter_bandpass_filter(sig,low,high,1000,order)
# #         sig_int,_ = interpolation(filtered_low,1000,100)
# #         for i in range(0,len(sig_int)-100):
# #             norm01,norm11 = norm(sig_int[i:i+100])
# #             bad_data=bad_data.append((norm01.T))
# # print(bad_data)

# path = r'C:\Users\zanel\Desktop\Axelife\DATA\pOpmetre\Patricia\28_05_2020_1\SZ_28_05_2020_1\CLASSIFIED' # use your path
# all_files = glob.glob(path + "/*.csv")
# print('4    ',path)

# for filename in all_files:
#     head, tail = ntpath.split(filename)
#     #print(tail)
#     if tail[0] == 'M':
#         df = pd.read_csv(filename, index_col=None, header=0)
#         sig = df.values.flatten()
#         low = 0.05
#         high = 15
#         order = 3
#         filtered_low = butter_bandpass_filter(sig,low,high,1000,order)
#         sig_int,_ = interpolation(filtered_low,1000,100)
#         for i in range(0,len(sig_int)-100):
#             norm01,norm11 = norm(sig_int[i:i+100])
#             bad_data=bad_data.append((norm01.T))
# print(bad_data)
# bad_data.to_excel('noise_pop_norm01_filt_v4.xlsx')

# plt.plot(sig_int)
# plt.show()
# # # # path = r'C:\Users\zanel\Desktop\Axelife\DATA\pOpmetre\BD_Ref\Signals\BD_Ref' # use your path
# # # # all_files = glob.glob(path + "/*.csv")
# # # # print(bad_data.shape[0])
# # # # info = []

# # # # for filename in all_files:
# # # #     try:
# # # #         if bad_data.shape[0]<400000:
# # # #             df = pd.read_csv(filename, index_col=None, header=0)
# # # #             df.columns=['hand','feet']
# # # #             low = 0.05
# # # #             high = 15
# # # #             order = 3
# # # #             sig = df['hand'].values.flatten()
# # # #             # plt.plot(sig)
# # # #             # plt.show()
# # # #             filtered_low = butter_bandpass_filter(sig,low,high,1000,order)
# # # #             # plt.plot(sig)
# # # #             # plt.show()
# # # #             # plt.plot(filtered_low)
# # # #             # plt.show()
# # # #             head, tail = ntpath.split(filename)
# # # #             print(tail)
# # # #             info.append(tail)

# # # #             #if tail[0] == 'M':
# # # #             sig_int,_ = interpolation(filtered_low,1000,100)
# # # #             # plt.plot(sig_int)
# # # #             # plt.show()
# # # #             # plt.plot(sig_int)
# # # #             # plt.show()
# # # #             for i in range(0,len(sig_int)-100,10):
# # # #                 # plt.plot(sig[i:i+100])
# # # #                 # # #print(sig)
# # # #                 # plt.title('intra_peaks' + tail + str(i))
# # # #                 # plt.xlabel('Samples')
# # # #                 # plt.ylabel('intra_peaks')
# # # #                 # plt.savefig('intra_peaks_' + tail +'_'+ str(i) + '.png')
# # # #                 norm01,norm11 = norm(sig_int[i:i+100])
# # # #                 # plt.plot(norm01)
# # # #                 # plt.show()
# # # #                 bad_data=bad_data.append((norm01.T))
# # # #             print('Columns :',bad_data.shape[0])
# # # #             #print(data_vortex)
# # # #             # plt.plot(norm11)

# # # #             # plt.show()
# # # #     except:
# # # #         print('Fail to extract')

# # # print(bad_data)
# # # # df = pd.DataFrame(info)
# # # # df.to_excel('info_waves_v3.xlsx')
# # # bad_data.to_excel('noise_pop_norm01_v4.xlsx')
# # import pandas as pd
# # data = pd.read_excel('false_pos_CNN_20210208.xlsx')
# # norm_wave = pd.DataFrame()
# # print(data)

# # for index, row in data.iterrows():
# #     print(index)
# #     print(row)
# #     sig = row
# #     print(sig)
# #     sig = sig.values.flatten()
# #     fig = plt.figure (dpi=100)
# #     plt.plot(sig)
# #     plt.savefig(str(index)+'.png')
# #     plt.close(fig)
# #     print(len(sig))



# # # df = pd.DataFrame(norm_wave)
# # # print('check',df.isnull().sum().sum())

# # # df.to_excel('Norm_Real_2_peaks_200samples.xlsx')



# # # norm_data_1 = pd.read_excel('norm_wave_real_period.xlsx')
# # # norm_data_0 = pd.read_excel('bad_signals_100Hz_norm.xlsx')
# # # norm_data = norm_data_1.append(norm_data_0)
# # # print(norm_data)
# from sklearn.utils import shuffle
# norm_data = pd.read_excel('XLSX_CSV\BD_Ref_100_samples_filt_20210306.xlsx')
# norm_data = shuffle(norm_data)
# print('check',norm_data.isnull().sum().sum())
# norm_data.to_excel('XLSX_CSV\BD_Ref_100_samples_filt_20210306_shuffled.xlsx')
# # # # #norm_data = norm_data.fillna(-1)
# # # norm_data = pd.read_excel('norm_data_period_1sec.xlsx')
# # # norm_data = shuffle(norm_data)

# # # print('check',norm_data.isnull().sum().sum())
# # # norm_data.to_excel('norm_data_period_1sec.xlsx')
# # # print(norm_data)
# # # 	# from sklearn.utils import shuffle
# # # 	# data = shuffle(norm_data)
# # # 	# print(data)
# # # 	# data.to_excel("shuffle_norm_data.xlsx")


# # # from sklearn.utils import shuffle


def butter_lowpass_filter(data, lowcut, fs, order=3):
    from scipy.signal import filtfilt,butter,correlate
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, [low], btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# from scipy.signal import argrelmin
# from scipy.signal import find_peaks
# from scipy.optimize import curve_fit
# import seaborn as sns
# norm_data = pd.read_excel(r'C:\Users\zanel\Desktop\Axelife\Projects\FN_CNN_optim.xlsx')
# # # from scipy.stats import pearsonr
# # # #norm_data.columns=['hand','feet']
# # # print(norm_data)

# for index, row in norm_data.iterrows():
#     sig = norm_data.loc[[index]].values.flatten()
#     fig = plt.figure (dpi=300)
#     sig_cut = sig[2::]
#     plt.plot(sig_cut)
#     plt.savefig(str(index)+'index.png')
#     plt.close(fig)

# sig = norm_data.loc[[index]].values.flatten()



# norm_data.columns = ['hand']
# sig = norm_data['hand'].values.flatten()
# #ppg = norm_data['feet'].values.flatten()
# #sig = sig-np.mean(sig)

# #sig_filt = butter_bandpass_filter(sig[8000:15000],0.05,15,1000)
# plt.plot(sig)
# plt.show()
# sig_filt = sig[5600:6276]

# x = np.linspace(0, len(sig_filt)/1000,len(sig_filt))
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
# #mpl.use('pgf')
# font = {'family':'serif'}
# plt.rc('font',**font)
# fig = plt.figure (dpi=300)
# ax = plt.subplot(111)
# ax.plot(x,sig_filt,'0.5')
# #ax.plot(np.asarray(valid_peak)/1000,sig_filt[valid_peak],"o")
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# fig.subplots_adjust(bottom=0.2)

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#legend = ax.legend(loc = 'upper right')

# plt.title('PPG signal with PVC')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.show()

# peak,_ = find_peaks(sig_filt)
# print(peak)
# plt.plot(peak,sig_filt[peak],'o')
# plt.axis('off')
# plt.show()
# valid_peak = []
# for i in range(len(peak)):
#     if sig_filt[peak[i]]>np.mean(sig):
#         valid_peak.append(peak[i])

# print(valid_peak)
# plt.plot(sig_filt)
# plt.plot(peak,sig_filt[peak],'o')
# plt.plot(valid_peak,sig_filt[valid_peak],'o')
# plt.axis('off')
# plt.show()
# point_min = argrelmin(sig_filt)
# print(sig_filt[point_min])
# sig_cut = sig_filt[::]
# x = np.linspace(0, len(sig_filt)/1000,len(sig_filt))
# plt.plot(x,sig01)
# plt.show()
# print(sig)
# import glob

# path = r'C:\Users\zanel\Desktop\Axelife\DATA\pOpmetre\BD_Ref\Signals\BD_Ref\temp_excluded from CNN DB' # use your path
# all_files = glob.glob(path + "/*.csv")

# #data_vortex = []
# from scipy import signal
# import sezanel_tools

# for filename in all_files:
#     df = pd.read_csv(filename, index_col=None, header=0)
#     print(filename)
#     #df = pd.read_csv(filename,sep='; ',engine='python')
#     df.columns=['hand','feet']
#     #sig = df['hand'].str.replace(',', '.')
#     print(df)
#     #sig = df['hand'].values.flatten()
#     high = 10
#     order = 3
#     fs = 1000
    
#     sig = df['hand'].values.flatten()
    
#     filt = butter_lowpass_filter(sig.astype(float), high, fs)
#     filt_cut = filt[6*fs:16*fs]
    
#     plt.plot(sig)
#     plt.show()
#     norm01,_ = sezanel_tools.norm(filt_cut)
#     print(norm01)
#     x = np.linspace(0,len(filt_cut)/fs,len(filt_cut))
#     plt.rcParams['text.usetex'] = True
#     plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
#     font = {'family':'serif'}
#     plt.rc('font',**font)
#     fig = plt.figure(figsize=(3, 2),dpi=300)
#     ax = plt.subplot(111)
#     ax.plot(x,norm01.values.flatten(),'0.5',linewidth=0.75)
#     # ax.spines['right'].set_visible(False)
#     # ax.spines['top'].set_visible(False)
#     fig.subplots_adjust(bottom=0.226,left = 0.19)
#     #plt.title('PPG signal in presence of diabetes')
#     plt.xlabel('Time [s]')
#     plt.ylabel('pOpmètre')
#     plt.show()




# for index, row in norm_data.iterrows():
#     sig = norm_data.loc[[index]].values.flatten()
#     sig_filt = butter_bandpass_filter(sig,0.05,15,100)
#     # plt.plot(sig_filt)
#     # plt.plot(sig)
#     # plt.show()
#     point_min = argrelmin(sig_filt)
#     print(sig_filt[point_min])
#     sig_cut = sig_filt[7::]
#     x = np.linspace(0, len(sig_cut)/100,len(sig_cut))
#     print(sig)
#     plt.rcParams['text.usetex'] = True
#     plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
#     # mpl.use('pgf')
#     font = {'family':'serif'}
#     plt.rc('font',**font)
#     fig = plt.figure (dpi=300)
#     plt.plot(x,sig_cut, "0.5")
#     plt.title('High quality PPG')
#     plt.xlabel('Time [s]')
#     plt.ylabel('Normalised amplitude')
#     plt.savefig(str(index)+'index.png')
#     plt.close(fig)



# # # ####  WORKING INTERPOLATION #####
# # # # from scipy import signal, interpolate
# # # # import numpy as np 
# # # # waves = pd.read_excel('interp_waves.xlsx')
# # # # print(waves)
# # # # wave_long = waves['long']
# # # # wave_short = waves['short']
# # # # nbr_samples = 100
# # # # wave_short = wave_short[0:75]
# # # # plt.plot(wave_short)
# # # # plt.plot(wave_long)
# # # # plt.show()

# # # # vec_t = np.linspace(0,len(wave_short)/100,len(wave_short))
# # # # print('vec_t',vec_t)
# # # # print('len signal', len(wave_short))
# # # # print('len vec_t',len(vec_t))
# # # # vec_t_Interpolation = np.linspace(0,len(wave_short)/100,100)
# # # # print('vec_t_inte',vec_t_Interpolation)
# # # # print('len vec_t_inte',len(vec_t_Interpolation))
# # # # f = interpolate.interp1d(vec_t, wave_short, kind='linear')
# # # # data_interpolation = f(vec_t_Interpolation)
# # # # #plt.plot(signal.resample(wave_short,nbr_samples))
# # # # x = np.linspace(0,100,100)
# # # # print(x)
# # # # plt.plot(x,data_interpolation)
# # # # print(len(data_interpolation))
# # # # print(wave_short)
# # # # print(data_interpolation[99])
# # # # #plt.plot(wave_short)
# # # # plt.show()