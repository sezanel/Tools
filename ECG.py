
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import filtfilt,butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y

def plot():
    import glob
    import ntpath

    path = r'C:\Users\zanel\Desktop\Axelife\DATA\Recherche_Serena\CSV\XLSX\Processed' # use your path
    all_files = glob.glob(path + "/*.xlsx")

    for filename in all_files:
        head, tail = ntpath.split(filename)
        test = pd.read_excel(filename)
        print(tail)
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



        diff = []
        s, = plt.plot(x_ECG,second_100Hz+max(hand_100Hz*gain_pop), label='II')
        p, = plt.plot(x_POP,hand_100Hz*gain_pop, label = 'pop')
        pf, = plt.plot(x_e4,e4_100Hz*gain_e4-max(hand_100Hz*gain_pop), label = 'Empatica')
        plt.legend(handles=[s,p,pf])
        plt.title(tail)
        plt.show()
        #import pdb; pdb.set_trace()

def upload_empatica():
    import glob
    import ntpath
    path = r'C:\Users\zanel\Desktop\Axelife\DATA\Recherche_Serena\CSV\XLSX' # use your path
    all_files = glob.glob(path + "/*.xlsx")
    pathe4 = r'C:\Users\zanel\Desktop\Axelife\DATA\Recherche_Serena\CSV\XLSX\Empatica' # use your path
    all_filese4 = glob.glob(pathe4 + "/*.zip")
    mindiff = 10000000000000
    for filename1 in all_files:
        pop = pd.read_excel(filename1)
        head, tail = ntpath.split(filename1)
        print(tail)
        timepop = pop.iloc[0][3]
        for filename in all_filese4:
            
            #import pdb;pdb.set_trace()
            import zipfile
            archive = zipfile.ZipFile(filename, 'r')
            df = pd.read_csv(archive.open('BVP.csv'))
            timee4 = float(df.columns[0])*1000
            if abs(timepop-timee4)<mindiff:
                mindiff = abs(timepop-timee4)
                name = filename
                time = timee4
        print('Compatibale file:', name)

        from datetime import datetime
        dt_object = datetime.fromtimestamp(time/1000)
        date = dt_object.strftime("%D")
        hour = dt_object.strftime("%H")
        min = dt_object.strftime("%M")
        sec = dt_object.strftime("%S")
        print('Empatica Details: ', date, 'hour', hour,'-',min,'-',sec)
        str1 = 'Empatica Details: '+ date+ '  hour '+hour+'-'+min+'-'+sec
        dt_object = datetime.fromtimestamp(timepop/1000)
        date = dt_object.strftime("%D")
        hour = dt_object.strftime("%H")
        min = dt_object.strftime("%M")
        sec = dt_object.strftime("%S")
        print('pOpmétre Details: ', date, 'hour', hour,'-',min,'-',sec)
        str2 = 'pOpmétre Details: '+ date+ '  hour '+hour+'-'+min+'-'+sec



        confirm = input('Please,confirm and save the matching files:   (y/n)    ')
        while confirm!='y' and confirm!='n':
            confirm = input('Please,confirm and save the matching files:   (y/n)    ')

        mindiff = 10000000000000
        nbr = input('Insert patient number:   (y/n)    ')

        if confirm == 'y':      
            archive = zipfile.ZipFile(name, 'r')
            signals = ['BVP','EDA','TEMP']
            for signal in signals:
                df = pd.read_csv(archive.open(signal+'.csv'))
                df.columns = [signal]
                df.loc[0,signal] =time
                pop[signal]=df
            df = pd.read_csv(archive.open('ACC.csv'))
            df.columns = ['x','y','z']
            df.loc[0,'x'] =time
            df.loc[0,'y'] =time
            df.loc[0,'z'] =time
            pop['x']=df['x']
            pop['y']=df['y']
            pop['z']=df['z']
            pop.drop( columns=['Unnamed: 0'])  
            path_to_save = 'C:\\Users\\zanel\\Desktop\\Axelife\DATA\\Recherche_Serena\\CSV\\XLSX\\Processed\\'+str(nbr)+'.xlsx'
            pop.to_excel(path_to_save)
           
      
    #import pdb;pdb.set_trace()

def print_ECG():
    import glob
    import ntpath
    import ecg_plot
    import pdb;pdb.set_trace()
    amp = pd.read_excel('amp_ECG2.xlsx')
    low = 0.67
    high =49
    
    order = 3
    amp_filt2 = butter_bandpass_filter(amp['II'].values.flatten(),low,high,500,order)
    ecg_plot.plot_1(amp_filt2[0:5000]/360.76)
    plt.show()
    


    path = r'C:\Users\zanel\Desktop\Axelife\DATA\Rest-Avignon\Signals\Verified' # use your path
    all_files = glob.glob(path + "/*.xlsx")

    for filename in all_files:
        head, tail = ntpath.split(filename)
        test = pd.read_excel(filename)
        
        first_ECG = test['ECG_I']
        second_ECG = test['ECG_II']

        time_ECG = test['ECG_I'].values[0]

        ###import ecg_plot
        low = 0.67
        high =49
        
        order = 3
        ###change filter popmetre

        second = second_ECG[3::]
        second_zeromean = second-np.mean(second)
        second_filtered = butter_bandpass_filter(second_zeromean.dropna(),low,high,500,order)
       
        
        first = first_ECG[3::]
        first_zeromean = first-np.mean(first)
        first_filtered = butter_bandpass_filter(first_zeromean.dropna(),low,high,500,order)
       
        third = -first_filtered+second_filtered

        #x_ECG = np.linspace(time_ECG/1000,time_ECG/1000+len(first_100Hz)/100,len(first_100Hz))
      
        plt.plot(second_filtered)
        plt.show()
        import ecg_plot
        
        #r = np.expand_dims(second_filtered, axis=0)
        import pdb; pdb.set_trace()

        
        # s, = plt.plot(x_ECG,second_100Hz+max(hand_100Hz*gain_pop), label='II')
        # p, = plt.plot(x_POP,hand_100Hz*gain_pop, label = 'pop')
        # pf, = plt.plot(x_e4,e4_100Hz*gain_e4-max(hand_100Hz*gain_pop), label = 'Empatica')
        # plt.legend(handles=[s,p,pf])
        # plt.title(tail)
        # plt.show()

#upload_empatica()
plot()
#print_ECG()
