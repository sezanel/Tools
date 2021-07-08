def interpolation(data,fs,fe_interpolation):
    import numpy as np
    from scipy import interpolate
    t = len(data)*fe_interpolation/fs
    #print(t)
    vec_t = np.linspace(0,len(data)/fs,len(data))
    vec_t_Interpolation = np.linspace(0,len(data)/fs,round(t))
    f = interpolate.interp1d(vec_t, data, kind='linear')
    data_interpolated = f(vec_t_Interpolation)
    return data_interpolated

def norm(data):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    norm = pd.DataFrame(data)
    scaler = MinMaxScaler()
    norm_sig01 = pd.DataFrame(scaler.fit_transform(norm))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    norm_sig11 = pd.DataFrame(scaler.fit_transform(norm))
    #plt.plot(norm_sig01)
    #plt.plot(norm_sig11)
    return norm_sig01,norm_sig11

def derivative(signal):
    y=[]
    for i in range(len(signal)-1):
        y.insert(i,(signal[i+1]-signal[i]))
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import filtfilt, butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, lowcut, fs, order=5):
    from scipy.signal import filtfilt, butter
    #import pdb; pdb.set_trace()
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, [low], btype='low', analog=False)
    # print('b',b)
    # print('a',a)
    y = filtfilt(b, a, data)
    return y

def load_dataset(path, test, val):
    import numpy as np
    from sklearn.utils import shuffle
    import pandas as pd
    data = pd.read_excel(path)
    # data = shuffle(d)
    print('check',data.isnull().sum().sum())
    # data.to_excel(r'C:\Users\zanel\Desktop\Axelife\Projects\rep_not_on_git\XLSX_CSV\PPG_diabetes_helathy_100samples_shuffled.xlsx')
    classes = data['Class']
    # print(vvv)
    #data = data.drop(['Class'], axis = 1)
    
    data = data.drop(['Class','Signal'], axis = 1)
    print(data)
    #con il nuovo dataset non è necessario togliere la prima riga
    d = data.iloc[:,1::]

    from sklearn.model_selection import train_test_split
    train_input, test_input, train_output, test_output = train_test_split(d, classes, test_size=test, stratify = classes)
    train_input, val_input, train_output, val_output = train_test_split(train_input,train_output, test_size=val, stratify = train_output)
    # test_input.to_excel('test_input_20210205_filt.xlsx')
    # test_output.to_excel('test_output_20210205_filt.xlsx')
    # val_input.to_excel('val_input_finetune_v2.xlsx')
    # val_output.to_excel('val_output_finetune_v2.xlsx')
    print('TEST OUTPUT SHAPE',test_output.shape)
    print('TEST OUTPUT SHAPE',test_input.shape)

    # train_input.to_excel('train_input.xlsx')
    # train_output.to_excel('train_output.xlsx')
    # test_input_df = pd.DataFrame(test_input)
    # test_input_df.to_excel('test_input.xlsx')
    # test_output_df = pd.DataFrame(test_output)
    # test_output_df.to_excel('test_output.xlsx')
    # val_input.to_excel('xval.xlsx')
    # val_output.to_excel('yval.xlsx')
    d_expanded = np.expand_dims(d, axis = 2)
    classes_expanded = np.expand_dims(classes, axis = 1)
    train_input_dim = np.expand_dims(train_input, axis=2)
    train_output_dim = np.expand_dims(train_output, axis = 1)
    test_input_dim = np.expand_dims(test_input, axis=2)
    test_output_dim = np.expand_dims(test_output, axis = 1)
    val_input_dim = np.expand_dims(val_input, axis=2)

    # print('val_input shape',val_input_dim.shape)
    # print('TEST OUTPUT SHAPE AUGMENTED',test_output_dim.shape)
    # print('TEST OUTPUT SHAPE AUGMENTED',test_input_dim.shape)
    val_output_dim = np.expand_dims(val_output, axis = 1)
    # one hot encode y 
    from tensorflow.keras.utils import to_categorical
    #from utils import to_categorical
    #from keras.utils import to_categorical
    classes_expanded = to_categorical(classes_expanded)
    train_output_dim = to_categorical(train_output_dim)
    test_output_dim = to_categorical(test_output_dim)
    val_output_dim = to_categorical(val_output_dim)

    n_timesteps = test_input_dim.shape[1]
    n_features = test_input_dim.shape[2]
    n_outputs = test_output_dim.shape[1]


    return train_input_dim, train_output_dim, test_input_dim, test_output_dim ,val_input_dim, val_output_dim, d_expanded,classes_expanded

def peak_valid(data):
    from scipy.signal import find_peaks
    import numpy as np
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

