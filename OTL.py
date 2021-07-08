SHIFT = 3
import ot
import ot.plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sezanel_tools
from scipy.stats import wasserstein_distance
data = pd.read_excel(r"C:\Users\zanel\Desktop\Axelife\Projects\rep_not_on_git\XLSX_CSV\waves.xlsx")
data_bad = pd.read_excel(r"C:\Users\zanel\Desktop\Axelife\Projects\rep_not_on_git\XLSX_CSV\DB_OT.xlsx")
# for index, row in data.iterrows():
#     sig = data.loc[[index]].values.flatten()
#     #sig_filt = sezanel_tools.butter_bandpass_filter(sig,0.05,15,100)
#     plt.plot(sig[1::])
#     plt.title(str(index))
#     plt.show()


# bad1 = data_bad.iloc[1,1::].values.flatten()
# bad1_norm01,_=sezanel_tools.norm(bad1)
# bad1_sum = bad1_norm01/np.sum(bad1_norm01)

wave1 = data.iloc[1,1::].values.flatten()
wave2 = data.iloc[11,1::].values.flatten()
wave3 = data.iloc[13,1::].values.flatten()
wave4 = data.iloc[3,1::].values.flatten()

wave1_test = data.iloc[0,1::].values.flatten()

plt.plot(wave1_test)
plt.show()
#import pdb;pdb.set_trace()

wave1_norm01,_=sezanel_tools.norm(wave1)
wave2_norm01,_=sezanel_tools.norm(wave2)
wave3_norm01,_=sezanel_tools.norm(wave3)
wave4_norm01,_= sezanel_tools.norm(wave4)

wave1_test_norm01,_ = sezanel_tools.norm(wave1_test)

wave1_sum = wave1_norm01/np.sum(wave1_norm01)
wave2_sum = wave2_norm01/np.sum(wave2_norm01)
wave3_sum = wave3_norm01/np.sum(wave3_norm01)
wave4_sum = wave4_norm01/np.sum(wave4_norm01)

wave1_test_sum = wave1_test_norm01/np.sum(wave1_test_norm01)


#import pdb; pdb.set_trace()

plt.plot(wave1_sum)
# plt.plot(wave2_sum)
# plt.plot(wave3_sum)
plt.plot(wave4_sum)
plt.plot(wave1_test_sum)
plt.show()


n_bins = len(wave1_sum)
bins = np.linspace(0,1,n_bins).reshape(n_bins,1)
M = ot.dist(bins, bins, 'euclidean')
M /= M.max()



cc12 = np.correlate(wave1_sum.values.flatten(), wave2_sum.values.flatten())
cc13 = np.correlate(wave1_sum.values.flatten(), wave3_sum.values.flatten())
dist12 = ot.emd2(wave1_sum.values.flatten(),wave2_sum.values.flatten(), M)
dist13 = ot.emd2(wave1_sum.values.flatten(),wave3_sum.values.flatten(), M)
dist14 = ot.emd2(wave1_sum.values.flatten(),wave4_sum.values.flatten(), M)
dist11_test = ot.emd2(wave1_sum.values.flatten(),wave1_test_sum.values.flatten(),M)

w12 = wasserstein_distance(wave1_sum.values.flatten(),wave2_sum.values.flatten())
w13 = wasserstein_distance(wave1_sum.values.flatten(),wave3_sum.values.flatten())
w14 = wasserstein_distance(wave1_sum.values.flatten(),wave4_sum.values.flatten())
w11_test = wasserstein_distance(wave1_sum.values.flatten(),wave1_test_sum.values.flatten())
#import pdb; pdb.set_trace()

G0 = ot.emd(wave1_sum.values.flatten(),wave4_sum.values.flatten(),M)
ot.plot.plot1D_mat(wave1_sum.values.flatten(), wave4_sum.values.flatten(), G0, 'OT matrix G0 1-4')
plt.show()

G0 = ot.emd(wave1_sum.values.flatten(),wave1_test_sum.values.flatten(),M)
ot.plot.plot1D_mat(wave1_sum.values.flatten(), wave1_test_sum.values.flatten(), G0, 'OT matrix G0 1-1_test')
plt.show()


G0 = ot.emd(wave1_sum.values.flatten(),wave2_sum.values.flatten(),M)
ot.plot.plot1D_mat(wave1_sum.values.flatten(), wave2_sum.values.flatten(), G0, 'OT matrix G0 1-2')
plt.show()

G0 = ot.emd(wave1_sum.values.flatten(),wave3_sum.values.flatten(),M)
ot.plot.plot1D_mat(wave1_sum.values.flatten(), wave3_sum.values.flatten(), G0, 'OT matrix G0 1-3')
plt.show()
# dist12_e = np.linalg.norm(wave1_sum.values.flatten() - wave2_sum.values.flatten())
# distbad11= ot.emd2(bad1_sum.values.flatten(),wave1_sum.values.flatten(), M)
# dist13 = ot.emd2(wave1_sum.values.flatten(),wave3_sum.values.flatten(), M)
# dist13_e = np.linalg.norm(wave1_sum.values.flatten() - wave3_sum.values.flatten())
# dist23 = ot.emd2(wave2_sum.values.flatten(),wave3_sum.values.flatten(), M)


print('dst12 = {},dist13 = {}, dist14 = {}, dist11_test={}'.format(dist12,dist13, dist14,dist11_test))
print('w12 = {},w13 = {}, w14 = {}, w11_test = {}'.format(w12,w13, w14,w11_test))

# if SHIFT:
#     dist_v = np.inf*np.ones((n_bins,1))
#     for i in range(n_bins):     
#         dist_v[i] = ot.emd2(class1_0,class1_1, M)
#         class1_0 = np.roll(class1_0,1)

#     plt.figure()
#     plt.plot(dist_v)
