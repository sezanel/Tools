import scipy.io
import csv
mat = scipy.io.loadmat(r'C:\Users\zanel\Downloads\Data_05142020\Data\4001.mat')
print(mat)
#mat.to_excel('ppg.xlsx')

with open('mycsvfile.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, mat.keys())
    w.writeheader()
    w.writerow(mat)