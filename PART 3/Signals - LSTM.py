#!/usr/bin/env python
# coding: utf-8

# In[283]:


import numpy as np


# In[284]:


import tensorflow


# In[285]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import matplotlib.pyplot as plt
import pandas


# In[286]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[287]:


import numpy as np
import random
import scipy
import csv
def loadCsv(filename):
    
    
    
    
    lines = csv.reader(open(filename, "rb"))
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        dataset = list(reader)
        
        return dataset

s = []
s1=loadCsv('s1.csv') 
[s.append(float(s1[x][0])) for x in range(len(s1)) ]

s2=loadCsv('s2.csv')
[s.append(float(s2[x][0])) for x in range(len(s2)) ]

s3=loadCsv('s3.csv')
[s.append(float(s3[x][0])) for x in range(len(s3)) ]
s4=loadCsv('s4.csv')
[s.append(float(s4[x][0])) for x in range(len(s4)) ]
s5=loadCsv('s5.csv')
[s.append(float(s5[x][0])) for x in range(len(s5)) ]
s6=loadCsv('s6.csv')
[s.append(float(s6[x][0])) for x in range(len(s6)) ]
s7=loadCsv('s7.csv')
[s.append(float(s7[x][0])) for x in range(len(s7)) ]
s8=loadCsv('s8.csv')
[s.append(float(s8[x][0])) for x in range(len(s8)) ]
s9=loadCsv('s9.csv')
[s.append(float(s9[x][0])) for x in range(len(s9)) ]
s10=loadCsv('s10.csv')
[s.append(float(s10[x][0])) for x in range(len(s10)) ]
s11=loadCsv('s11.csv')
[s.append(float(s11[x][0])) for x in range(len(s11)) ]
s12=loadCsv('s12.csv')
[s.append(float(s12[x][0])) for x in range(len(s12)) ]
s13=loadCsv('s13.csv')
[s.append(float(s13[x][0])) for x in range(len(s13)) ]
s14=loadCsv('s14.csv')
[s.append(float(s14[x][0])) for x in range(len(s14)) ]
s15=loadCsv('s15.csv')
[s.append(float(s15[x][0])) for x in range(len(s15)) ]
s16=loadCsv('s16.csv')
[s.append(float(s16[x][0])) for x in range(len(s16)) ]
s17=loadCsv('s17.csv')
[s.append(float(s17[x][0])) for x in range(len(s17)) ]
s18=loadCsv('s18.csv')
[s.append(float(s18[x][0])) for x in range(len(s18)) ]
s19=loadCsv('s19.csv')
[s.append(float(s19[x][0])) for x in range(len(s19)) ]
s20=loadCsv('s20.csv')
[s.append(float(s20[x][0])) for x in range(len(s20)) ]
s21=loadCsv('s21.csv')
[s.append(float(s21[x][0])) for x in range(len(s21)) ]
s22=loadCsv('s22.csv')
[s.append(float(s22[x][0])) for x in range(len(s22)) ]
s23=loadCsv('s23.csv')
[s.append(float(s23[x][0])) for x in range(len(s23)) ]
s24=loadCsv('s24.csv')
[s.append(float(s24[x][0])) for x in range(len(s24)) ]
s25=loadCsv('s25.csv')
[s.append(float(s25[x][0])) for x in range(len(s25)) ]
s26=loadCsv('s26.csv')
[s.append(float(s26[x][0])) for x in range(len(s26)) ]
s27=loadCsv('s27.csv')
[s.append(float(s27[x][0])) for x in range(len(s27)) ]
s28=loadCsv('s28.csv')
[s.append(float(s28[x][0])) for x in range(len(s28)) ]
s29=loadCsv('s29.csv')
[s.append(float(s29[x][0])) for x in range(len(s29)) ]
s30=loadCsv('s30.csv')
[s.append(float(s30[x][0])) for x in range(len(s30)) ]

points=loadCsv('GT.csv')


X = s
z0 = (np.asarray(points))[1:21]

Y = []
f=0
for k in [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27,s28,s29,s30]:
    for i in range(len(k)):
        c= len(Y)
        for j in range(0,20):
            if int(z0[j][f])==int(i):
                
                Y.append(0.99)
                break
        c1 = len(Y)
        if c != c1:
            continue
        Y.append(0)
    f = f+1
    


# In[288]:



s = np.asarray(s)
np.reshape(s,(178304,1))

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# In[289]:


Y1 =s
Y1 = np.asarray(Y1)

Y1= Y1.T
x = create_dataset(Y1,5)
x = x[0]

Y = Y[6:len(Y)]
len(Y)


# In[290]:


from keras.layers import Dropout
x = np.reshape(x, (len(x), 1, 5))
Y = np.asarray(Y)
model = Sequential()
model.add(LSTM(4, input_shape=(1, 5)))
model.add(Dropout(0.8))
model.add(Dense(1))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

model.fit(x[5000:170000], Y[5000:170000], epochs=7, batch_size=1, verbose=2)


# In[ ]:


trainPredict = model.predict(x[0:len(s1)])

np.shape(x)


# In[ ]:



import matplotlib.pyplot as plt
plt.plot(trainPredict)
plt.show()
x_coordinate = [ i for i in range(len(s1)) ]
z1 = [ float(s1[j][0]) for j in range(len(s1)) ]
plt.plot(x_coordinate,z1,'-gD',  marker='o',
     markerfacecolor='blue', markersize=1)

plt.show()


# In[ ]:


from scipy.signal import find_peaks
np.shape(trainPredict)


# In[ ]:


peaks, value = find_peaks(trainPredict[:,0], height=0)


# In[ ]:



peak2 = []
peak4 = []
for i in range(0,len(peaks)):
    
    peak2.append(trainPredict[i])
 
    if float(trainPredict[i][0])>0.0:
        peak4.append(i)
    
p = np.mean(peak2)

peak2 = np.asarray(peak2)
peak3, _ = find_peaks(peak2[:,0], height=0)
#print(peak3)

#for i in range(0,len(peak3)):
    
 #   print(peaks[peak3[i]],trainPredict[peaks[peak3[i]]])
    


# In[ ]:


peaks1 = []
peaks2 = []
peaks1.append([])
j=0
for i in range(1,len(peaks)-1):
    if peaks[i+1]<peaks[i-1]+50 or peaks[i-1]<peaks[i+1]-50 :
        peaks1[j].append(peaks[i])
        peaks2.append(peaks[i])
            
        continue
    else:
        if j!=0 and peaks[i]>500:
            peaks2.append(peaks1[j][int(len(peaks1[j])/2)])
            
        j=j+1
        peaks1.append([])
        peaks1[j].append(peaks[i])
        peaks2.append(peaks[i])
        
    

markers_on = peaks2

x_coordinate = [ i for i in range(len(s1)) ]
z1 = [ float(s1[j][0]) for j in range(len(s1)) ]
plt.plot(x_coordinate,z1,'-gD', markevery=markers_on,marker='o',
    markerfacecolor='red', markersize=4)
plt.show()



# In[ ]:


plt.plot(peak2)


# In[ ]:





# In[ ]:


peak4 = []

for i in range(0,len(peaks2)-6):
    peak4.append([0,0])
    peak4[i][0] = peaks2[i]
    peak4[i][1] = float(trainPredict[peaks2[i]][0])

def sortSecond(val): 
    return val[1]  
def sortFirst(val): 
    return val[0]  

peak4.sort(key=sortSecond)
peak5 = peak4[len(peak4)-40:len(peak4)]
peak5.sort(key = sortFirst)


peak6 = []
for i in range(0,len(peak5)):
    peak6.append(peak5[i][0])

markers_on1 = [518, 718, 775, 851, 929, 972, 1045, 1248, 1412, 1740, 1981, 2370, 2443, 2507, 2573, 2577, 2661, 2848, 2909, 3249, 3418, 3597, 3669, 3734, 3925, 4049, 4069, 4125, 4266, 4272, 4290, 4538, 4614, 4680, 4969, 5041, 5116, 5121, 5224, 5283]
markers_on1 = np.asarray(markers_on1)

plt.plot(x_coordinate,z1,marker='o',markerfacecolor='red', markersize=4)
plt.show()


# In[ ]:





# In[ ]:




