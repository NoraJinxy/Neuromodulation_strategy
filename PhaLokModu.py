# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:25:29 2022

@author: user
"""


import numpy as np
import torch
import torch.nn as nn
import random
import scipy.io as scio
from scipy import signal
import matplotlib.pyplot as plt
from torch.fft import fft
from scipy import signal
from torch.autograd import Variable
from scipy.signal import hilbert
torch.set_default_tensor_type(torch.DoubleTensor)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        # out[:, -1, :]-->(batch_size, hidden_size)
        # (batch_size, hidden_size)-->(batch_size, num_classes)
        out = self.fc(out[:, -1, :])
        return out

def Normalization(data):
    minVals = data.min(1)
    maxVals = data.max(1)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    for i in range(m):
        normData[i, :] = (data[i, :] - minVals[i]) / ranges[i]
    return normData

def Noise_Uniform(data):
    x=np.size(data,0)
    y=np.size(data,1)
    noise=np.zeros((x,y))
    
    for i in range(x):
        for j in range(y):
            noise[i,j]=random.random()
    
    out=torch.from_numpy(noise)
    return out
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def G_Input(data, rd, marker, Sample, pre_len):
    sub=np.size(data,0)
    data_len=np.size(data,1)
    
    data_x=torch.zeros((Sample,batch))
    data_rd=torch.zeros((Sample,batch+pre_len))
    data_m=torch.zeros((Sample,batch+pre_len))
    for i in range(Sample):  
        random_sub=random.randint(0,sub-1)
        train_begin=random.randint(0,int(data_len-batch-pre_len-1))
        
        data_x[i,:]=data[random_sub,train_begin:train_begin+batch].clone()
        data_rd[i,:]=rd[random_sub,train_begin:train_begin+batch+pre_len].clone()
        data_m[i,:]=marker[random_sub,train_begin:train_begin+batch+pre_len].clone()
    
    return data_x, data_rd, data_m

def Power(filtered_signal,num_fft):
    Y = fft(filtered_signal, num_fft)
    Y = torch.abs(Y)
    ps = Y**2 / num_fft
    ps = ps[0,:num_fft//2]
    sum_power = torch.sum(ps[5:15])
    
    return ps, sum_power


# --------------------------------- Load data ---------------------------------
Data_REST = scio.loadmat("Filter_letswave\REST_my_8-12Hz.mat")
data_rest = Data_REST["Data"]
marker_rest = Data_REST["Stimu_impulse"]

Fs=1000
fs = 500
step = int(Fs/fs)
run = np.size(data_rest, 0)
Data_len = np.size(data_rest, 1)
T = int(Data_len/Fs)
data_len = fs*T

# Downsample
data_downsample = signal.resample(data_rest, data_len, axis=1)
marker_downsample = np.zeros((run, data_len))

# for r in range(run):
#     m_vep = marker[r,:].reshape(-1, step)
#     marker_downsample[r,:] = np.sum(m_vep, 1)

# Normalize and change data type
data_scaled = Normalization(data_downsample)

data_rest_tensor = torch.from_numpy(data_scaled).float()
rd_rest_tensor = Noise_Uniform(data_scaled).float()
marker_rest_tensor = torch.from_numpy(marker_downsample).float()

# --------------------------------- Parameters --------------------------------
Dt=1/fs
T_rest=50
T_modu=50
T_power=int(T_rest/2)
Time=np.arange(0, T_modu, Dt).reshape(-1, 1)

Data_rest=T_rest*fs
Data_modu = T_modu*fs
Data_power = T_power*fs

batch = int(0.2*fs)
y_len = 1
Sample = 1
num_fft = fs


step=1
N_delay=batch
b,a=signal.butter(2,[2*8/fs,2*12/fs],'bandpass')
a=torch.from_numpy(a)
b=torch.from_numpy(b)
m_signal = 0.5
 

# ----------------------------- Hyper Parameters ------------------------------
INPUT_SIZE = 3
HIDDEN_SIZE = 16
NUM_LAYER = 1
OUTPUT_SIZE = y_len
DROPOUT = 0
LR = 1e-3


# ---------------------------------- Load Net ---------------------------------
G = RNN(INPUT_SIZE, 
         HIDDEN_SIZE, 
         NUM_LAYER, 
         OUTPUT_SIZE)
optimizer = torch.optim.Adam(G.parameters(), lr=LR)  # optimize all cnn parameters
loss_function = nn.MSELoss()

G.load_state_dict(torch.load('G_Model\GANRNN_B100_HG16_1_HD8_2_E350-150_G.pth'))
G = G.eval() 


# ---------------------------- Power of rest data ----------------------------
rest_signal, rest_rd, rest_marker = G_Input(data_rest_tensor, rd_rest_tensor, marker_rest_tensor, Sample, Data_rest)

for i_rest in range(Data_rest):
    data_x=rest_signal[:,-batch:].reshape(-1, batch, 1)
    data_rd=rest_rd[:,i_rest:i_rest+batch].reshape(-1, batch, 1)
    data_m=rest_marker[:,i_rest:i_rest+batch].reshape(-1, batch, 1)
    g_state=torch.cat((data_x, data_rd, data_m), 2)
    
    output=G(g_state)
    rest_signal = torch.cat((rest_signal, output.detach()), 1)
    
    print("Num_rest: {}/{}".format(i_rest+1, Data_rest))
    
rest_ps, rest_sum = Power(rest_signal, num_fft)
rest_fiflter = signal.filtfilt(b , a, rest_signal-m_signal)  

#----------------------------------- test -------------------------------------
prediction=torch.zeros((1,Data_modu))

for delay in range(N_delay):
    Filtered_signal=rest_fiflter.copy()
    
    test_x=rest_signal.clone()
    test_rd=rest_rd.clone()
    test_m=torch.zeros(1, Data_modu+batch*2+1)
    
    for i_pre in range(batch,Data_modu+batch):
        if (i_pre + 1) % 20 == 0:
            print('Delay:{}, Prediction: {}/{}'.format(delay, i_pre, Data_modu))
                
        test_s=test_x[:,i_pre-batch:i_pre].reshape(-1, batch, 1)
        test_r=test_rd[:,i_pre-batch:i_pre].reshape(-1, batch, 1)
        test_u=test_m[:,i_pre-batch:i_pre].reshape(-1, batch, 1)
        test_X = torch.cat((test_s, test_r, test_u),2)
        
        test_ouput = G(test_X)
        test_x[:, i_pre] = test_ouput.detach()
        
        
        test_zx=test_x[:, i_pre-batch:i_pre+1].detach().numpy()-m_signal
        
        a_output=test_zx[0,-np.size(b,0):].copy()
        ar_output=a_output[::-1]
        a_filtered=Filtered_signal[0,i_pre-np.size(a,0)+1:i_pre]
        ar_filtered=a_filtered[::-1]
        
        Filtered_signal[0,i_pre]=np.dot(ar_output,b)-np.dot(ar_filtered,a[1:])
        i_post=Filtered_signal[:,i_pre-1]
        i_now=Filtered_signal[:,i_pre]
        if i_post<0 and i_now>0:
            test_m[:,i_pre+delay*step]=torch.ones((1)) * 0.5
                    
    prediction=torch.cat((prediction,test_x[:,batch:].reshape(1,-1)),0)

predictions1 = prediction[1:,:].detach().numpy()
predictions=predictions1-m_signal

# d=test_x-m_test
# filter_data = signal.filtfilt(b,a,d)
# plt.plot(d[0,:500])
# plt.plot(filter_data[0,:500])


Test_x=test_x.detach().numpy()
a1=a.detach().numpy()
b1=b.detach().numpy()
a_mark=test_m.detach().numpy()

plt.figure(figsize=(10, 7))
plt.subplot(211)
plt.plot(predictions[0,:], 'm', label='Prediction Data')
plt.ylabel("Amplitude")
plt.legend(loc=1)
plt.title('Modulated Data')
plt.subplot(212)
plt.plot(predictions[0,-1000:], 'm', label='Prediction Data')
plt.ylabel("Amplitude")
plt.legend(loc=1)

plt.figure(figsize=(10, 7))
plt.subplot(211)
plt.plot(Filtered_signal[0,:], 'm', label='Prediction Data')
plt.ylabel("Angle")
plt.legend(loc=1)
plt.title('Filtered Data')
plt.subplot(212)
plt.plot(Filtered_signal[0,-1000:], 'm', label='Prediction Data')
plt.ylabel("Angle")
plt.legend(loc=1)

scio.savemat('GeneratedData\GNet_PhaLok_B100_HG16_1_HD8_2_E300-100_G_1.6.mat', {'predictions':predictions,
                                                                                'rest_signal':rest_signal,
                                                                                'rest_rd':rest_rd,
                                                                                'rest_marker':rest_marker,
                                                                                'rest_fiflter':rest_fiflter})
# plt.figure()
# plt.plot(test_zx[0,:],'r')
# plt.plot(Test_x[0,:],'k')

# f_prediction=signal.filtfilt(b,a,predictions)
# plt.figure(figsize=(10, 7))
# plt.plot(f_prediction[0,:], 'm', label='Prediction Data')
# plt.ylabel("Angle")
# plt.legend(loc=1)
# plt.title('RNN_alpha')