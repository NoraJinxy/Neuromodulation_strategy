# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:12:46 2022

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


# ----------------------------- Load Data -----------------------------
fs = 500
lambda_stimu = 0.5

Data_pha = scio.loadmat("GeneratedData\PhaLoc_PMax_data.mat")
Data_pmax = Data_pha['Data_pmax']
Data_pmin = Data_pha['Data_pmin']
Data_pmax = torch.from_numpy(Data_pmax)
Data_pmin = torch.from_numpy(Data_pmin)

Data_input = scio.loadmat("GeneratedData\GNet_PhaLok_B100_HG16_1_HD8_2_E300-100_G_1.6.mat")
rest_signal = torch.tensor(Data_input['rest_signal']).float()
rest_rd = torch.tensor(Data_input['rest_rd']).float()
rest_marker = torch.tensor(Data_input['rest_marker']).float()
rest_filter = torch.tensor(Data_input['rest_fiflter']).float()


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


#----------------------------------- test -------------------------------------
prediction=torch.zeros((2,Data_modu))
filtered=np.zeros((2,Data_modu))
marker=torch.zeros((2,Data_modu))
num_stimu=torch.zeros((2))
Delay=[64, 88]

for delay_num in range(2):
    delay=int(Delay[delay_num])
    Filtered_signal=rest_filter.clone().numpy()
    
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
            num_stimu[delay_num]=num_stimu[delay_num]+1
                    
    prediction[delay_num,:]=test_x[:,batch:batch+Data_modu].reshape(1,-1)
    filtered[delay_num,:]=Filtered_signal[:,batch:batch+Data_modu].reshape(1,-1)
    marker[delay_num,:]=test_m[:,batch:batch+Data_modu].reshape(1,-1)

predictions=prediction-m_signal

# d=test_x-m_test
# filter_data = signal.filtfilt(b,a,d)
# plt.plot(d[0,:500])
# plt.plot(filter_data[0,:500])

for i in range(2):
    Modu_signal = filtered[i,:].reshape(1,-1)
    Modu_marker = marker[i,:].reshape(1,-1).detach().numpy()
    
    stimu_pos = np.argwhere(Modu_marker[0,:]>0).reshape(1,-1)
    stimu_time = Time[stimu_pos,0]
    stimu_mark = Modu_signal[0, stimu_pos]
    stimu_diff = np.diff(stimu_pos)
    
    m_signal = Modu_signal.mean()
    Modu_signal_m = Modu_signal - m_signal
    modu_hilbert = hilbert(Modu_signal_m)
    modu_angle = np.angle(hilbert(Modu_signal_m))
    Modu_x = np.cos(modu_angle) * abs(modu_hilbert)
    Modu_y = np.sin(modu_angle) * abs(modu_hilbert)
    Stimu_x = Modu_x[0, stimu_pos]
    Stimu_y = Modu_y[0, stimu_pos]
    
    
    plt.figure(figsize=(6, 5))
    plt.plot(Modu_x[0,:],Modu_y[0,:])
    plt.plot(Stimu_x[0,:], Stimu_y[0,:],'r.')
    plt.xlabel('x')
    plt.ylabel('y')
    
    
    plt.figure()
    plt.plot(Time[:,0], Modu_signal[0,:])
    plt.plot(stimu_time[0,:],stimu_mark[0,:],'r.')
    # plt.xlim([8,15])
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    
    plt.figure()
    plt.plot(stimu_diff[0,:], label='{} Stimulus'.format(int(num_stimu[i])))
    # plt.ylim([0, 100])
    plt.legend(loc=1)
    plt.xlabel('Stimuli')
    plt.ylabel('Stimulation Interval [ms]')

# ----------------------------- Power Spectrum -----------------------------
# rest_ps, rest_sum = Power(rest_signal[0,-Data_power:].reshape(1,-1), num_fft)
# modu_ps, modu_sum = Power(predictions[0,-Data_power:].reshape(1,-1), num_fft)
# phas_ps, phas_sum = Power(Data_pmin[:,-Data_power:], num_fft)
# Rest_ps = 10*np.log10(rest_ps.detach().numpy())
# Modu_ps = 10*np.log10(modu_ps.detach().numpy())
# Phas_ps = 10*np.log10(phas_ps.detach().numpy())


# plt.figure()
# plt.plot(rest_ps[1:30],label='Rest={:2f}'.format(rest_sum))
# plt.plot(modu_ps[1:30],label='S Model={:2f}'.format(modu_sum))
# plt.plot(phas_ps[1:30],label='Phase Lock={:2f}'.format(phas_sum))
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Power [dB]")
# plt.title('Suppression')
# # plt.title('Improvement')
# plt.legend(loc=1)
