# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:09:48 2022

@author: user
"""



import numpy as np
import torch
import random
import argparse
from scipy import signal
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
from torch.fft import fft
from scipy.signal import hilbert
from torch.autograd import Variable
torch.set_default_tensor_type(torch.DoubleTensor)
np.random.seed(0)
import time
start = time.time()


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
    
class DeepQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(DeepQNetwork,self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size,hidden_size,num_layers,
                                batch_first=True,bidirectional=True)
            self.fc = nn.Linear(hidden_size*2,output_size)
     
    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers*2,x.size(0),self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers*2,x.size(0),self.hidden_size))
 
        out,_ = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out 
  
def Normalization(data):
    
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    normData = torch.zeros(np.shape(data))
    m = data.shape[0]
    for i in range(m):
        normData[i, :] = (data[i, :] - minVals) / ranges
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

def Power(filtered_signal,num_fft):
    Y = fft(filtered_signal, num_fft)
    Y = torch.abs(Y)
    ps = Y**2 / num_fft
    ps = ps[0,:num_fft//2]
    sum_power = torch.sum(ps[5:15])
    
    return ps, sum_power

def G_Input(data, rd, marker, Sample, pre_len):
    sub=np.size(data,0)
    data_len=np.size(data,1)
    
    data_x=torch.zeros((Sample,input_size))
    data_rd=torch.zeros((Sample,input_size+pre_len))
    data_m=torch.zeros((Sample,input_size+pre_len))
    for i in range(Sample):  
        random_sub=random.randint(0,sub-1)
        train_begin=random.randint(0,int(data_len-input_size-pre_len-1))
        
        data_x[i,:]=data[random_sub,train_begin:train_begin+input_size].clone()
        data_rd[i,:]=rd[random_sub,train_begin:train_begin+input_size+pre_len].clone()
        data_m[i,:]=marker[random_sub,train_begin:train_begin+input_size+pre_len].clone()
    
    return data_x, data_rd, data_m

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to Modulate Generated Model""")
    parser.add_argument("--batch_size", type=int, default=100, help="The number of EEG per batch")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.05)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=100000)
    parser.add_argument("--no_random_iters", type=int, default=10000)
    parser.add_argument("--replay_memory_size", type=int, default=5000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args
opt = get_args()

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


# -------------------------------- Parameters --------------------------------
Dt=1/fs
T_rest=50
T_modu=50
T_power=int(T_rest/2)
Time=np.arange(0, T_modu, Dt).reshape(-1, 1)

input_size = int(0.2*fs)
q_input = input_size

Data_rest=T_rest*fs
Data_modu = T_modu*fs
Data_power = T_power*fs

stimu_threshod_h=int(1/8*fs)
stimu_threshod_l=int(1/12*fs)

num_fft = fs
Sample = 1

b,a=signal.butter(2,[2*8/fs,2*12/fs],'bandpass')
a=torch.from_numpy(a)
b=torch.from_numpy(b)
m_signal=0.5


# --------- Generator ---------
g_input_size = 3
g_hidden_size = 16
g_output_size = 1
g_num_layer = 1

# --------- Deep Q ---------
q_input_size = q_input
q_hidden_size = 32
q_output_size = 2
q_num_layer = 2
lr = 1e-3


# --------- Model Definition ---------
G = RNN(g_input_size, 
         g_hidden_size, 
         g_num_layer, 
         g_output_size)
G.load_state_dict(torch.load('G_Model\GANRNN_B100_HG16_1_HD8_2_E350-150_G.pth'))
G=G.eval()

model = DeepQNetwork(q_input_size,
                     q_hidden_size,
                     q_num_layer,
                     q_output_size)
model = torch.load("{}/StimuModel_Improve_H32_2_2.2.1".format(opt.saved_path), map_location=lambda storage, loc: storage)
model = model.eval()

G=G.cpu()
model=model.cpu()


# ------------------------------ S Model Test ------------------------------ 
modu_signal=torch.zeros((Sample, Data_modu))
modu_signal[:,:q_input]=rest_signal[0, :q_input].clone()
modu_filter=torch.zeros((Sample, Data_modu))
modu_filter[:,:q_input]=rest_filter[0, :q_input].clone()
modu_rd=rest_rd.clone()
modu_marker=torch.zeros((Sample, Data_modu+1))
modu_marker[0,input_size-1] = lambda_stimu
# q_marker=Noise_Uniform(modu_marker)*0.1
q_marker=torch.zeros((Sample, Data_modu+1))
q_marker[0,input_size-1] = lambda_stimu

num_stimu=0
for k in range(Data_modu):
    if k>q_input-1:
        # Real-time data generation
        data_gx=modu_signal[:,k-input_size:k].clone().reshape(-1, input_size, 1)
        data_grd=modu_rd[:,k-input_size:k].clone().reshape(-1, input_size, 1)
        data_gm=modu_marker[:,k-input_size:k].clone().reshape(-1, input_size, 1)
        g_state=torch.cat((data_gx, data_grd, data_gm), 2)
        
        output=G(g_state)
        modu_signal[:,k] = output.detach()
        
        # Stimuli generation
        test_zx=modu_signal[:,k-input_size:k+1].clone().detach().numpy()
        a_output=test_zx[0,-np.size(b,0):].copy()
        ar_output=a_output[::-1]
        a_filtered=modu_filter[0,k-np.size(a,0)+1:k].clone().detach().numpy()
        ar_filtered=a_filtered[::-1]
        modu_filter[0,k]=torch.tensor(np.dot(ar_output,b)-np.dot(ar_filtered,a[1:]))
        
        data_qx=modu_filter[:,k-q_input:k].clone().reshape(-1, 1, q_input)+m_signal
        data_qm=q_marker[:,k-q_input:k].clone().reshape(-1, 1, q_input)
        q_state=torch.cat((data_qx, data_qm), 1)
        
        v_velue = model(q_state)
        action = torch.argmax(v_velue[0,:])
        
        
        if action>0:
            modu_marker[0,k] = action * lambda_stimu
            q_marker[0,k] = action * lambda_stimu
            num_stimu=num_stimu+1
            
        print("Num_data: {}/{}, action: {}, Num_stimu: {}".format(k+1, Data_modu, action, num_stimu))

end = time.time()
print('Training Time:{}'.format(end-start))

Modu_signal = modu_filter.detach().numpy()
Modu_marker = modu_marker.detach().numpy()

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
plt.plot(stimu_diff[0,:], label='{} Stimulus'.format(num_stimu))
# plt.ylim([0, 100])
plt.legend(loc=1)
plt.xlabel('Stimuli')
plt.ylabel('Stimulation Interval [ms]')

# ----------------------------- Power Spectrum -----------------------------
rest_ps, rest_sum = Power(rest_signal[:,-Data_power:], num_fft)
modu_ps, modu_sum = Power(modu_signal[:,-Data_power:], num_fft)
phas_ps, phas_sum = Power(Data_pmax[:,-Data_power:], num_fft)
Rest_ps = 10*np.log10(rest_ps.detach().numpy())
Modu_ps = 10*np.log10(modu_ps.detach().numpy())
Phas_ps = 10*np.log10(phas_ps.detach().numpy())


plt.figure()
plt.plot(rest_ps[1:30],label='Rest={:2f}'.format(rest_sum))
plt.plot(modu_ps[1:30],label='S Model={:2f}'.format(modu_sum))
plt.plot(phas_ps[1:30],label='Phase Lock={:2f}'.format(phas_sum))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
# plt.title('Suppression')
plt.title('Improvement')
plt.legend(loc=1)

scio.savemat('GeneratedData\Data_Improve_H32_2_2.2.1.mat', {'Time':Time,
                                    'Modu_signal':Modu_signal,
                                    'stimu_time':stimu_time,
                                    'stimu_mark':stimu_mark,
                                    'Modu_x':Modu_x,
                                    'Modu_y':Modu_y,
                                    'Stimu_x':Stimu_x,
                                    'Stimu_y':Stimu_y,
                                    'Rest_ps':Rest_ps,
                                    'Modu_ps':Modu_ps,
                                    'Phas_ps':Phas_ps})