# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:59:50 2022

@author: user

############# GPU #############

1. 先训练刺激间隔
2. 再训练power


Stimu model: BiRNN
    input size = 100
    input = [EEG, Stimu_seq]
    Stimu_seq ∈ [0, 1]
    
    hiden size = 32/2

reward-interval: 1/-1   interval=(30-100)
reward-power: 1/-1

''

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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        
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
        h0 = Variable(torch.zeros(self.num_layers*2,x.size(0),self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers*2,x.size(0),self.hidden_size)).cuda()
 
        out,_ = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out 

def Normalization(data):
    
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
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
    
    if torch.cuda.is_available():
        data_x=data_x.cuda()
        data_rd=data_rd.cuda()
        data_m=data_m.cuda()
    return data_x, data_rd, data_m

def next_frame1(action, i_pos):
    # Next State
    data_x=modu_filter[:,k-q_input+1:k+1].clone().reshape(-1, 1, q_input)+m_signal
    data_m=q_marker[:,k-q_input+1:k+1].clone().reshape(-1, 1, q_input)
    next_state=torch.cat((data_x, data_m), 1)
    
    
    # reward
    reward1=0
    distribution_a=False
    terminal=False
    j1 = action > 0
    if j1:  
        distribution_a=True
        pos_action[0,i_pos]=k
        i_pos += 1
        if i_pos>1:        
            interval=pos_action[0,i_pos-1]-pos_action[0,i_pos-2]
            if interval<stimu_threshod_l or interval>stimu_threshod_h:
                reward1=-1
            else: 
                reward1=1
            
    return next_state, reward1, terminal, distribution_a, i_pos

def next_frame2(action, i_pos):
    # Next State
    data_x=modu_filter[:,k-q_input+1:k+1].clone().reshape(-1, 1, q_input)+m_signal
    data_m=q_marker[:,k-q_input+1:k+1].clone().reshape(-1, 1, q_input)
    next_state=torch.cat((data_x, data_m), 1)
    
    
    # reward
    reward=0
    reward1=0
    reward2=0
    distribution_a=False
    terminal=False
    j1 = action > 0
    if j1:  
        distribution_a=True
        pos_action[0,i_pos]=k
        i_pos += 1
        if i_pos>1:        
            interval=pos_action[0,i_pos-1]-pos_action[0,i_pos-2]
            if interval<stimu_threshod_l or interval>stimu_threshod_h:
                reward1=-1
            else: 
                reward1=1
                
        # Power reward
        data_sx=modu_signal[:,k-input_size:k].clone()
        data_nx=data_sx.clone()
        
        data_=torch.zeros((1, Data_pre))        
        data_nm=torch.cat((modu_marker[:,k-input_size:k], data_), 1)
        data_sm=data_nm.clone()
        data_sm[0,input_size] = action * lambda_stimu
        
        data_rd = Noise_Uniform(data_nm)
        
        for i in range(Data_pre):
            data_x1=data_sx[0,-input_size:].reshape(-1, input_size, 1)
            data_r1=data_rd[0,i:i+input_size].reshape(-1, input_size, 1)
            data_m1=data_sm[0,i:i+input_size].reshape(-1, input_size, 1)
            state1=torch.cat((data_x1, data_r1, data_m1), 2)
            
            if torch.cuda.is_available():
                    state1=state1.cuda()
            output1=G(state1).cpu()
            data_sx=torch.cat((data_sx, output1.detach()), 1)
            
            data_x2=data_nx[0,-input_size:].reshape(-1, input_size, 1)
            data_r2=data_rd[0,i:i+input_size].reshape(-1, input_size, 1)
            data_m2=data_nm[0,i:i+input_size].reshape(-1, input_size, 1)
            state2=torch.cat((data_x2, data_r2, data_m2), 2)
            
            if torch.cuda.is_available():
                    state2=state2.cuda()
            output2=G(state2).cpu()
            data_nx=torch.cat((data_nx, output2.detach()), 1)
    
        ps_stimu,sum_stimu=Power(data_sx[:,-Data_pre:], num_fft)
        ps_no_stimu,sum_no_stimu=Power(data_nx[:,-Data_pre:], num_fft)
        
        power_supression=sum_stimu<sum_no_stimu
        # reward4=torch.tanh((sum_stimu-sum_no_stimu)/(sum_no_stimu*0.1)).detach().numpy()
        if power_supression:
            terminal=True
            reward2=1
        else:
            reward2=-1
    
    reward=reward+reward1+reward2
    return next_state, reward, terminal, distribution_a, i_pos

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to Modulate Generated Model""")
    parser.add_argument("--batch_size", type=int, default=500, help="The number of EEG per batch")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.05)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=22)
    parser.add_argument("--no_random_iters", type=int, default=2)
    parser.add_argument("--replay_memory_size", type=int, default=5000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args
opt = get_args()

# ----------------------------- Load Data -----------------------------
Data_RERP = scio.loadmat("Filter_letswave\RVEP_my_8-12Hz.mat")
data = Data_RERP["Data"]
marker = Data_RERP["Stimu_impulse"]
Data_REST = scio.loadmat("Filter_letswave\REST_my_8-12Hz.mat")
data_rest = Data_REST["Data"]
marker_rest = Data_REST["Stimu_impulse"]

Fs=1000
fs = 500
lambda_stimu = 0.5
step = int(Fs/fs)
run = np.size(data, 0)
Data_len = np.size(data, 1)
T = int(Data_len/Fs)
data_len = fs*T

# Downsample
data_downsample = signal.resample(data, data_len, axis=1)
data_rest_downsample = signal.resample(data_rest, data_len, axis=1)
marker_downsample = np.zeros((run, data_len))
marker_rest_downsample = np.zeros((run, data_len))

for r in range(run):
    m_vep = marker[r,:].reshape(-1, step)
    marker_downsample[r,:] = np.sum(m_vep, 1)

# Normalize and change data type
data_scaled = Normalization(data_downsample)
data_rest_scaled = Normalization(data_rest_downsample)

data_tensor = torch.from_numpy(data_scaled).float()
rd_tensor = Noise_Uniform(data_scaled).float()
marker_tensor = (torch.from_numpy(marker_downsample)*lambda_stimu).float()
data_rest_tensor = torch.from_numpy(data_rest_scaled).float()
rd_rest_tensor = Noise_Uniform(data_rest_scaled).float()
marker_rest_tensor = (torch.from_numpy(marker_rest_downsample)*lambda_stimu).float()


# -------------------------------- Parameters --------------------------------
Dt=1/fs
T_rest=5
T_modu=5
T_pre=1

input_size = int(0.2*fs)
q_input = input_size

Data_len=T_modu*fs + q_input
Data_rest=T_rest*fs
Data_modu = T_modu*fs
Data_pre = T_pre*fs

stimu_threshod_h=100
stimu_threshod_l=30

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
G.load_state_dict(torch.load('G_Model\GANRNN_B100_HG16_1_HD8_2_E350-150_G_cpu.pth'))
G=G.eval()

model = DeepQNetwork(q_input_size,
                     q_hidden_size,
                     q_num_layer,
                     q_output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
if torch.cuda.is_available():
    G=G.cuda()
    model=model.cuda()
    

# ---------------------------- Power of rest data ----------------------------
rest_signal, rest_rd, rest_marker = G_Input(data_rest_tensor, rd_rest_tensor, marker_rest_tensor, Sample, Data_rest)


for i_rest in range(Data_rest):
    data_x=rest_signal[:,-input_size:].reshape(-1, input_size, 1)
    data_rd=rest_rd[:,i_rest:i_rest+input_size].reshape(-1, input_size, 1)
    data_m=rest_marker[:,i_rest:i_rest+input_size].reshape(-1, input_size, 1)
    g_state=torch.cat((data_x, data_rd, data_m), 2)
    
    output=G(g_state)
    rest_signal = torch.cat((rest_signal, output.detach()), 1)
    
rest_ps, rest_sum = Power(rest_signal, num_fft)
rest_fiflter = signal.filtfilt(b , a, rest_signal.cpu())


# ------------------------------ Model training ------------------------------ 
replay_memory = []
num_termi=0

for train_num in range(2):
    iter = 0
    random_iters = opt.num_iters - opt.no_random_iters
    
    while iter < opt.num_iters:
        rd_begin=random.randint(500,Data_rest-q_input)
        modu_signal=torch.zeros((Sample, Data_modu)).float()
        modu_filter=torch.zeros((Sample, Data_modu)).float()
        modu_signal[:,:q_input]=rest_signal[0, rd_begin:rd_begin+q_input].clone()
        modu_filter[:,:q_input]=torch.from_numpy(rest_fiflter[0, rd_begin:rd_begin+q_input].copy())
        modu_rd=Noise_Uniform(modu_signal)
        modu_marker=torch.zeros((Sample, Data_modu+1))
        modu_marker[:,:q_input]=lambda_stimu
        q_marker=torch.zeros((Sample, Data_modu+1))
        q_marker[:,:q_input]=lambda_stimu
        pos_action=torch.zeros((1, Data_modu))
        
        epsilon = opt.final_epsilon + (
                (random_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / random_iters)
        
        i_pos=0
        for k in range(Data_modu):
            reward=0
            if k>q_input-1:
                # Real-time data generation
                data_gx=modu_signal[:,k-input_size:k].clone().reshape(-1, input_size, 1)
                data_grd=modu_rd[:,k-input_size:k].clone().reshape(-1, input_size, 1)
                data_gm=modu_marker[:,k-input_size:k].clone().reshape(-1, input_size, 1)
                g_state=torch.cat((data_gx, data_grd, data_gm), 2)
                
                if torch.cuda.is_available():
                    g_state=g_state.cuda()
                output=G(g_state).cpu()
                modu_signal[:,k] = output.detach()
                m=modu_signal[:,:k].mean()
                
                # Stimuli generation
                test_zx=modu_signal[:,k-input_size:k+1].clone().detach().numpy()
                a_output=test_zx[0,-np.size(b,0):].copy()
                ar_output=a_output[::-1]
                a_filtered=modu_filter[0,k-np.size(a,0)+1:k].clone().detach().numpy()
                ar_filtered=a_filtered[::-1]
                modu_filter[0,k]=torch.tensor(np.dot(ar_output,b)-np.dot(ar_filtered,a[1:]))
                # plt.plot(modu_signal[0,:k].numpy())
                # plt.plot(modu_filter[0,:k].numpy()+0.5)
                # plt.plot(modu_marker[0,:k].numpy())
                
                data_qx=modu_filter[:,k-q_input:k].clone().reshape(-1, 1, q_input)+m_signal
                data_qm=q_marker[:,k-q_input:k].clone().reshape(-1, 1, q_input)
                q_state=torch.cat((data_qx, data_qm), 1)
                
                if torch.cuda.is_available():
                    q_state=q_state.cuda()
                v_velue = model(q_state).cpu()
                
                model_action=0
                u = random.random()
                random_action = u <= epsilon
                if random_action:
                    action = random.randint(0, 1)
                    print("---Iteration: {}/{}, Action: {}, Perform a random action---".format(
                        iter + 1, 
                        opt.num_iters,
                        action))
                else:
                    action = torch.argmax(v_velue[0,:])
                    model_action = torch.argmax(v_velue[0,:])
                
                if action>0:
                    modu_marker[:,k] = action * lambda_stimu
                    q_marker[:,k] = action * lambda_stimu
                
                # Next state
                if train_num==0:
                    next_state, reward, terminal, distribution_a, i_pos = next_frame1(action, i_pos)
                else:
                    next_state, reward, terminal, distribution_a, i_pos = next_frame2(action, i_pos)
                  
                replay_memory.append([q_state.detach(), action, reward, next_state.detach(), terminal])
                if len(replay_memory) > opt.replay_memory_size:
                    del replay_memory[0]
                batch = random.sample(replay_memory, min(len(replay_memory), opt.batch_size))
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
            
                state_batch = torch.cat(tuple(state for state in state_batch))
                action_batch = torch.from_numpy(
                    np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.double))
                reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.double)[:, None])
                next_state_batch = torch.cat(tuple(state for state in next_state_batch))
                
                if torch.cuda.is_available():
                    state_batch=state_batch.cuda()
                    next_state_batch=next_state_batch.cuda()
                current_prediction_batch = model(state_batch.detach()).cpu()
                next_prediction_batch = model(next_state_batch.detach()).cpu()
            
                y_batch = torch.cat(
                    tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                          zip(reward_batch, terminal_batch, next_prediction_batch)))
            
                q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
                optimizer.zero_grad()
                loss = criterion(q_value, y_batch)
                loss.backward()
                optimizer.step()
                
                
                if terminal==True:
                    termi=1
                elif terminal==False:
                    termi=0
                if iter>random_iters:
                    if termi>0:
                        num_termi = num_termi+1
                    else: 
                        num_termi = 0
                        
                if distribution_a or terminal:
                    print("Iteration: {}/{}-{},Time: {}/{},Loss: {},Reward: {},Q-value: {},Model_stimu: {}/{},NUm_Terminal:{},Terminal:{}".format(
                        iter + 1,
                        opt.num_iters,train_num,
                        k, Data_modu, 
                        loss,
                        reward, 
                        torch.max(v_velue),
                        model_action,
                        action,
                        num_termi,
                        termi))
        
        if num_termi>=30:
            break
        iter += 1

torch.save(model, "{}/StimuModel_Suppression_H32_2_2.1.1".format(opt.saved_path))

end = time.time()
print('Training Time:{}'.format(end-start))

