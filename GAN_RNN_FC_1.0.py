# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:43:38 2022

@author: user

Normalization: 0~1
Noise: uniform 0~1
Stimu: 0~0.5
G_loss = g_loss.mean() + pre_loss + var_loss

pre_loss: 第一点的预测误差
pre_loss = g_loss_function1(fake_data1[:,0], data_y[:,0])

Downsample到500Hz

"""


import numpy as np
import torch
import random
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy import signal
import torch.nn.functional as F
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

class Discriminator(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim, f):
        super(Discriminator,self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        self.f = f

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return self.f(x)

def Noise_Uniform(data):
    x=np.size(data,0)
    y=np.size(data,1)
    noise=np.zeros((x,y))
    
    for i in range(x):
        for j in range(y):
            noise[i,j]=random.random()
    
    out=torch.from_numpy(noise).float()
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

def G_Input(G_Sample):
    data_x=torch.zeros((G_Sample,batch))
    data_rd=torch.zeros((G_Sample,batch+pre_len))
    data_m=torch.zeros((G_Sample,batch+pre_len))
    data_y=torch.zeros((G_Sample,pre_len))
    for i in range(half_sample): 
        random_sub = random.randint(0,all_sample-1)
        data_x[i,:]=VEP_data[random_sub, :batch].clone()
        data_rd[i,:]=VEP_rd[random_sub, :batch+pre_len].clone()
        data_m[i,:]=VEP_marker[random_sub, :batch+pre_len].clone()
        data_y[i,:]=VEP_data[random_sub, batch:batch+pre_len].clone()
    
    for i in range(half_sample, G_Sample):
        random_sub = random.randint(0,all_sample-1)
        data_x[i,:]=rest_data[random_sub, :batch].clone()
        data_rd[i,:]=rest_rd[random_sub, :batch+pre_len].clone()
        data_m[i,:]=rest_marker[random_sub, :batch+pre_len].clone()
        data_y[i,:]=rest_data[random_sub, batch:batch+pre_len].clone()   
    
    if torch.cuda.is_available():
        data_x=data_x.cuda()
        data_rd=data_rd.cuda()
        data_m=data_m.cuda()
        data_y=data_y.cuda()
        
    return data_x, data_rd, data_m, data_y
    
def RealData(Real_Sample):
    real_data=torch.zeros((Real_Sample, pre_len))
    real_rd=torch.zeros((Real_Sample, pre_len))
    real_marker=torch.zeros((Real_Sample, pre_len))
    for i in range(int(Real_Sample/2)):  
        random_sub = random.randint(0,all_sample-1)
        
        real_data[i,:] = VEP_data[random_sub, batch:batch+pre_len].clone()
        real_rd[i,:] = VEP_rd[random_sub, batch:batch+pre_len].clone()
        real_marker[i,:] = VEP_marker[random_sub, batch:batch+pre_len].clone()
    
    for i in range(int(Real_Sample/2), Real_Sample):  
        random_run = random.randint(0,all_sample-1)
        
        real_data[i,:] = rest_data[random_run, batch:batch+pre_len].clone()  
        real_rd[i,:] = rest_rd[random_run, batch:batch+pre_len].clone()
        real_marker[i,:] = rest_marker[random_run, batch:batch+pre_len].clone()
    
    if torch.cuda.is_available():
        real_data=real_data.cuda()
        real_rd=real_rd.cuda()
        real_marker=real_marker.cuda()
        
    return real_data, real_rd, real_marker

def D_training(G, D, if_training):
    # train with real data
    data_x, data_rd, data_m, data_y=G_Input(Sample)
    real_data, real_rd, real_marker = RealData(Sample)
    d_real_data = real_data.reshape(-1, 1, pre_len)
    d_real_decision = D(d_real_data)
    
    # train with fake data
    for i_pre in range(pre_len):
        signal_g=data_x[:,i_pre:i_pre+batch].reshape(-1, batch, 1)
        rd_g=data_rd[:,i_pre:i_pre+batch].reshape(-1, batch, 1)
        stimu_g=data_m[:,i_pre:i_pre+batch].reshape(-1, batch, 1)
        state = torch.cat((signal_g, rd_g, stimu_g), 2)
        
        output = G(state)
        
        g_next=output.detach()
        data_x = torch.cat((data_x, g_next), 1)
    
    fake_data=data_x[:,batch:]
    # noise3=Noise_Uniform(fake_data)
    d_fake_data = Variable(fake_data).reshape(-1, 1, pre_len)
    d_fake_decision = D(d_fake_data)
    
    # d_loss calculation
    d_loss = (-torch.log(torch.clamp(d_real_decision, 1e-1000000, 1.0))) \
        -torch.log(1-torch.clamp(d_fake_decision, 0.0, 1.0-1e-1000000))
    D_loss = d_loss.mean()
    
    if if_training:
        D.zero_grad()
        D_loss.backward()
        d_optimizer.step()
        
    # return valus 
    DNet_loss = extract(D_loss)[0]
    dr = extract(d_real_decision)[0]
    df = extract(d_fake_decision)[0]
    return DNet_loss, dr, df

def G_training(G, D, if_training):
    replay_memory=[]
    data_x, data_rd, data_m, data_y=G_Input(Sample)
    for i_pre in range(pre_len):
        signal_g=data_x[:,i_pre:i_pre+batch].reshape(-1, batch, 1)
        rd_g=data_rd[:,i_pre:i_pre+batch].reshape(-1, batch, 1)
        stimu_g=data_m[:,i_pre:i_pre+batch].reshape(-1, batch, 1)
        state = torch.cat((signal_g, rd_g, stimu_g), 2)
        
        output = G(state)
        
        g_next=output.detach()
        data_x = torch.cat((data_x, g_next), 1)
        
        replay_memory.append([state, g_next])

    state_batch, next_batch = zip(*replay_memory)
    state_batch1 = torch.cat(tuple(state for state in state_batch),0)
    total_output = G(state_batch1)
    
    fake_data=total_output.reshape(-1, Sample)
    fake_data1=fake_data.transpose(0,1)
    # noise2=Noise_Uniform(fake_data1)
    g_fake_data = Variable(fake_data1).reshape(-1, 1, pre_len)
    dg_fake_decision = D(g_fake_data)
    
    g_loss = -torch.log(torch.clamp(dg_fake_decision, 1e-1000000, 1.0))
    pre_loss = g_loss_function1(fake_data1[:,0], data_y[:,0])
    var_loss = g_loss_function2(torch.var(fake_data1), torch.var(data_y))
    G_loss = g_loss.mean() + pre_loss + var_loss

    
    if if_training:
        G.zero_grad()
        G_loss.backward()
        g_optimizer.step()  # Only optimizes G's parameters
        if epoch%10==0:
            plt.plot(fake_data1[0,:].cpu().detach().numpy())
            plt.title('Epoch:{}, D_LR:{}'.format(epoch, learning_rate ))
            plt.pause(0.01)
    
    GNet_loss = extract(G_loss)[0]
    dgf = extract(dg_fake_decision)[0]
    return GNet_loss, dgf

def extract(v):
    return v.data.storage().tolist()

def get_generator_input_sampler():
    return lambda m, n, p: torch.rand(m, n, p)

print (torch.cuda.get_device_capability(0))
print (torch.cuda.get_device_name(0))



# ----------------------------- Load Data -----------------------------
Data_RERP = scio.loadmat("Filter_letswave\RVEP_my_8-12Hz.mat")
data = Data_RERP["Data"]
marker = Data_RERP["Stimu_impulse"]
Data_REST = scio.loadmat("Filter_letswave\REST_my_8-12Hz.mat")
data_rest = Data_REST["Data"]
marker_rest = Data_REST["Stimu_impulse"]

Fs=1000
fs = 500
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
marker_tensor = (torch.from_numpy(marker_downsample)*0.5).float()
data_rest_tensor = torch.from_numpy(data_rest_scaled).float()
rd_rest_tensor = Noise_Uniform(data_rest_scaled).float()
marker_rest_tensor = (torch.from_numpy(marker_rest_downsample)*0.5).float()


# ----------------------------- Parameters -----------------------------
pre_t = 1
sample_t = 1.2
dt = 1/fs
y_len = 1
pre_len = int(pre_t * fs)
# time = np.arange(0, pre_t, dt).reshape(-1, 1)

batch = 100
feature = 3
max_epochs = 300
warmup_steps = 100

VEP_data = data_tensor.reshape(-1, int(sample_t*fs))
VEP_rd = rd_tensor.reshape(-1, int(sample_t*fs))
VEP_marker = marker_tensor.reshape(-1, int(sample_t*fs))
rest_data = data_rest_tensor.reshape(-1, int(sample_t*fs))
rest_rd = rd_rest_tensor.reshape(-1, int(sample_t*fs))
rest_marker = marker_rest_tensor.reshape(-1, int(sample_t*fs))

all_sample = np.size(VEP_data, 0)
Sample = 100
half_sample = int(Sample/2)


# ----------------------------- Hyper Parameters -----------------------------
# --------- Generator ---------
g_input_size = feature
g_hidden_size = 16
g_output_size = y_len
g_num_layer = 1
g_dropout = 0         
g_lr = 1e-3
g_step = 5

# --------- Discriminator ---------
d_input_size = pre_len
d_hidden_size1 = 32
d_hidden_size2 = 16
d_output_size = 1
init_lr = 0.1
d_step = 5

# --------- Model Definition ---------
G = RNN(g_input_size, 
         g_hidden_size, 
         g_num_layer, 
         g_output_size)

g_optimizer = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.0, 0.9))
g_loss_function1 = nn.MSELoss()
g_loss_function2 = nn.L1Loss()

discriminator_activation_function = torch.sigmoid
D = Discriminator(d_input_size,
                  d_hidden_size1,
                  d_hidden_size2,
                  d_output_size,
                  discriminator_activation_function) 
d_optimizer = torch.optim.Adam(D.parameters(), lr=init_lr, betas=(0.0, 0.9))
criterion = nn.MSELoss()

if torch.cuda.is_available():
    G = G.cuda()
    D = D.cuda()


# ----------------------------- Model Training -----------------------------
# ----------------- Pre Training -----------------
# ------------------- Training -------------------
## model G input size is (-1,2,batch), output size is (1,2,1)
## model D input size is (-1), output size is (1)
# rate=np.zeros((max_epochs))
n=0
for epoch in range(max_epochs):
    for p in D.parameters():  # reset requires_grad
        p.requires_grad = True
     
    if warmup_steps and epoch < warmup_steps:
        warmup_percent_done = epoch / warmup_steps
        warmup_learning_rate = init_lr * warmup_percent_done  #gradual warmup_lr
        
        learning_rate = warmup_learning_rate
        d_optimizer.param_groups[0]["lr"]=learning_rate
    else:
        learning_rate = learning_rate*0.98
        d_optimizer.param_groups[0]["lr"]=learning_rate 
    # rate[epoch]=learning_rate
    
    # 1 Train D odn real+fake
    # for i in range(d_step):
    d_loss, dr, df= D_training(G, D, if_training=True)
    
    # # 2. Train G on D's response (but DO NOT train D on these labels)
    # for j in range(g_step):
    g_loss, dgf= G_training(G, D, if_training=True)
        
       
    g_loss, dgf = G_training(G, D, if_training=False)
    d_loss, dr, df= D_training(G, D, if_training=False)
    print('Epoch: {}/{}, D_loss: {},  G_loss: {}, d_r_d:{}, d_f_d:{}, d_gf_d:{}'
          .format(epoch, max_epochs, d_loss, g_loss, dr, df, dgf))
    
    # if dgf==0.5:
    #     n=n+1
        
    # if n>10:
    #     break


torch.save(G.state_dict(), 'GAN\GANRNN_FC_B100_HG16_1_HD32_16_E300-100_S100_G_1.0.pth') # save model parameters to files
torch.save(D.state_dict(), 'GAN\GANRNN_FC_B100_HG16_1_HD32_16_E300-100_S100_D_1.0.pth')

end = time.time()
print('Training Time:{}'.format(end-start))