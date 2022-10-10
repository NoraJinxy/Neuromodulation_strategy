# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:11:35 2021

@author: user
"""


import numpy as np
import torch
import random
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
from torch.fft import fft
from scipy import signal
import torch.nn.functional as F
from torch.autograd import Variable
torch.set_default_tensor_type(torch.DoubleTensor)
np.random.seed(0)


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
    
    out=torch.from_numpy(noise)
    return out

def Noise_Gaussian(data, mu, sigma):
    # 正态分布随机数
    x=np.size(data,0)
    y=np.size(data,1)
    noise=np.zeros((x,y))
    
    for i in range(x):
        for j in range(y):
            noise[i,j]=random.normalvariate(mu=mu, sigma=sigma)
    
    out=torch.from_numpy(noise)  
    return out

def G_Input(G_Sample):
    data_x=torch.zeros((G_Sample,batch))
    data_rd=torch.zeros((G_Sample,batch+test_len))
    data_m=torch.zeros((G_Sample,batch+test_len))
    data_y=torch.zeros((G_Sample,test_len))
    for i in range(G_Sample-1):  
        random_sub=random.randint(0,sub-1)
        train_begin=random.randint(0,int(data_len-batch-test_len-1))
        
        data_x[i,:]=data_tensor[random_sub,train_begin:train_begin+batch].clone()
        data_rd[i,:]=rd_tensor[random_sub,train_begin:train_begin+batch+test_len].clone()
        data_m[i,:]=marker_tensor[random_sub,train_begin:train_begin+batch+test_len].clone()
        data_y[i,:]=data_tensor[random_sub,train_begin+batch:train_begin+batch+test_len].clone()
    
    random_run=random.randint(0,sub-1)
    rest_begin=random.randint(0,int(data_len-batch-test_len-1))
    
    data_x[G_Sample-1,:]=data_rest_tensor[random_run,rest_begin:rest_begin+batch].clone()
    data_rd[G_Sample-1,:]=rd_tensor[random_run,rest_begin:rest_begin+batch+test_len].clone()
    data_m[G_Sample-1,:]=marker_rest_tensor[random_run,rest_begin:rest_begin+batch+test_len].clone()
    data_y[G_Sample-1,:]=data_rest_tensor[random_run,rest_begin+batch:rest_begin+batch+test_len].clone()
    
    return data_x, data_rd, data_m, data_y

def Normalization(data):
    minVals = data.min(1)
    maxVals = data.max(1)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    for i in range(m):
        normData[i, :] = (data[i, :] - minVals[i]) / ranges[i]
    
    return normData

def Power(filtered_signal,num_fft):
    Y = fft(filtered_signal.detach(), num_fft)
    Y = torch.abs(Y)
    ps = Y**2 / num_fft
    ps=ps[0,:num_fft//2]
    
    return ps

def extract(v):
    return v.data.storage().tolist()

def get_generator_input_sampler():
    return lambda m, n, p: torch.rand(m, n, p)


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

data_tensor = torch.from_numpy(data_scaled)
rd_tensor = Noise_Uniform(data_scaled)
marker_tensor = torch.from_numpy(marker_downsample)*0.5
data_rest_tensor = torch.from_numpy(data_rest_scaled)
rd_rest_tensor = Noise_Uniform(data_rest_scaled)
marker_rest_tensor = torch.from_numpy(marker_rest_downsample)*0.5


# ----------------------------- Parameters -----------------------------
pre_t = 1
dt = 1/fs
y_len = 1
pre_len = int(pre_t * fs)

batch = int(0.2*fs)
feature = 3
num_fft = fs


# ----------------------------- Hyper Parameters -----------------------------
# --------- Generator ---------
g_input_size = feature
g_hidden_size = 16
g_output_size = y_len
g_num_layer = 1

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

discriminator_activation_function = torch.sigmoid
D = Discriminator(d_input_size,
                  d_hidden_size1,
                  d_hidden_size2,
                  d_output_size,
                  discriminator_activation_function)
        
G.load_state_dict(torch.load('GAN\GANRNN_FC_B100_HG16_1_HD32_16_E300-100_S100_G_1.0.pth'))
D.load_state_dict(torch.load('GAN\GANRNN_FC_B100_HG16_1_HD32_16_E300-100_S100_D_1.0.pth'))
G=G.eval().cpu()
D=D.eval().cpu()


# ------------- Test -------------
sub = np.size(data_tensor,0)
data_len = np.size(data_tensor,1)

test_sample=30
test_t=20
test_len=test_t*fs
    
test_x, test_rd, test_m, test_y = G_Input(test_sample)

xx=test_x
for j in range(test_len):
    if (j+1)%100==0:
        print('Test_len:{}/{}'.format(j,test_len))
    
    test_dx=xx[:,-batch:].reshape(-1, batch, 1)
    test_dr=test_rd[:,j:j+batch].reshape(-1, batch, 1)
    test_dm=test_m[:,j:j+batch].reshape(-1, batch, 1)
    test_X=torch.cat((test_dx, test_dr, test_dm), 2)
    
    g_prediction = G(test_X)
    g_next=g_prediction.detach()
    xx=torch.cat((xx,g_next), 1)
    
fake_data=xx[:,-test_len:]
g_fake_data = Variable(fake_data)
g_real_data = Variable(test_y)

dg_fake_decision = D(g_fake_data.reshape(-1, 1, pre_len))
dg_real_decision = D(g_real_data.reshape(-1, 1, pre_len))

fake_decision = dg_fake_decision.mean()
real_decision = dg_real_decision.mean()

torch.set_printoptions(precision=20)
print('Batch:{}, Hidden size:{}/{}'.format(batch, g_hidden_size, g_num_layer))
print('D_fake: {}'.format(fake_decision))
print('D_real: {}'.format(real_decision))

power_real = Power(test_y, num_fft)
power_fake = Power(fake_data, num_fft)
p_real = 10*np.log10(power_real)
p_fake = 10*np.log10(power_fake)

plt.figure()
plt.plot(p_real[1:100].detach().numpy(),'r', label='Real Data')
plt.plot(p_fake[1:100].detach().numpy(),'g', label='Fake Data')
plt.legend(loc=1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [dB]')

data_fake=g_fake_data.detach().numpy()
data_real=test_y.detach().numpy()
data_marker=test_m.detach().numpy()
scio.savemat('GeneratedData\GANRNN_FC_B100_HG16_1_HD32_16_E300-100_S100_1.0.mat', {'data_fake':data_fake,'data_marker':data_marker,'data_real':data_real})

plt.figure()
plt.subplot(411)
plt.plot(g_real_data[0,-5*fs:], label='Real VEP Data')
plt.ylim(0,1)
plt.xticks([])
plt.legend(loc=1)
plt.title('GANRNN_FC_B100_HG16_1_HD32_16_E300-100_S100_1.0')
plt.subplot(412)
plt.plot(g_fake_data[0,-5*fs:], label='Fake VEP Data')
plt.ylim(0,1)
plt.xticks([])
plt.legend(loc=1)
plt.subplot(413)
plt.plot(g_real_data[-1,-5*fs:], label='Real Rest Data')
plt.ylim(0,1)
plt.xticks([])
plt.legend(loc=1)
plt.subplot(414)
plt.plot(g_fake_data[-1,-5*fs:], label='Fake Rest Data')
plt.ylim(0,1)
plt.legend(loc=1)
 

# sns.set_palette("hls")
# plt.figure(dpi=120)
# sns.set(style='dark')
# sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
# r = sns.distplot(g_real_data,
#                  hist=True,
#                  kde=True,  # 开启核密度曲线kernel density estimate (KDE)
#                  kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29',
#                           # 设置外框线属性
#                           },
#                  color='#098154',
#                  axlabel='Standardized Residual',  # 设置x轴标题
#                  )
# g = sns.distplot(g_fake_data,
#                  hist=True,
#                  kde=True,  # 开启核密度曲线kernel density estimate (KDE)
#                  )
