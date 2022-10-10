clc;clear all;close all

fs=500;
dt=1/fs;
window_width=1.5*fs+1;
time=-0.5:dt:1;
batch=100;

Bandwidth=[8 12];
[Filter_b,Filter_a]=butter(2,Bandwidth/(fs/2),'bandpass');

data=load('GeneratedData\GANRNN_FC_B100_HG16_1_HD32_16_E300-100_S50_1.0.mat');
data_fake=data.data_fake;
data_marker=data.data_marker;
data_real=data.data_real;
len=size(data_fake,2);
sample=size(data_fake,1)/2;

Signal_g=data_fake(1:sample,:);
marker=data_marker(1:sample,batch:end);
Signal_real=data_real(1:sample,:);


T=len/fs;
num_stimu=T*10;


m_fake=mean(Signal_g,2);
m_real=mean(Signal_real,2);

t=dt:dt:T;
% figure
% hold on
% plot(t,Signal_g(1,:))
% plot(t,Signal_real(1,:))
% box on
% xlabel('Time/s')
% ylabel('Amplitude')
% legend('Fake data','Real data')



VEP_g=zeros(sample*num_stimu,window_width);
VEP_g_filtered=zeros(sample*num_stimu,window_width);
VEP_real=zeros(sample*num_stimu,window_width);
error=zeros(sample*num_stimu,window_width);
error_filtered=zeros(sample*num_stimu,window_width);

i_stimu=0;
for i=1:sample
    clc;disp(i);
    stimu_time=find(marker(i,:)>0);
    signal_g=Signal_g(i,:)-m_fake(i);
    signal_pre_filtered=filtfilt(Filter_b,Filter_a,signal_g);
    signal_real=Signal_real(i,:)-m_real(i);
    
    for n=1:size(stimu_time,2)
        u=stimu_time(n);
        VEP_begin=u-0.5*fs;
        VEP_end=u+1*fs;
        if u>500 && VEP_end<len
            i_stimu=i_stimu+1;
            
            v_p=signal_g(VEP_begin:VEP_end);
            v_p_filtered=signal_pre_filtered(VEP_begin:VEP_end);
            v_r=signal_real(VEP_begin:VEP_end);
            
            h_p=hilbert(v_p);
            h_p_filtered=hilbert(v_p_filtered);
            h_r=hilbert(v_r);
            
            a_p=abs(angle(h_p_filtered));
            
            error(i_stimu,:)=abs(angle(h_p./h_r));
            error_filtered(i_stimu,:)=abs(angle(h_p_filtered./h_r));
            
            VEP_g(i_stimu,:)=v_p;
            VEP_g_filtered(i_stimu,:)=v_p_filtered;
            VEP_real(i_stimu,:)=v_r;
        end
    end
end
mean_VEP_g=mean(VEP_g,1);
mean_VEP_gf=mean(VEP_g_filtered,1);
mean_VEP_r=mean(VEP_real,1);
mean_error_p=mean(error,1);
mean_error_pf=mean(error_filtered,1);

figure
hold on
plot(time,mean_VEP_g(1,:))
plot(time,mean_VEP_r(1,:))
legend('Generated VEP','Real VEP')
xlabel('Time/s')
ylabel('Amplitude')
box on
% 
% figure
% hold on
% plot(time,mean_VEP_gf(1,:))
% plot(time,mean_VEP_r(1,:))
% legend('Generated VEP','Real VEP')
% xlabel('Time/s')
% ylabel('Amplitude')

% figure
% hold on
% plot(time,mean_error_p(1,:))
% ylim([0,1.6])
% xlabel('Time/s')
% ylabel('Error')