import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import buttord,butter,freqz,bilinear,filtfilt
from scipy.signal import TransferFunction as tf


import streamlit as st
import numpy as np
from scipy.signal import buttord,butter,freqz,bilinear,lp2hp
import matplotlib.pyplot as plt
import math
import scipy.io as spio
from scipy.signal import TransferFunction as tf

st.markdown("<h1 style='text-align: center; color: white;font-size: 40px;'>Design and Implementation of Highpass IIR Filter For ECG Signal Using Butterworth and BLT</h1>",
            unsafe_allow_html=True)
st.sidebar.markdown("By Antony Jerald ")
st.sidebar.markdown("121901004")

order_options = [1,2,3,4,5,6,7]
order = st.sidebar.select_slider("Choose the filter order",options = order_options)

#order = 5

ECG_baseline_10s = spio.loadmat('baseline_ecg_fs_360_10seconds.mat')['ecg'][0][:]       #10 sec ECG signal
t_10 =  spio.loadmat('baseline_ecg_fs_360_10seconds.mat')['t'][0][:]

Fs = 360;             #Sampling frequency                    
T = 1/Fs;             #Sampling period       
L = 3600;             #Length of signal




#Given Data
Ap = 3;      #Passband Attenuation
As = 40;     #Stopband Attenuation
Fc = 1;      #Cutoff Frequency


wd = 2*np.pi*Fc;              #Digital Angular Frequency;
wa = (2/T)*np.tan(wd*T/2);    #Analog Angular Frequency after frequency  prewarp;

# scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba')[source]
# #Taking filter order as 5
# num = [1];
# den = [1, 3.2361, 5.2361, 5.2361, 3.2361, 1];

def filter_coefficents(order):
    f_coeff = {1:([1],[1,1]), 2:([1],[1 , 1.4142 , 1]), 3:( [1],[1,2,2,1]), 4:([1],[1 , 2.6131, 3.4142 , 2.6131, 1]),5:([1],[1, 3.2361,5.2361,5.2361,3.2361,1]),6:([1],[1,3.8637,7.4641,9.1416,7.4641,3.8637,1])}
    return(f_coeff[order])
# order = 5
#############
[B,A] = butter(order,wa,btype='high',analog=True);

#Hp = tf(num,den);                       #Transfer function for analog low pass prototype
#[num,den] = filter_coefficents(order);
#[B,A] = lp2hp(num,den,wa);              #To convert lowpass prototype to highpass filter
[b,a] = bilinear(B,A,Fs);               #To perform bilear transformation 
Hz = tf(b, b, dt=T);
w = np.arange(0,np.pi,np.pi/512)
_,Hw = freqz(b, a,worN=w);              #To find magnitude response of filter
phi=180*np.unwrap(np.angle(Hw))/np.pi   #To find phase response

fig1 = plt.figure(1)                           #To plot frequenct Response
plt.subplot(2, 1, 1)
plt.plot(w/np.pi*Fs/2, np.abs(Hw))
plt.grid(which='both', axis='both')
plt.title("Frequency Response of  Pass Filter of order = $%d$ with Fc = 1Hz")
plt.title('Frequency Response of High Pass Filter with Fc = 1Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel("Magnitude Response")
plt.subplot(2, 1, 2)
plt.plot(w/np.pi*Fs/2, phi);
plt.grid(which='both', axis='both')
plt.xlabel('Frequency (Hz)')
plt.ylabel("Phase Response")


#st.map(data=(w/np.pi*Fs/2, np.abs(Hw)) zoom=None, use_container_width=True)

#To plot given ECG signal 
fig2=plt.figure(2)
plt.plot(t_10,ECG_baseline_10s)
plt.title('Original ECG signal (10sec)')
plt.xlabel('t (s)')
plt.ylabel("Amplitude")

ECG_filtered_10s = filtfilt(b,a,ECG_baseline_10s)   #To filter the ECG Signal
fig3 = plt.figure(3)
plt.plot(t_10,ECG_filtered_10s)
plt.title('High Pass Filtered Signal (10 sec)')
plt.xlabel("t (s)")
plt.ylabel("Amplitude")


ECG_baseline_5s = spio.loadmat('baseline_ecg_fs_360_10seconds.mat')['ecg'][0][:1800]       #10 sec ECG signal
t_5 =  spio.loadmat('baseline_ecg_fs_360_10seconds.mat')['t'][0][:1800]


#To plot given 5 sec ECG signal 
fig4 = plt.figure(4)
plt.plot(t_5,ECG_baseline_5s)
plt.title('Original ECG signal (5sec)')
plt.xlabel('t (s)')
plt.ylabel("Amplitude")


fig5 = plt.figure(5)
ECG_filtered_5s = filtfilt(b,a,ECG_baseline_5s)   #To filter the 5 sec ECG Signal
plt.figure(5)
plt.plot(t_5,ECG_filtered_5s)
plt.title('High Pass Filtered ECG Signal (5 sec)')
plt.xlabel("t (s)")
plt.ylabel("Amplitude")



# st.markdown("<h1 style='text-align: center; color: white;font-size: 40px;'>Design and Implementation Highpass IIR Filter For ECG Signal Using Butterworth and BLT</h1>",
#             unsafe_allow_html=True)
# st.sidebar.markdown("By Antony Jerald ")
# st.sidebar.markdown("121901004")




st.write("##")
st.subheader('Given Data')
st.text("Cutoff Frequency (Fc) = 1 Hz")
st.text('Fs=360 Hz')
st.text('Passband attenuation= 3 dB')
st.text('Stopband Attenuation=40 dB')
st.write("##")

st.subheader('Results')
st.subheader('Frequency Response of High Pass Filter Designed')
st.pyplot(fig1)
st.write("##")
st.write("##")

st.subheader('Original and Filtered ECG Signal (10 Sec)')
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig2)
with col2:
    st.pyplot(fig3)
    
st.write("##")
st.write("##")

st.subheader('Original and Filtered ECG Signal (5 Sec)')
col3, col4 = st.columns(2)
with col3:
    st.pyplot(fig4)
with col4:
    st.pyplot(fig5)
st.write("##")
st.write("##")
