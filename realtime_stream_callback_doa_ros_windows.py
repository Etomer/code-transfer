import wave
import sys
import os
import numpy as np
import torch
#import pyaudio
import matplotlib.pyplot as plt
import time
import math
import queue
import soundfile as sf
import scipy as sp
import datetime

os.environ["SD_ENABLE_ASIO"] = "1"
import sounddevice as sd

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class SoundAnglePublisher(Node):
    def __init__(self):
        super().__init__('sound_angle_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, 'sound_angle', 10)


    def publish_numbers(self, numbers):
        msg = Float64MultiArray()
        msg.data = numbers
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')

class doa_streamer:

    def get_soundcard_device(self):
        id = -1
        devices = sd.query_devices()
        nrd = len(devices)
        for i in range(nrd):
            info = devices[i]
            if info['name'] == self.audiocard:
                id = i
        print(id)
        print(devices[id])
        return id

    def set_rec_positions(self,rec):
        self.rec = rec
        self.recdiff = torch.cat([torch.stack([(rec[j,:] - rec[i,:]) for j in range(i+1,self.CHANNELS)]) for i in range(self.CHANNELS-1)]).to(torch.float64)

    
    def __init__(self, CHANNELS = 12, CHUNK = 9600, RATE = 96000,hz_lims_low = 4000, hz_lims_high = 6000 , audiocard = 'STUDIO-CAPTURE', vsound = 343):
        #self.FORMAT = pyaudio.paInt16
        self.CHANNELS  = CHANNELS 
        self.CHUNK = CHUNK
        self.RATE = RATE
        self.duration = self.CHUNK / self.RATE
        self.audiocard = audiocard
        self.vsound = vsound
        self.hz_lims_low = hz_lims_low
        self.hz_lims_high = hz_lims_high
        self.q = queue.Queue()
        self.data = []

        rclpy.init()
        self.node = SoundAnglePublisher()
        




    def _correlate_signals(self, s):
        diffs = []
        s = s[:,1:] - s[:,:-1]
        sfft = np.fft.fft(s)
        component_lim_low = int(self.hz_lims_low*self.duration)
        component_lim_high = int(self.hz_lims_high*self.duration)
        sfft[:,:component_lim_low+1] = 0
        sfft[:,sfft.shape[-1] - component_lim_low:] = 0
        sfft[:,component_lim_high+1:sfft.shape[-1] - component_lim_high] = 0
        local_diff = []

        for i in range(self.CHANNELS):
            for j in range(i+1,self.CHANNELS):
                temp = sfft[i] * sfft[j].conj()
                temp = temp/(np.abs(temp) + 1e-5)
                re = np.fft.fftshift(np.fft.ifft(temp).real)
                local_diff.append(self.vsound*(re.argmax() - self.duration*self.RATE/2)/self.RATE) 
        return torch.tensor(local_diff)




    def _doa_from_correlates(self, a):
        theta = torch.linspace(-torch.pi,torch.pi,300).unsqueeze(1)
        phi = torch.linspace(0, torch.pi/2,200).unsqueeze(0)
        temp = torch.stack([torch.cos(phi)*torch.cos(theta), torch.cos(phi)*torch.sin(theta), torch.sin(phi)*torch.ones_like(theta)],dim=-1)
        projs = temp@(self.rec.T)
        temp = torch.cat([torch.stack([(projs[:,:,j] - projs[:,:,i]) for j in range(i+1,self.CHANNELS)]) for i in range(self.CHANNELS-1)])
        temp = temp - a.unsqueeze(1).unsqueeze(1)
        res_of_angle = temp.abs().median(0).values
        best_theta_i = res_of_angle.min(dim=1).values.argmin()
        best_phi_i = res_of_angle.min(dim=0).values.argmin()
        angle_est = torch.tensor([theta[best_theta_i,0], phi[0,best_phi_i]])
        return angle_est
    


    def _two_sol(self,r1,t1,r2,t2,ep1,ep2):
        r1x = r1[:,0]
        r1y = r1[:,1]
        r1z = r1[:,2]
        r2x = r2[:,0]
        r2y = r2[:,1]
        r2z = r2[:,2]
        r1x2 = r1x*r1x
        r1y2 = r1y*r1y
        r1z2 = r1z*r1z
        r2x2 = r2x*r2x
        r2y2 = r2y*r2y
        r2z2 = r2z*r2z
        ep12 = ep1*ep1
        ep22 = ep2*ep2
        t12 = t1*t1
        t22 = t2*t2

        f1 = (- ep12*r2x2 - ep12*r2y2 - ep12*r2z2 + 2*ep1*ep2*r1x*r2x + 2*ep1*ep2*r1y*r2y + 2*ep1*ep2*r1z*r2z + 2*ep1*r1x*r2x*t2 - 2*ep1*r2x2*t1 + 2*ep1*r1y*r2y*t2 - 2*ep1*r2y2*t1 + 2*ep1*r1z*r2z*t2 - 2*ep1*r2z2*t1 - ep22*r1x2 - ep22*r1y2 - ep22*r1z2 - 2*ep2*r1x2*t2 + 2*ep2*r1x*r2x*t1 - 2*ep2*r1y2*t2 + 2*ep2*r1y*r2y*t1 - 2*ep2*r1z2*t2 + 2*ep2*r1z*r2z*t1 + r1x2*r2y2 + r1x2*r2z2 - r1x2*t22 - 2*r1x*r2x*r1y*r2y - 2*r1x*r2x*r1z*r2z + 2*r1x*r2x*t1*t2 + r2x2*r1y2 + r2x2*r1z2 - r2x2*t12 + r1y2*r2z2 - r1y2*t22 - 2*r1y*r2y*r1z*r2z + 2*r1y*r2y*t1*t2 + r2y2*r1z2 - r2y2*t12 - r1z2*t22 + 2*r1z*r2z*t1*t2 - r2z2*t12).abs()
        
        q11 = (ep1*r2y - ep2*r1y - r1y*t2 + r2y*t1)/(r1x*r2y - r2x*r1y) + ((r1y*r2z - r2y*r1z)*(ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 + r1x*r2y*torch.sqrt(f1) - r2x*r1y*torch.sqrt(f1) - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1))/((r1x*r2y - r2x*r1y)*(r1x2*r2y2 + r1x2*r2z2 - 2*r1x*r2x*r1y*r2y - 2*r1x*r2x*r1z*r2z + r2x2*r1y2 + r2x2*r1z2 + r1y2*r2z2 - 2*r1y*r2y*r1z*r2z + r2y2*r1z2))
        q12 = (ep1*r2y - ep2*r1y - r1y*t2 + r2y*t1)/(r1x*r2y - r2x*r1y) + ((r1y*r2z - r2y*r1z)*(ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 - r1x*r2y*torch.sqrt(f1) + r2x*r1y*torch.sqrt(f1) - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1))/((r1x*r2y - r2x*r1y)*(r1x2*r2y2 + r1x2*r2z2 - 2*r1x*r2x*r1y*r2y - 2*r1x*r2x*r1z*r2z + r2x2*r1y2 + r2x2*r1z2 + r1y2*r2z2 - 2*r1y*r2y*r1z*r2z + r2y2*r1z2))
        q21 = - (ep1*r2x - ep2*r1x - r1x*t2 + r2x*t1)/(r1x*r2y - r2x*r1y) - ((r1x*r2z - r2x*r1z)*(ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 + r1x*r2y*torch.sqrt(f1) - r2x*r1y*torch.sqrt(f1) - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1))/((r1x*r2y - r2x*r1y)*(r1x2*r2y2 + r1x2*r2z2 - 2*r1x*r2x*r1y*r2y - 2*r1x*r2x*r1z*r2z + r2x2*r1y2 + r2x2*r1z2 + r1y2*r2z2 - 2*r1y*r2y*r1z*r2z + r2y2*r1z2))
        q22 = - (ep1*r2x - ep2*r1x - r1x*t2 + r2x*t1)/(r1x*r2y - r2x*r1y) - ((r1x*r2z - r2x*r1z)*(ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 - r1x*r2y*torch.sqrt(f1) + r2x*r1y*torch.sqrt(f1) - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1))/((r1x*r2y - r2x*r1y)*(r1x2*r2y2 + r1x2*r2z2 - 2*r1x*r2x*r1y*r2y - 2*r1x*r2x*r1z*r2z + r2x2*r1y2 + r2x2*r1z2 + r1y2*r2z2 - 2*r1y*r2y*r1z*r2z + r2y2*r1z2))
        q31 = (ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 + r1x*r2y*torch.sqrt(f1) - r2x*r1y*torch.sqrt(f1) - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1)/(r1x2*r2y2 + r1x2*r2z2 - 2*r1x*r2x*r1y*r2y - 2*r1x*r2x*r1z*r2z + r2x2*r1y2 + r2x2*r1z2 + r1y2*r2z2 - 2*r1y*r2y*r1z*r2z + r2y2*r1z2)
        q32 = (ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 - r1x*r2y*torch.sqrt(f1) + r2x*r1y*torch.sqrt(f1) - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1)/(r1x2*r2y2 + r1x2*r2z2 - 2*r1x*r2x*r1y*r2y - 2*r1x*r2x*r1z*r2z + r2x2*r1y2 + r2x2*r1z2 + r1y2*r2z2 - 2*r1y*r2y*r1z*r2z + r2y2*r1z2)
        sol1 = torch.stack([q11,q21,q31],1)
        sol2 = torch.stack([q12,q22,q32],1)
    
        return sol1,sol2
 

    def _two_sol4(self,r1,t1,r2,t2,ep):
        sol1a,sol2a =  self._two_sol(r1,t1,r2,t2,ep,ep)
        sol1b,sol2b =  self._two_sol(r1,t1,r2,t2,-ep,ep)
        sol1c,sol2c =  self._two_sol(r1,t1,r2,t2,ep,-ep)
        sol1d,sol2d =  self._two_sol(r1,t1,r2,t2,-ep,-ep)
        sol = torch.vstack([sol1a, sol2a,sol1b, sol2b,sol1c, sol2c,sol1d, sol2d])
        return sol
    

    def _least_square(self,delta_r,tau):
        A = torch.stack([(delta_r[i,:].unsqueeze(0).T @ delta_r[i,:].unsqueeze(0)) for i in range(delta_r.shape[0])]).sum(0)
        b = (tau*delta_r).sum(0)
        a11 = A[0,0]
        a12 = A[0,1]
        a13 = A[0,2]
        a21 = A[1,0]
        a22 = A[1,1]
        a23 = A[1,2]
        a31 = A[2,0]
        a32 = A[2,1]
        a33 = A[2,2]
        b1 = b[0]
        b2 = b[1]
        b3 = b[2]
        a11_2 = a11*a11
        a12_2 = a12*a12
        a13_2 = a13*a13
        a21_2 = a21*a21
        a22_2 = a22*a22
        a23_2 = a23*a23
        a31_2 = a31*a31
        a32_2 = a32*a32
        a33_2 = a33*a33
        b1_2 = b1*b1
        b2_2 = b2*b2
        b3_2 = b3*b3
        p2 = a11_2*a22_2 - 2*a11_2*a22*a23 - 2*a11_2*a22*a32 + 2*a11_2*a22*a33 + a11_2*a23_2 + 2*a11_2*a23*a32 - 2*a11_2*a23*a33 + a11_2*a32_2 - 2*a11_2*a32*a33 + a11_2*a33_2 - 2*a11_2*b2_2 + 4*a11_2*b2*b3 - 2*a11_2*b3_2 - 2*a11*a12*a21*a22 + 2*a11*a12*a21*a23 + 2*a11*a12*a21*a32 - 2*a11*a12*a21*a33 + 2*a11*a12*a22*a23 + 2*a11*a12*a22*a31 - 2*a11*a12*a22*a33 - 2*a11*a12*a23_2 - 2*a11*a12*a23*a31 - 2*a11*a12*a23*a32 + 4*a11*a12*a23*a33 - 2*a11*a12*a31*a32 + 2*a11*a12*a31*a33 + 2*a11*a12*a32*a33 - 2*a11*a12*a33_2 + 2*a11*a12*b2_2 - 4*a11*a12*b2*b3 + 2*a11*a12*b3_2 + 2*a11*a13*a21*a22 - 2*a11*a13*a21*a23 - 2*a11*a13*a21*a32 + 2*a11*a13*a21*a33 - 2*a11*a13*a22_2 + 2*a11*a13*a22*a23 - 2*a11*a13*a22*a31 + 4*a11*a13*a22*a32 - 2*a11*a13*a22*a33 + 2*a11*a13*a23*a31 - 2*a11*a13*a23*a32 + 2*a11*a13*a31*a32 - 2*a11*a13*a31*a33 - 2*a11*a13*a32_2 + 2*a11*a13*a32*a33 + 2*a11*a13*b2_2 - 4*a11*a13*b2*b3 + 2*a11*a13*b3_2 + 2*a11*a21*a22*a32 - 2*a11*a21*a22*a33 - 2*a11*a21*a23*a32 + 2*a11*a21*a23*a33 - 2*a11*a21*a32_2 + 4*a11*a21*a32*a33 - 2*a11*a21*a33_2 + 4*a11*a21*b1*b2 - 4*a11*a21*b1*b3 - 4*a11*a21*b2*b3 + 4*a11*a21*b3_2 - 2*a11*a22_2*a31 + 2*a11*a22_2*a33 + 4*a11*a22*a23*a31 - 2*a11*a22*a23*a32 - 2*a11*a22*a23*a33 + 2*a11*a22*a31*a32 - 2*a11*a22*a31*a33 - 2*a11*a22*a32*a33 + 2*a11*a22*a33_2 - 2*a11*a22*b1*b2 + 2*a11*a22*b1*b3 + 2*a11*a22*b2*b3 - 2*a11*a22*b3_2 - 2*a11*a23_2*a31 + 2*a11*a23_2*a32 - 2*a11*a23*a31*a32 + 2*a11*a23*a31*a33 + 2*a11*a23*a32_2 - 2*a11*a23*a32*a33 - 2*a11*a23*b1*b2 + 2*a11*a23*b1*b3 + 2*a11*a23*b2*b3 - 2*a11*a23*b3_2 - 4*a11*a31*b1*b2 + 4*a11*a31*b1*b3 + 4*a11*a31*b2_2 - 4*a11*a31*b2*b3 + 2*a11*a32*b1*b2 - 2*a11*a32*b1*b3 - 2*a11*a32*b2_2 + 2*a11*a32*b2*b3 + 2*a11*a33*b1*b2 - 2*a11*a33*b1*b3 - 2*a11*a33*b2_2 + 2*a11*a33*b2*b3 + a12_2*a21_2 - 2*a12_2*a21*a23 - 2*a12_2*a21*a31 + 2*a12_2*a21*a33 + a12_2*a23_2 + 2*a12_2*a23*a31 - 2*a12_2*a23*a33 + a12_2*a31_2 - 2*a12_2*a31*a33 + a12_2*a33_2 - 2*a12_2*b2_2 + 4*a12_2*b2*b3 - 2*a12_2*b3_2 - 2*a12*a13*a21_2 + 2*a12*a13*a21*a22 + 2*a12*a13*a21*a23 + 4*a12*a13*a21*a31 - 2*a12*a13*a21*a32 - 2*a12*a13*a21*a33 - 2*a12*a13*a22*a23 - 2*a12*a13*a22*a31 + 2*a12*a13*a22*a33 - 2*a12*a13*a23*a31 + 2*a12*a13*a23*a32 - 2*a12*a13*a31_2 + 2*a12*a13*a31*a32 + 2*a12*a13*a31*a33 - 2*a12*a13*a32*a33 + 2*a12*a13*b2_2 - 4*a12*a13*b2*b3 + 2*a12*a13*b3_2 - 2*a12*a21_2*a32 + 2*a12*a21_2*a33 + 2*a12*a21*a22*a31 - 2*a12*a21*a22*a33 - 2*a12*a21*a23*a31 + 4*a12*a21*a23*a32 - 2*a12*a21*a23*a33 + 2*a12*a21*a31*a32 - 2*a12*a21*a31*a33 - 2*a12*a21*a32*a33 + 2*a12*a21*a33_2 - 2*a12*a21*b1*b2 + 2*a12*a21*b1*b3 + 2*a12*a21*b2*b3 - 2*a12*a21*b3_2 - 2*a12*a22*a23*a31 + 2*a12*a22*a23*a33 - 2*a12*a22*a31_2 + 4*a12*a22*a31*a33 - 2*a12*a22*a33_2 + 4*a12*a22*b1*b2 - 4*a12*a22*b1*b3 - 4*a12*a22*b2*b3 + 4*a12*a22*b3_2 + 2*a12*a23_2*a31 - 2*a12*a23_2*a32 + 2*a12*a23*a31_2 - 2*a12*a23*a31*a32 - 2*a12*a23*a31*a33 + 2*a12*a23*a32*a33 - 2*a12*a23*b1*b2 + 2*a12*a23*b1*b3 + 2*a12*a23*b2*b3 - 2*a12*a23*b3_2 + 2*a12*a31*b1*b2 - 2*a12*a31*b1*b3 - 2*a12*a31*b2_2 + 2*a12*a31*b2*b3 - 4*a12*a32*b1*b2 + 4*a12*a32*b1*b3 + 4*a12*a32*b2_2 - 4*a12*a32*b2*b3 + 2*a12*a33*b1*b2 - 2*a12*a33*b1*b3 - 2*a12*a33*b2_2 + 2*a12*a33*b2*b3 + a13_2*a21_2 - 2*a13_2*a21*a22 - 2*a13_2*a21*a31 + 2*a13_2*a21*a32 + a13_2*a22_2 + 2*a13_2*a22*a31 - 2*a13_2*a22*a32 + a13_2*a31_2 - 2*a13_2*a31*a32 + a13_2*a32_2 - 2*a13_2*b2_2 + 4*a13_2*b2*b3 - 2*a13_2*b3_2 + 2*a13*a21_2*a32 - 2*a13*a21_2*a33 - 2*a13*a21*a22*a31 - 2*a13*a21*a22*a32 + 4*a13*a21*a22*a33 + 2*a13*a21*a23*a31 - 2*a13*a21*a23*a32 - 2*a13*a21*a31*a32 + 2*a13*a21*a31*a33 + 2*a13*a21*a32_2 - 2*a13*a21*a32*a33 - 2*a13*a21*b1*b2 + 2*a13*a21*b1*b3 + 2*a13*a21*b2*b3 - 2*a13*a21*b3_2 + 2*a13*a22_2*a31 - 2*a13*a22_2*a33 - 2*a13*a22*a23*a31 + 2*a13*a22*a23*a32 + 2*a13*a22*a31_2 - 2*a13*a22*a31*a32 - 2*a13*a22*a31*a33 + 2*a13*a22*a32*a33 - 2*a13*a22*b1*b2 + 2*a13*a22*b1*b3 + 2*a13*a22*b2*b3 - 2*a13*a22*b3_2 - 2*a13*a23*a31_2 + 4*a13*a23*a31*a32 - 2*a13*a23*a32_2 + 4*a13*a23*b1*b2 - 4*a13*a23*b1*b3 - 4*a13*a23*b2*b3 + 4*a13*a23*b3_2 + 2*a13*a31*b1*b2 - 2*a13*a31*b1*b3 - 2*a13*a31*b2_2 + 2*a13*a31*b2*b3 + 2*a13*a32*b1*b2 - 2*a13*a32*b1*b3 - 2*a13*a32*b2_2 + 2*a13*a32*b2*b3 - 4*a13*a33*b1*b2 + 4*a13*a33*b1*b3 + 4*a13*a33*b2_2 - 4*a13*a33*b2*b3 + a21_2*a32_2 - 2*a21_2*a32*a33 + a21_2*a33_2 - 2*a21_2*b1_2 + 4*a21_2*b1*b3 - 2*a21_2*b3_2 - 2*a21*a22*a31*a32 + 2*a21*a22*a31*a33 + 2*a21*a22*a32*a33 - 2*a21*a22*a33_2 + 2*a21*a22*b1_2 - 4*a21*a22*b1*b3 + 2*a21*a22*b3_2 + 2*a21*a23*a31*a32 - 2*a21*a23*a31*a33 - 2*a21*a23*a32_2 + 2*a21*a23*a32*a33 + 2*a21*a23*b1_2 - 4*a21*a23*b1*b3 + 2*a21*a23*b3_2 + 4*a21*a31*b1_2 - 4*a21*a31*b1*b2 - 4*a21*a31*b1*b3 + 4*a21*a31*b2*b3 - 2*a21*a32*b1_2 + 2*a21*a32*b1*b2 + 2*a21*a32*b1*b3 - 2*a21*a32*b2*b3 - 2*a21*a33*b1_2 + 2*a21*a33*b1*b2 + 2*a21*a33*b1*b3 - 2*a21*a33*b2*b3 + a22_2*a31_2 - 2*a22_2*a31*a33 + a22_2*a33_2 - 2*a22_2*b1_2 + 4*a22_2*b1*b3 - 2*a22_2*b3_2 - 2*a22*a23*a31_2 + 2*a22*a23*a31*a32 + 2*a22*a23*a31*a33 - 2*a22*a23*a32*a33 + 2*a22*a23*b1_2 - 4*a22*a23*b1*b3 + 2*a22*a23*b3_2 - 2*a22*a31*b1_2 + 2*a22*a31*b1*b2 + 2*a22*a31*b1*b3 - 2*a22*a31*b2*b3 + 4*a22*a32*b1_2 - 4*a22*a32*b1*b2 - 4*a22*a32*b1*b3 + 4*a22*a32*b2*b3 - 2*a22*a33*b1_2 + 2*a22*a33*b1*b2 + 2*a22*a33*b1*b3 - 2*a22*a33*b2*b3 + a23_2*a31_2 - 2*a23_2*a31*a32 + a23_2*a32_2 - 2*a23_2*b1_2 + 4*a23_2*b1*b3 - 2*a23_2*b3_2 - 2*a23*a31*b1_2 + 2*a23*a31*b1*b2 + 2*a23*a31*b1*b3 - 2*a23*a31*b2*b3 - 2*a23*a32*b1_2 + 2*a23*a32*b1*b2 + 2*a23*a32*b1*b3 - 2*a23*a32*b2*b3 + 4*a23*a33*b1_2 - 4*a23*a33*b1*b2 - 4*a23*a33*b1*b3 + 4*a23*a33*b2*b3 - 2*a31_2*b1_2 + 4*a31_2*b1*b2 - 2*a31_2*b2_2 + 2*a31*a32*b1_2 - 4*a31*a32*b1*b2 + 2*a31*a32*b2_2 + 2*a31*a33*b1_2 - 4*a31*a33*b1*b2 + 2*a31*a33*b2_2 - 2*a32_2*b1_2 + 4*a32_2*b1*b2 - 2*a32_2*b2_2 + 2*a32*a33*b1_2 - 4*a32*a33*b1*b2 + 2*a32*a33*b2_2 - 2*a33_2*b1_2 + 4*a33_2*b1*b2 - 2*a33_2*b2_2
        p1 = 2*a11_2*a22_2*a33 - 2*a11_2*a22*a23*a32 - 2*a11_2*a22*a23*a33 - 2*a11_2*a22*a32*a33 + 2*a11_2*a22*a33_2 + 2*a11_2*a22*b2*b3 - 2*a11_2*a22*b3_2 + 2*a11_2*a23_2*a32 + 2*a11_2*a23*a32_2 - 2*a11_2*a23*a32*a33 + 2*a11_2*a23*b2*b3 - 2*a11_2*a23*b3_2 - 2*a11_2*a32*b2_2 + 2*a11_2*a32*b2*b3 - 2*a11_2*a33*b2_2 + 2*a11_2*a33*b2*b3 - 4*a11*a12*a21*a22*a33 + 2*a11*a12*a21*a23*a32 + 2*a11*a12*a21*a23*a33 + 2*a11*a12*a21*a32*a33 - 2*a11*a12*a21*a33_2 - 2*a11*a12*a21*b2*b3 + 2*a11*a12*a21*b3_2 + 2*a11*a12*a22*a23*a31 + 2*a11*a12*a22*a23*a33 + 2*a11*a12*a22*a31*a33 - 2*a11*a12*a22*a33_2 - 2*a11*a12*a22*b2*b3 + 2*a11*a12*a22*b3_2 - 2*a11*a12*a23_2*a31 - 2*a11*a12*a23_2*a32 - 4*a11*a12*a23*a31*a32 + 2*a11*a12*a23*a31*a33 + 2*a11*a12*a23*a32*a33 + 2*a11*a12*a31*b2_2 - 2*a11*a12*a31*b2*b3 + 2*a11*a12*a32*b2_2 - 2*a11*a12*a32*b2*b3 + 2*a11*a13*a21*a22*a32 + 2*a11*a13*a21*a22*a33 - 4*a11*a13*a21*a23*a32 - 2*a11*a13*a21*a32_2 + 2*a11*a13*a21*a32*a33 - 2*a11*a13*a21*b2*b3 + 2*a11*a13*a21*b3_2 - 2*a11*a13*a22_2*a31 - 2*a11*a13*a22_2*a33 + 2*a11*a13*a22*a23*a31 + 2*a11*a13*a22*a23*a32 + 2*a11*a13*a22*a31*a32 - 4*a11*a13*a22*a31*a33 + 2*a11*a13*a22*a32*a33 + 2*a11*a13*a23*a31*a32 - 2*a11*a13*a23*a32_2 - 2*a11*a13*a23*b2*b3 + 2*a11*a13*a23*b3_2 + 2*a11*a13*a31*b2_2 - 2*a11*a13*a31*b2*b3 + 2*a11*a13*a33*b2_2 - 2*a11*a13*a33*b2*b3 + 2*a11*a21*a22*a32*a33 - 2*a11*a21*a22*a33_2 - 2*a11*a21*a22*b1*b3 + 2*a11*a21*a22*b3_2 - 2*a11*a21*a23*a32_2 + 2*a11*a21*a23*a32*a33 - 2*a11*a21*a23*b1*b3 + 2*a11*a21*a23*b3_2 + 4*a11*a21*a32*b1*b2 - 2*a11*a21*a32*b1*b3 - 2*a11*a21*a32*b2*b3 + 4*a11*a21*a33*b1*b2 - 2*a11*a21*a33*b1*b3 - 2*a11*a21*a33*b2*b3 - 2*a11*a22_2*a31*a33 + 2*a11*a22_2*a33_2 + 2*a11*a22_2*b1*b3 - 2*a11*a22_2*b3_2 + 2*a11*a22*a23*a31*a32 + 2*a11*a22*a23*a31*a33 - 4*a11*a22*a23*a32*a33 - 2*a11*a22*a31*b1*b2 + 4*a11*a22*a31*b1*b3 - 2*a11*a22*a31*b2*b3 - 2*a11*a22*a32*b1*b2 - 2*a11*a22*a32*b1*b3 + 4*a11*a22*a32*b2*b3 - 2*a11*a23_2*a31*a32 + 2*a11*a23_2*a32_2 + 2*a11*a23_2*b1*b3 - 2*a11*a23_2*b3_2 - 2*a11*a23*a31*b1*b2 + 4*a11*a23*a31*b1*b3 - 2*a11*a23*a31*b2*b3 - 2*a11*a23*a33*b1*b2 - 2*a11*a23*a33*b1*b3 + 4*a11*a23*a33*b2*b3 - 2*a11*a31*a32*b1*b2 + 2*a11*a31*a32*b2_2 - 2*a11*a31*a33*b1*b2 + 2*a11*a31*a33*b2_2 + 2*a11*a32_2*b1*b2 - 2*a11*a32_2*b2_2 + 2*a11*a33_2*b1*b2 - 2*a11*a33_2*b2_2 + 2*a12_2*a21_2*a33 - 2*a12_2*a21*a23*a31 - 2*a12_2*a21*a23*a33 - 2*a12_2*a21*a31*a33 + 2*a12_2*a21*a33_2 + 2*a12_2*a21*b2*b3 - 2*a12_2*a21*b3_2 + 2*a12_2*a23_2*a31 + 2*a12_2*a23*a31_2 - 2*a12_2*a23*a31*a33 + 2*a12_2*a23*b2*b3 - 2*a12_2*a23*b3_2 - 2*a12_2*a31*b2_2 + 2*a12_2*a31*b2*b3 - 2*a12_2*a33*b2_2 + 2*a12_2*a33*b2*b3 - 2*a12*a13*a21_2*a32 - 2*a12*a13*a21_2*a33 + 2*a12*a13*a21*a22*a31 + 2*a12*a13*a21*a22*a33 + 2*a12*a13*a21*a23*a31 + 2*a12*a13*a21*a23*a32 + 2*a12*a13*a21*a31*a32 + 2*a12*a13*a21*a31*a33 - 4*a12*a13*a21*a32*a33 - 4*a12*a13*a22*a23*a31 - 2*a12*a13*a22*a31_2 + 2*a12*a13*a22*a31*a33 - 2*a12*a13*a22*b2*b3 + 2*a12*a13*a22*b3_2 - 2*a12*a13*a23*a31_2 + 2*a12*a13*a23*a31*a32 - 2*a12*a13*a23*b2*b3 + 2*a12*a13*a23*b3_2 + 2*a12*a13*a32*b2_2 - 2*a12*a13*a32*b2*b3 + 2*a12*a13*a33*b2_2 - 2*a12*a13*a33*b2*b3 - 2*a12*a21_2*a32*a33 + 2*a12*a21_2*a33_2 + 2*a12*a21_2*b1*b3 - 2*a12*a21_2*b3_2 + 2*a12*a21*a22*a31*a33 - 2*a12*a21*a22*a33_2 - 2*a12*a21*a22*b1*b3 + 2*a12*a21*a22*b3_2 + 2*a12*a21*a23*a31*a32 - 4*a12*a21*a23*a31*a33 + 2*a12*a21*a23*a32*a33 - 2*a12*a21*a31*b1*b2 - 2*a12*a21*a31*b1*b3 + 4*a12*a21*a31*b2*b3 - 2*a12*a21*a32*b1*b2 + 4*a12*a21*a32*b1*b3 - 2*a12*a21*a32*b2*b3 - 2*a12*a22*a23*a31_2 + 2*a12*a22*a23*a31*a33 - 2*a12*a22*a23*b1*b3 + 2*a12*a22*a23*b3_2 + 4*a12*a22*a31*b1*b2 - 2*a12*a22*a31*b1*b3 - 2*a12*a22*a31*b2*b3 + 4*a12*a22*a33*b1*b2 - 2*a12*a22*a33*b1*b3 - 2*a12*a22*a33*b2*b3 + 2*a12*a23_2*a31_2 - 2*a12*a23_2*a31*a32 + 2*a12*a23_2*b1*b3 - 2*a12*a23_2*b3_2 - 2*a12*a23*a32*b1*b2 + 4*a12*a23*a32*b1*b3 - 2*a12*a23*a32*b2*b3 - 2*a12*a23*a33*b1*b2 - 2*a12*a23*a33*b1*b3 + 4*a12*a23*a33*b2*b3 + 2*a12*a31_2*b1*b2 - 2*a12*a31_2*b2_2 - 2*a12*a31*a32*b1*b2 + 2*a12*a31*a32*b2_2 - 2*a12*a32*a33*b1*b2 + 2*a12*a32*a33*b2_2 + 2*a12*a33_2*b1*b2 - 2*a12*a33_2*b2_2 + 2*a13_2*a21_2*a32 - 2*a13_2*a21*a22*a31 - 2*a13_2*a21*a22*a32 - 2*a13_2*a21*a31*a32 + 2*a13_2*a21*a32_2 + 2*a13_2*a21*b2*b3 - 2*a13_2*a21*b3_2 + 2*a13_2*a22_2*a31 + 2*a13_2*a22*a31_2 - 2*a13_2*a22*a31*a32 + 2*a13_2*a22*b2*b3 - 2*a13_2*a22*b3_2 - 2*a13_2*a31*b2_2 + 2*a13_2*a31*b2*b3 - 2*a13_2*a32*b2_2 + 2*a13_2*a32*b2*b3 + 2*a13*a21_2*a32_2 - 2*a13*a21_2*a32*a33 + 2*a13*a21_2*b1*b3 - 2*a13*a21_2*b3_2 - 4*a13*a21*a22*a31*a32 + 2*a13*a21*a22*a31*a33 + 2*a13*a21*a22*a32*a33 + 2*a13*a21*a23*a31*a32 - 2*a13*a21*a23*a32_2 - 2*a13*a21*a23*b1*b3 + 2*a13*a21*a23*b3_2 - 2*a13*a21*a31*b1*b2 - 2*a13*a21*a31*b1*b3 + 4*a13*a21*a31*b2*b3 - 2*a13*a21*a33*b1*b2 + 4*a13*a21*a33*b1*b3 - 2*a13*a21*a33*b2*b3 + 2*a13*a22_2*a31_2 - 2*a13*a22_2*a31*a33 + 2*a13*a22_2*b1*b3 - 2*a13*a22_2*b3_2 - 2*a13*a22*a23*a31_2 + 2*a13*a22*a23*a31*a32 - 2*a13*a22*a23*b1*b3 + 2*a13*a22*a23*b3_2 - 2*a13*a22*a32*b1*b2 - 2*a13*a22*a32*b1*b3 + 4*a13*a22*a32*b2*b3 - 2*a13*a22*a33*b1*b2 + 4*a13*a22*a33*b1*b3 - 2*a13*a22*a33*b2*b3 + 4*a13*a23*a31*b1*b2 - 2*a13*a23*a31*b1*b3 - 2*a13*a23*a31*b2*b3 + 4*a13*a23*a32*b1*b2 - 2*a13*a23*a32*b1*b3 - 2*a13*a23*a32*b2*b3 + 2*a13*a31_2*b1*b2 - 2*a13*a31_2*b2_2 - 2*a13*a31*a33*b1*b2 + 2*a13*a31*a33*b2_2 + 2*a13*a32_2*b1*b2 - 2*a13*a32_2*b2_2 - 2*a13*a32*a33*b1*b2 + 2*a13*a32*a33*b2_2 - 2*a21_2*a32*b1_2 + 2*a21_2*a32*b1*b3 - 2*a21_2*a33*b1_2 + 2*a21_2*a33*b1*b3 + 2*a21*a22*a31*b1_2 - 2*a21*a22*a31*b1*b3 + 2*a21*a22*a32*b1_2 - 2*a21*a22*a32*b1*b3 + 2*a21*a23*a31*b1_2 - 2*a21*a23*a31*b1*b3 + 2*a21*a23*a33*b1_2 - 2*a21*a23*a33*b1*b3 + 2*a21*a31*a32*b1_2 - 2*a21*a31*a32*b1*b2 + 2*a21*a31*a33*b1_2 - 2*a21*a31*a33*b1*b2 - 2*a21*a32_2*b1_2 + 2*a21*a32_2*b1*b2 - 2*a21*a33_2*b1_2 + 2*a21*a33_2*b1*b2 - 2*a22_2*a31*b1_2 + 2*a22_2*a31*b1*b3 - 2*a22_2*a33*b1_2 + 2*a22_2*a33*b1*b3 + 2*a22*a23*a32*b1_2 - 2*a22*a23*a32*b1*b3 + 2*a22*a23*a33*b1_2 - 2*a22*a23*a33*b1*b3 - 2*a22*a31_2*b1_2 + 2*a22*a31_2*b1*b2 + 2*a22*a31*a32*b1_2 - 2*a22*a31*a32*b1*b2 + 2*a22*a32*a33*b1_2 - 2*a22*a32*a33*b1*b2 - 2*a22*a33_2*b1_2 + 2*a22*a33_2*b1*b2 - 2*a23_2*a31*b1_2 + 2*a23_2*a31*b1*b3 - 2*a23_2*a32*b1_2 + 2*a23_2*a32*b1*b3 - 2*a23*a31_2*b1_2 + 2*a23*a31_2*b1*b2 + 2*a23*a31*a33*b1_2 - 2*a23*a31*a33*b1*b2 - 2*a23*a32_2*b1_2 + 2*a23*a32_2*b1*b2 + 2*a23*a32*a33*b1_2 - 2*a23*a32*a33*b1*b2
        p0 = a11_2*a22_2*a33_2 - a11_2*a22_2*b3_2 - 2*a11_2*a22*a23*a32*a33 + 2*a11_2*a22*a32*b2*b3 + a11_2*a23_2*a32_2 - a11_2*a23_2*b3_2 + 2*a11_2*a23*a33*b2*b3 - a11_2*a32_2*b2_2 - a11_2*a33_2*b2_2 - 2*a11*a12*a21*a22*a33_2 + 2*a11*a12*a21*a22*b3_2 + 2*a11*a12*a21*a23*a32*a33 - 2*a11*a12*a21*a32*b2*b3 + 2*a11*a12*a22*a23*a31*a33 - 2*a11*a12*a22*a31*b2*b3 - 2*a11*a12*a23_2*a31*a32 + 2*a11*a12*a31*a32*b2_2 + 2*a11*a13*a21*a22*a32*a33 - 2*a11*a13*a21*a23*a32_2 + 2*a11*a13*a21*a23*b3_2 - 2*a11*a13*a21*a33*b2*b3 - 2*a11*a13*a22_2*a31*a33 + 2*a11*a13*a22*a23*a31*a32 - 2*a11*a13*a23*a31*b2*b3 + 2*a11*a13*a31*a33*b2_2 - 2*a11*a21*a22*a32*b1*b3 - 2*a11*a21*a23*a33*b1*b3 + 2*a11*a21*a32_2*b1*b2 + 2*a11*a21*a33_2*b1*b2 + 2*a11*a22_2*a31*b1*b3 - 2*a11*a22*a31*a32*b1*b2 + 2*a11*a23_2*a31*b1*b3 - 2*a11*a23*a31*a33*b1*b2 + a12_2*a21_2*a33_2 - a12_2*a21_2*b3_2 - 2*a12_2*a21*a23*a31*a33 + 2*a12_2*a21*a31*b2*b3 + a12_2*a23_2*a31_2 - a12_2*a23_2*b3_2 + 2*a12_2*a23*a33*b2*b3 - a12_2*a31_2*b2_2 - a12_2*a33_2*b2_2 - 2*a12*a13*a21_2*a32*a33 + 2*a12*a13*a21*a22*a31*a33 + 2*a12*a13*a21*a23*a31*a32 - 2*a12*a13*a22*a23*a31_2 + 2*a12*a13*a22*a23*b3_2 - 2*a12*a13*a22*a33*b2*b3 - 2*a12*a13*a23*a32*b2*b3 + 2*a12*a13*a32*a33*b2_2 + 2*a12*a21_2*a32*b1*b3 - 2*a12*a21*a22*a31*b1*b3 - 2*a12*a21*a31*a32*b1*b2 - 2*a12*a22*a23*a33*b1*b3 + 2*a12*a22*a31_2*b1*b2 + 2*a12*a22*a33_2*b1*b2 + 2*a12*a23_2*a32*b1*b3 - 2*a12*a23*a32*a33*b1*b2 + a13_2*a21_2*a32_2 - a13_2*a21_2*b3_2 - 2*a13_2*a21*a22*a31*a32 + 2*a13_2*a21*a31*b2*b3 + a13_2*a22_2*a31_2 - a13_2*a22_2*b3_2 + 2*a13_2*a22*a32*b2*b3 - a13_2*a31_2*b2_2 - a13_2*a32_2*b2_2 + 2*a13*a21_2*a33*b1*b3 - 2*a13*a21*a23*a31*b1*b3 - 2*a13*a21*a31*a33*b1*b2 + 2*a13*a22_2*a33*b1*b3 - 2*a13*a22*a23*a32*b1*b3 - 2*a13*a22*a32*a33*b1*b2 + 2*a13*a23*a31_2*b1*b2 + 2*a13*a23*a32_2*b1*b2 - a21_2*a32_2*b1_2 - a21_2*a33_2*b1_2 + 2*a21*a22*a31*a32*b1_2 + 2*a21*a23*a31*a33*b1_2 - a22_2*a31_2*b1_2 - a22_2*a33_2*b1_2 + 2*a22*a23*a32*a33*b1_2 - a23_2*a31_2*b1_2 - a23_2*a32_2*b1_2
        if (p1*p1-4*p0*p2) >= 0:
            lam1 = -(p1 - torch.sqrt(p1*p1 - 4*p0*p2))/(2*p2)
            lam2 = -(p1 + torch.sqrt(p1*p1 - 4*p0*p2))/(2*p2)
            v1 = torch.linalg.solve(A+lam1,b)
            v2 = torch.linalg.solve(A+lam2,b)
            res1 = delta_r@v1.unsqueeze(1)-tau
            res2 = delta_r@v2.unsqueeze(1)-tau
            min1 = np.linalg.norm(res1)
            min2 = np.linalg.norm(res2)
            return (v1,min1) if min1 < min2 else (v2,min2)
        else:
            return(0,-1)
            
  

    def _find_inlier_max_doa(self,a, ep = 0.5):
        
        sz = (self.recdiff.shape)[0]
        # ncmb = math.comb(sz,2)
        r1 = torch.cat([torch.stack([(self.recdiff[j,:]) for j in range(i+1,sz)]) for i in range(sz-1)])
        r2 = torch.cat([torch.stack([(self.recdiff[i,:]) for j in range(i+1,sz)]) for i in range(sz-1)])
        t1 = torch.cat([torch.stack([(a[j]) for j in range(i+1,sz)]) for i in range(sz-1)])
        t2 = torch.cat([torch.stack([(a[i]) for j in range(i+1,sz)]) for i in range(sz-1)])
        sol = self._two_sol4(r1,t1,r2,t2,ep)
        resok = ((sol@self.recdiff.T-a).abs()<=ep).sum(1)
        besti = resok.argmax()
        vv_est = sol[besti,:]     
        return vv_est
    

    def _find_inlier_max_doa_ls(self, a, ep = 0.5):
        
        sz = (self.recdiff.shape)[0]
        # ncmb = math.comb(sz,2)
        r1 = torch.cat([torch.stack([(self.recdiff[j,:]) for j in range(i+1,sz)]) for i in range(sz-1)])
        r2 = torch.cat([torch.stack([(self.recdiff[i,:]) for j in range(i+1,sz)]) for i in range(sz-1)])
        t1 = torch.cat([torch.stack([(a[j]) for j in range(i+1,sz)]) for i in range(sz-1)])
        t2 = torch.cat([torch.stack([(a[i]) for j in range(i+1,sz)]) for i in range(sz-1)])
        sol = self._two_sol4(r1,t1,r2,t2,ep)
        resok = ((sol@self.recdiff.T-a).abs()<=ep).sum(1)
        besti = resok.argmax()
        vv_est = sol[besti,:]     
        inliers = (vv_est@self.recdiff.T-a).abs()<=ep
        rd_in = self.recdiff[inliers,:]
        tau_in = a[inliers]
        vv_ls,mini = self._least_square(rd_in,tau_in.unsqueeze(1))
        if mini>0:
            return vv_ls
        else:
            return vv_est



    # callback function to stream audio, another thread.
    def callback(self,data,frames ,time, status):
        
        #s = np.frombuffer(data, dtype=np.int16).reshape((CHANNELS,CHUNK),order = 'F')
        #self.audio = np.frombuffer(data, dtype=np.int16)
        #s = self.audio.reshape((self.CHANNELS,self.CHUNK),order = 'F')
        s = data.T
        diffs = self._correlate_signals(s) 
        angle_est = self._doa_from_correlates(diffs)
        print(angle_est)
        self.node.publish_numbers(angle_est)
        rclpy.spin_once(self.node, timeout_sec=0.0)
        #self.q.put(data.copy())
        self.data.append(data.copy())


        #return (self.audio, pyaudio.paContinue)





    def run_doa_stream(self, device_index):

        # pyaudio_instance = pyaudio.PyAudio()
        # self.stream = pyaudio_instance.open(format=self.FORMAT,
        #         channels=self.CHANNELS,
        #         rate=self.RATE,
        #         input=True,
        #         input_device_index=device_index,
        #         frames_per_buffer=self.CHUNK,
        #         stream_callback = self._callback)


        # self.audio = np.empty((self.CHUNK),dtype="int16")
        # self.stream.start_stream()

        
        #with sf.SoundFile("New_Recording.wav", mode='x', samplerate=self.RATE,
        #              channels=self.CHANNELS) as file:
        with sd.InputStream(device=device_index, channels=CHANNELS, callback=self.callback,
                        blocksize=int(self.CHUNK),
                        samplerate=self.RATE):
            while True:
                try:
                    time.sleep(0.01)

                except KeyboardInterrupt:
                    temp = np.concatenate(self.data)
                    sp.io.wavfile.write("Recording_"+str(datetime.datetime.now())+".wav",self.RATE,temp)
                    self.node.destroy_node()
                    rclpy.shutdown()
                    return
        
CHANNELS = 6
mydoa = doa_streamer(CHANNELS)
idx = mydoa.get_soundcard_device()
if idx < 0:
    print('Soundcard device not found')
    sys.exit(1)

#rec = torch.rand((CHANNELS,3))
rec = torch.tensor([
    [0.0,0.0,0.0],
    [0.0,1.0,0.0],
    [0.0,2.0,0.0],
    [1.5,2.0,0.0],
    [1.5,1.0,0.0],
    [1.5,0.0,0.0],
    ])

mydoa.set_rec_positions(rec)

print("* recording")
mydoa.run_doa_stream(idx)
print("* done recording")






