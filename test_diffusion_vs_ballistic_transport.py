#-------------------------------------------------------------------------------
# Name:        Test_diffusion_vs_ballistic_transport
# Purpose:
#
# Author:      Wenjiao_Wang
#
# Created:     23/04/2014
# Copyright:   (c) Wenjiao_Wang 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
from Node_class_1 import Nodes
import time

def diffusion_pressure_profile(Sticking_coefficient, AR):    #calculate the ratio of pressure drop as a function of sticking coefficient and AR. Here the AR is the array of the trench depth.
    length=len(AR)
    result=np.zeros(len(AR))
    M=np.exp(-np.sqrt(1.5*Sticking_coefficient)*AR[length-1])*(-0.75*Sticking_coefficient+np.sqrt(1.5*Sticking_coefficient))
    N=np.exp(np.sqrt(1.5*Sticking_coefficient)*AR[length-1])*(0.75*Sticking_coefficient+np.sqrt(1.5*Sticking_coefficient))
    for i in range(0,length,1):
        result[i]=M/(M+N)*np.exp(np.sqrt(1.5*Sticking_coefficient)*AR[i])+N/(M+N)*np.exp(-np.sqrt(1.5*Sticking_coefficient)*AR[i])

    return result

#calculate pressure profile for different sticking coefficient and different aspect ratios.
AR=np.linspace(0,20,100)
pressure_1=diffusion_pressure_profile(1,AR)
pressure_0_5=diffusion_pressure_profile(0.5,AR)
pressure_0_1=diffusion_pressure_profile(0.1,AR)
pressure_0_01=diffusion_pressure_profile(0.01,AR)
pressure_0_001=diffusion_pressure_profile(0.001,AR)
pressure_0_0001=diffusion_pressure_profile(0.0001,AR)
plt.plot(AR,pressure_1,'-',AR,pressure_0_5,'-',AR,pressure_0_1,'-',AR,pressure_0_01,'-',AR,pressure_0_001,'-',AR,pressure_0_0001,'-')
#plt.yscale('log')
plt.show()

'''
Sc=np.zeros(len(AR))
Step_coverage=[0.5,0.9,0.99]
for j in range(0,len(Step_coverage),1):
    for i in range(0,len(AR),1):
        Sc[i]=(np.log(Step_coverage[j])*(-2.0/3)/(0.5*(AR[i])**2+AR[i]))
    if j==0:
        plt.plot(AR[4:99],Sc[4:99],'b-')
    elif j==1:
        plt.plot(AR[4:99],Sc[4:99],'r-')
    elif j==2:
        plt.plot(AR[4:99],Sc[4:99],'g-')

plt.yscale('log')
plt.xlim([4,100])

plt.show()
'''
'''
#the following is for ballistic transport model
Flux=1
Sc_surface=1.0/Flux
C_sticking=0.1  #defines a constant sticking coefficient
trench=Nodes(N=400,alpha=90.0,AR=10,Sc_surface=Sc_surface)
a0=3.0098*0.1*1e3

a=trench.direct_flux_distribution(a0)
time1=time.time()
a1=trench.stable_receiving_flux_vector_2(0,a0)
time2=time.time()
print time2-time1
trench_position=trench.center_positions
plt.plot(-trench_position.imag,a/a0,'bo')
plt.plot(-trench_position.imag,a1/a0,'ro')
plt.plot(AR,pressure_1,'-')
plt.show()
'''