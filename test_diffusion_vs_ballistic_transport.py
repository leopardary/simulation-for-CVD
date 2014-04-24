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
from Node_class_1 import Nodes
import time

def diffusion_pressure_profile(Sticking_coefficient, AR):    #calculate the ratio of pressure drop as a function of sticking coefficient and AR. Here the AR is the array of the trench depth.
    length=len(AR)
    result=np.zeros(len(AR))
    for i in range(0,length,1):
        result[i]=np.exp(-3.0/2*Sticking_coefficient*(0.5*AR[i]**2+AR[i]))

    return result

AR=np.linspace(0,10,100)
pressure_1=diffusion_pressure_profile(1,AR)
pressure_0_5=diffusion_pressure_profile(0.5,AR)
pressure_0_1=diffusion_pressure_profile(0.1,AR)
pressure_0_01=diffusion_pressure_profile(0.01,AR)
pressure_0_001=diffusion_pressure_profile(0.001,AR)
plt.plot(AR,pressure_1,'-',AR,pressure_0_5,'-',AR,pressure_0_1,'-',AR,pressure_0_01,'-',AR,pressure_0_001,'-',)
plt.show()

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
